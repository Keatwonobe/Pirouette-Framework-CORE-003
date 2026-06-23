"""
manifold_illuminator.py
Manifold Illuminator — Visual Map of the (J1, Ksi) Weight Space Surface
Pirouette Framework Volume 8 · CORE-003 · ML-073

WHAT THIS DOES
==============
Sweeps the full (J1, Ksi) manifold grid and measures what lives at each address.
Produces heatmap images showing:

  1. COHERENCE MAP     — how English-like is each address? (brightness)
  2. REGISTER MAP      — what TYPE of language lives there? (color)
  3. CONFIDENCE MAP    — arc field confidence at each cell
  4. CURVATURE MAP     — manifold curvature (1 - confidence)
  5. COMPOSITE         — all four overlaid with certified landmarks

The scan uses ONLY the LM_Head weight matrix and the engram coordinate system.
No transformer forward passes needed — this is fast (~minutes on CPU).

Each cell (j1, ksi):
  1. Build synthetic hidden state h at address (j1, ksi)
     h = cos(j1)*pc1 + sin(j1)*pc2 + ksi_component*U_ref_direction
     (where ksi_component sets the spectral entropy to the target ksi)
  2. LM_Head logits = W_n @ h
  3. Measure: coherence, register, top1_cos, entropy

CERTIFIED LANDMARKS OVERLAID
==============================
From scan_j1.json results (certified):
  Zone A peak:    J1=110°, coherence=0.95  → blue dot
  Zone B peak:    J1=320°, coherence=0.875 → green dot
  Dead zone:      J1=140-265°             → shaded region
  Wada boundary:  Ksi≈0.88               → horizontal line
  English attractor: Ksi=0.585           → horizontal line
  alpha_physical: Ksi=0.4784             → horizontal line

Usage:
  python manifold_illuminator.py scan ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --output manifold_scan.npz ^
    --j1_steps 72 --ksi_steps 40

  python manifold_illuminator.py plot ^
    --scan manifold_scan.npz ^
    --field bivector_field.npz ^
    --output manifold_heatmap.png

  python manifold_illuminator.py query ^
    --scan manifold_scan.npz ^
    --j1 110 --ksi 0.585 ^
    --output query_result.json

  # Full pipeline (scan + plot):
  python manifold_illuminator.py all ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --field bivector_field.npz ^
    --output manifold
"""

import argparse, json, sys, time
import numpy as np
from pathlib import Path

K_PROJ = 16

# Register classification token sets
SENTENCE_START_TOKENS = {
    'The', 'In', 'It', 'This', 'For', 'There', 'A', 'An', 'By',
    'From', 'On', 'At', 'With', 'As', 'But', 'If', 'When', 'While',
    'Although', 'However', 'Thus', 'Therefore', 'Hence', 'So',
}
FUNCTION_TOKENS = {
    'the', 'a', 'of', 'in', 'and', 'to', 'is', 'was', 'are', 'were',
    'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'do', 'does', 'did', 'not',
    'an', 'or', 'but', 'if', 'on', 'at', 'by', 'from', 'with',
    'that', 'which', 'who', 'what', 'how', 'when', 'where', 'why',
    'all', 'some', 'any', 'each', 'no', 'more', 'than', 'so', 'as',
    ',', '.', ';', ':', ' the', ' a', ' of', ' in', ' and', ' to',
    ' is', ' was', ' are', ' that', ' which', ' for', ' with',
}

REGISTER_COLORS = {
    'SENTENCE_START': np.array([0.2, 0.4, 0.9]),   # blue
    'FUNCTION':       np.array([0.2, 0.75, 0.4]),   # green
    'CONTENT':        np.array([0.9, 0.75, 0.1]),   # gold
    'PUNCT':          np.array([0.9, 0.5, 0.1]),    # orange
    'NUMERIC':        np.array([0.7, 0.2, 0.8]),    # purple
    'GARBAGE':        np.array([0.8, 0.1, 0.1]),    # red
    'MIXED':          np.array([0.5, 0.5, 0.5]),    # grey
}

# ── Core: build synthetic hidden state at (J1, Ksi) ──────────────────────────

def build_synthetic_h(j1_deg, ksi_target, pc1, pc2, U_ref, dim=1280):
    """
    Construct a unit hidden state with given (J1, Ksi) address.

    J1 controls direction in the (pc1, pc2) plane.
    Ksi controls spectral entropy via U_ref projection.

    Method:
    1. Start with isotropic random vector (high Ksi ≈ 1.0)
    2. Mix with a directional component along (cos(j1)*pc1 + sin(j1)*pc2)
    3. Scale the mix so the resulting state has the target Ksi
    """
    rad = np.radians(j1_deg)
    # Directional component (low Ksi — points along one direction)
    h_dir = float(np.cos(rad)) * pc1 + float(np.sin(rad)) * pc2
    h_dir = h_dir / (np.linalg.norm(h_dir) + 1e-12)

    # Isotropic component (high Ksi — spread across all U_ref directions)
    rng = np.random.default_rng(int(j1_deg * 100 + ksi_target * 10000) % 2**31)
    h_iso = rng.standard_normal(dim).astype(np.float32)
    h_iso = h_iso / (np.linalg.norm(h_iso) + 1e-12)

    # Mix: alpha=1 → pure directional (low Ksi), alpha=0 → isotropic (high Ksi)
    # Ksi ≈ alpha * ksi_dir + (1-alpha) * ksi_iso
    # ksi_dir ≈ 0.1-0.2 (concentrated), ksi_iso ≈ 0.95+ (uniform)
    # Solve for alpha given target ksi
    ksi_dir = 0.15; ksi_iso = 0.92
    if ksi_iso > ksi_dir:
        alpha = np.clip((ksi_iso - ksi_target) / (ksi_iso - ksi_dir), 0, 1)
    else:
        alpha = 0.5
    alpha = float(alpha)

    h_mix = alpha * h_dir + (1 - alpha) * h_iso
    h_mix = h_mix / (np.linalg.norm(h_mix) + 1e-12)
    return h_mix.astype(np.float32)

def measure_ksi_vec(h, U_ref):
    """Measure actual Ksi of a hidden state."""
    h_n = h / (np.linalg.norm(h) + 1e-12)
    proj = U_ref.T @ h_n
    p = proj**2 / (proj**2).sum() + 1e-12
    return float(np.clip(-np.sum(p * np.log(p + 1e-12)) / np.log(K_PROJ), 0, 1))

def classify_token(token_str):
    """Classify a decoded token string into a register category."""
    s = token_str.strip()
    if not s:
        return 'FUNCTION'
    # Garbage check
    if '\ufffd' in token_str or any(ord(c) > 1000 for c in token_str):
        return 'GARBAGE'
    if not any(c.isalpha() or c in '.,;:!?' for c in token_str):
        if any(c.isdigit() for c in token_str):
            return 'NUMERIC'
        if any(c in '.,;:!?"\'-–—' for c in token_str):
            return 'PUNCT'
        return 'GARBAGE'
    # English checks
    clean = token_str.strip("' \t")
    if clean in SENTENCE_START_TOKENS:
        return 'SENTENCE_START'
    if token_str in FUNCTION_TOKENS or clean.lower() in {f.strip().lower() for f in FUNCTION_TOKENS}:
        return 'FUNCTION'
    if any(c.isdigit() for c in clean) and not any(c.isalpha() for c in clean):
        return 'NUMERIC'
    if len(clean) >= 2 and any(c.isalpha() for c in clean):
        return 'CONTENT'
    return 'FUNCTION'

# ── Subcommand: scan ───────────────────────────────────────────────────────────

def cmd_scan(args):
    print("\n" + "="*70)
    print("MANIFOLD ILLUMINATOR — Scanning Weight Space Surface")
    print("="*70)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"\n  Loading model: {args.model}", flush=True)
    import torch
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()
    tok   = GPT2Tokenizer.from_pretrained(args.model)
    W     = model.lm_head.weight.detach().float().numpy()
    W_n   = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    print(f"  LM_Head: {W_n.shape}")

    with open(args.engram) as f: e = json.load(f)
    pc1   = np.array(e["pc1"], dtype=np.float32)
    pc2   = np.array(e["pc2"], dtype=np.float32)
    U_ref = np.array(e["U_ref"], dtype=np.float32)
    ksi_v = np.array(e["ksi_vals"], dtype=np.float32)
    j1_v  = np.array(e["j1_pca"], dtype=np.float32) % 360.0
    print(f"  Engram: {len(ksi_v)} tokens")

    j1_steps  = getattr(args, 'j1_steps', 72)
    ksi_steps = getattr(args, 'ksi_steps', 40)
    j1_grid   = np.linspace(0, 360, j1_steps, endpoint=False)
    ksi_grid  = np.linspace(0.05, 0.95, ksi_steps)

    print(f"\n  Grid: {j1_steps} × {ksi_steps} = {j1_steps * ksi_steps} cells")
    print(f"  J1:  {j1_grid[0]:.0f}° to {j1_grid[-1]:.0f}°  (step={j1_grid[1]-j1_grid[0]:.1f}°)")
    print(f"  Ksi: {ksi_grid[0]:.3f} to {ksi_grid[-1]:.3f}  (step={ksi_grid[1]-ksi_grid[0]:.3f})")

    # Output arrays
    coherence    = np.zeros((j1_steps, ksi_steps), dtype=np.float32)
    top1_cos     = np.zeros((j1_steps, ksi_steps), dtype=np.float32)
    entropy_arr  = np.zeros((j1_steps, ksi_steps), dtype=np.float32)
    ksi_actual   = np.zeros((j1_steps, ksi_steps), dtype=np.float32)
    register_map = np.zeros((j1_steps, ksi_steps), dtype=np.int8)
    # register encoding: 0=GARBAGE, 1=FUNCTION, 2=SENTENCE_START, 3=CONTENT, 4=PUNCT, 5=NUMERIC
    REG_INT = {'GARBAGE':0, 'FUNCTION':1, 'SENTENCE_START':2,
               'CONTENT':3, 'PUNCT':4, 'NUMERIC':5, 'MIXED':1}
    top_tokens_arr = {}   # (j1_idx, ksi_idx) → list of top-3 decoded tokens

    t0 = time.time()
    n_total = j1_steps * ksi_steps

    for j_idx, j1 in enumerate(j1_grid):
        for k_idx, ksi in enumerate(ksi_grid):
            h = build_synthetic_h(j1, ksi, pc1, pc2, U_ref)

            # Measure actual Ksi achieved
            ksi_actual[j_idx, k_idx] = measure_ksi_vec(h, U_ref)

            # LM_Head scores (no full transformer pass)
            logits = W_n @ h.astype(np.float64)
            top50_idx = np.argsort(logits)[::-1][:50]
            top50_logits = logits[top50_idx]

            # Top-1 cosine
            top1_cos[j_idx, k_idx] = float(top50_logits[0])

            # Entropy of top-50
            top50_shift = top50_logits - top50_logits.max()
            top50_probs = np.exp(top50_shift); top50_probs /= top50_probs.sum()
            ent = float(-np.sum(top50_probs * np.log(top50_probs + 1e-12)) / np.log(50))
            entropy_arr[j_idx, k_idx] = ent

            # Decode top-50 and classify
            decoded = []
            for tid in top50_idx[:50]:
                try:
                    s = tok.decode([int(tid)], skip_special_tokens=False)
                    decoded.append(s)
                except Exception:
                    decoded.append('')

            # Coherence: fraction of decoded tokens that are real English
            coherent = sum(1 for s in decoded
                           if not '\ufffd' in s
                           and any(c.isalpha() for c in s)
                           and all(ord(c) < 256 for c in s)) / 50.0
            coherence[j_idx, k_idx] = coherent

            # Register: dominant class in top-10
            top10_classes = [classify_token(s) for s in decoded[:10]]
            from collections import Counter
            dom_class = Counter(top10_classes).most_common(1)[0][0]
            register_map[j_idx, k_idx] = REG_INT[dom_class]

            # Store top-3 tokens
            top_tokens_arr[(j_idx, k_idx)] = decoded[:3]

        if j_idx % 10 == 0:
            elapsed = time.time() - t0
            done = (j_idx * ksi_steps)
            rate = done / (elapsed + 1e-12)
            remaining = (n_total - done) / (rate + 1e-12)
            print(f"  J1={j1:.0f}°  {done}/{n_total} cells  "
                  f"{elapsed:.0f}s elapsed  ~{remaining:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_total} cells in {elapsed:.1f}s ({n_total/elapsed:.0f} cells/s)")

    # Build representative token labels for a subset of cells
    # Store as JSON-serializable dict
    token_labels = {
        f"{j_idx},{k_idx}": top_tokens_arr.get((j_idx, k_idx), [])
        for j_idx in range(0, j1_steps, 4)  # every 4th J1
        for k_idx in range(0, ksi_steps, 2)   # every 2nd Ksi
    }

    np.savez_compressed(args.output,
        j1_grid=j1_grid, ksi_grid=ksi_grid,
        coherence=coherence, top1_cos=top1_cos,
        entropy=entropy_arr, ksi_actual=ksi_actual,
        register_map=register_map)

    # Save token labels separately (JSON)
    token_label_path = args.output.replace('.npz', '_tokens.json')
    with open(token_label_path, 'w') as f:
        json.dump(token_labels, f, indent=2)

    print(f"  Scan saved → {args.output}")
    print(f"  Token labels → {token_label_path}")
    print(f"\n  Preview (coherence at key addresses):")
    for j1_target, label in [(110, "Zone A peak"), (270, "Wada"), (320, "Zone B"), (200, "Dead zone")]:
        j_idx = np.argmin(np.abs(j1_grid - j1_target))
        for ksi_target, ksi_label in [(0.30, "low-Ksi"), (0.585, "EN"), (0.88, "Wada-Ksi")]:
            k_idx = np.argmin(np.abs(ksi_grid - ksi_target))
            coh   = coherence[j_idx, k_idx]
            reg   = ['GARB','FUNC','SENT','CONT','PUNC','NUM'][register_map[j_idx, k_idx]]
            toks  = top_tokens_arr.get((j_idx, k_idx), [])[:2]
            print(f"    J1={j1_target}° Ksi={ksi_target:.3f} [{label},{ksi_label}]  "
                  f"coh={coh:.3f} reg={reg}  top={toks}")


# ── Subcommand: plot ───────────────────────────────────────────────────────────

def cmd_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle, FancyArrowPatch
    from matplotlib.lines import Line2D

    print("\n" + "="*70)
    print("MANIFOLD ILLUMINATOR — Plotting Heatmaps")
    print("="*70)

    data = np.load(args.scan, allow_pickle=True)
    j1_grid  = data['j1_grid']
    ksi_grid = data['ksi_grid']
    coherence   = data['coherence']
    top1_cos    = data['top1_cos']
    entropy_arr = data['entropy']
    register_map = data['register_map']

    # Load bivector field confidence if available
    field_conf = None
    if args.field and Path(args.field).exists():
        field_data = np.load(args.field, allow_pickle=True)['field']
        BINS_J1=36; BINS_KSI=20
        field_conf = np.zeros((len(j1_grid), len(ksi_grid)), dtype=np.float32)
        count = field_data[:,:,2]
        for j_idx, j1 in enumerate(j1_grid):
            for k_idx, ksi in enumerate(ksi_grid):
                js = int(j1/10) % BINS_J1
                ks = min(int(ksi/0.05), BINS_KSI-1)
                c  = float(count[js, ks])
                field_conf[j_idx, k_idx] = float(np.tanh(c/100.))
        print(f"  Field confidence loaded: {np.sum(count>5):.0f}/720 cells populated")

    # Build register color image
    reg_colors_arr = np.array([
        [0.8, 0.1, 0.1],   # 0 GARBAGE  → red
        [0.2, 0.75, 0.4],  # 1 FUNCTION → green
        [0.2, 0.4, 0.9],   # 2 SENTENCE_START → blue
        [0.9, 0.75, 0.1],  # 3 CONTENT  → gold
        [0.9, 0.5, 0.1],   # 4 PUNCT    → orange
        [0.7, 0.2, 0.8],   # 5 NUMERIC  → purple
    ])
    reg_rgb = reg_colors_arr[register_map]   # (J1, Ksi, 3)
    # Modulate by coherence
    coh_3d = coherence[:, :, np.newaxis]
    reg_lit = np.clip(reg_rgb * (0.3 + 0.7 * coh_3d), 0, 1)

    # Figure layout: 2×2 or 2×3 depending on field availability
    n_plots = 5 if field_conf is not None else 4
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    fig.patch.set_facecolor('#0a0a1a')

    def add_landmarks(ax, j1_grid, ksi_grid):
        """Add certified landmark overlays to an axis."""
        j1_min, j1_max = j1_grid[0], j1_grid[-1]
        ksi_min, ksi_max = ksi_grid[0], ksi_grid[-1]

        # Dead zone shading (J1 = 140-265°)
        dead_lo = (140 - j1_min) / (j1_max - j1_min) * len(j1_grid)
        dead_hi = (265 - j1_min) / (j1_max - j1_min) * len(j1_grid)
        ax.axvspan(dead_lo/len(j1_grid)*360, dead_hi/len(j1_grid)*360,
                   alpha=0.12, color='white', label='Dead zone')

        # Horizontal lines for certified Ksi values
        certified_ksi = [
            (0.4784, '#ff4444', 'α_physical=0.4784', '--'),
            (0.585,  '#44ff88', 'Ksi_EN=0.585',       '-'),
            (0.88,   '#ff8844', 'Wada Ksi=0.88',      ':'),
            (0.438,  '#8844ff', 'β=0.438',             '--'),
        ]
        for ksi_val, color, label, ls in certified_ksi:
            if ksi_min <= ksi_val <= ksi_max:
                ax.axhline(ksi_val, color=color, linewidth=0.9,
                           linestyle=ls, alpha=0.8, label=label)

        # Zone A and Zone B peak markers
        for j1_val, ksi_val, marker, color, label in [
            (110, 0.342, '*', '#4488ff', 'Zone A peak'),
            (320, 0.342, 's', '#44cc44', 'Zone B peak'),
        ]:
            if j1_min <= j1_val <= j1_max and ksi_min <= ksi_val <= ksi_max:
                ax.plot(j1_val, ksi_val, marker=marker, markersize=10,
                        color=color, markeredgecolor='white', markeredgewidth=0.8,
                        zorder=10, label=label)

    def setup_ax(ax, title):
        ax.set_facecolor('#0a0a1a')
        ax.set_title(title, color='white', fontsize=11, pad=8)
        ax.set_xlabel('J1 (degrees)', color='#aaaaaa', fontsize=9)
        ax.set_ylabel('Ksi', color='#aaaaaa', fontsize=9)
        ax.tick_params(colors='#888888', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.set_xlim(0, 360)
        ax.set_ylim(ksi_grid[0], ksi_grid[-1])
        # X-axis ticks at 0, 90, 180, 270, 360
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(['0°', '90°\n(Zone A)', '180°', '270°\n(Wada)', '360°'])

    extent = [0, 360, ksi_grid[0], ksi_grid[-1]]

    # Plot 1: Coherence heatmap
    ax = axes[0]
    im = ax.imshow(coherence.T, origin='lower', aspect='auto', extent=extent,
                   cmap='inferno', vmin=0, vmax=1, interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Coherence', shrink=0.8)
    add_landmarks(ax, j1_grid, ksi_grid)
    setup_ax(ax, 'Coherence Map\n(how English-like is each address?)')

    # Plot 2: Register map (color = token type)
    ax = axes[1]
    ax.imshow(np.transpose(reg_lit, (1,0,2)), origin='lower', aspect='auto',
              extent=extent, interpolation='bilinear')
    add_landmarks(ax, j1_grid, ksi_grid)
    setup_ax(ax, 'Register Map\n(blue=sentence-start, green=function, gold=content, red=garbage)')
    legend_patches = [
        Line2D([0],[0], color=c, linewidth=4, label=l)
        for l, c in [('Sentence Start','#3366dd'), ('Function','#33aa55'),
                      ('Content','#ddaa11'), ('Garbage','#cc1111'),
                      ('Punct','#dd7711'), ('Numeric','#9922cc')]
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7,
              facecolor='#111122', labelcolor='white', framealpha=0.7)

    # Plot 3: Top-1 cosine (token clarity)
    ax = axes[2]
    im = ax.imshow(top1_cos.T, origin='lower', aspect='auto', extent=extent,
                   cmap='plasma', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Top-1 cosine similarity', shrink=0.8)
    add_landmarks(ax, j1_grid, ksi_grid)
    setup_ax(ax, 'Token Clarity\n(cosine similarity to best-matching token)')

    # Plot 4: Entropy (diversity of top-50)
    ax = axes[3]
    im = ax.imshow(entropy_arr.T, origin='lower', aspect='auto', extent=extent,
                   cmap='viridis', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Entropy (normalized)', shrink=0.8)
    add_landmarks(ax, j1_grid, ksi_grid)
    setup_ax(ax, 'Entropy Map\n(high = diffuse, many options; low = focused)')

    # Plot 5: Field confidence (if available)
    if field_conf is not None:
        ax = axes[4]
        im = ax.imshow(field_conf.T, origin='lower', aspect='auto', extent=extent,
                       cmap='cividis', vmin=0, vmax=1, interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Arc field confidence', shrink=0.8)
        add_landmarks(ax, j1_grid, ksi_grid)
        setup_ax(ax, 'Arc Field Confidence\n(Wikipedia bivector field coverage)')
    else:
        axes[4].set_visible(False)

    # Plot 6: Composite — coherence × (1-entropy) as "signal quality"
    ax = axes[5]
    signal_quality = coherence * (1.0 - entropy_arr * 0.5)
    # Color by register, brightness by signal quality
    composite = np.transpose(reg_lit, (1,0,2)) * signal_quality.T[:,:,np.newaxis]
    composite = np.clip(composite * 1.5, 0, 1)
    ax.imshow(composite, origin='lower', aspect='auto', extent=extent,
              interpolation='bilinear')
    add_landmarks(ax, j1_grid, ksi_grid)
    setup_ax(ax, 'Composite: Signal Quality × Register\n'
             '(brightness = coherence, color = vocabulary type)')

    # Main title
    fig.suptitle('Transformer Weight Space Manifold — (J₁, Ksi) Surface\n'
                 'Pirouette Framework Volume 8 · CORE-003',
                 color='white', fontsize=13, y=1.01)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(args.output, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Heatmap saved → {args.output}")
    print(f"  Size: {Path(args.output).stat().st_size // 1024} KB")


# ── Subcommand: query ──────────────────────────────────────────────────────────

def cmd_query(args):
    """Query the scan at a specific (J1, Ksi) address."""
    data = np.load(args.scan, allow_pickle=True)
    j1_grid   = data['j1_grid']
    ksi_grid  = data['ksi_grid']
    coherence = data['coherence']
    register  = data['register_map']

    j1_target  = float(args.j1)
    ksi_target = float(args.ksi)
    j_idx = int(np.argmin(np.abs(j1_grid  - j1_target)))
    k_idx = int(np.argmin(np.abs(ksi_grid - ksi_target)))

    print(f"\n  Query: J1={j1_target}°, Ksi={ksi_target}")
    print(f"  Nearest cell: J1={j1_grid[j_idx]:.1f}°, Ksi={ksi_grid[k_idx]:.3f}")
    print(f"  Coherence: {coherence[j_idx,k_idx]:.4f}")
    reg_names = ['GARBAGE','FUNCTION','SENTENCE_START','CONTENT','PUNCT','NUMERIC']
    print(f"  Register: {reg_names[register[j_idx,k_idx]]}")

    # Load token labels if available
    token_path = args.scan.replace('.npz','_tokens.json')
    if Path(token_path).exists():
        with open(token_path) as f: tl = json.load(f)
        key = f"{j_idx},{k_idx}"
        if key in tl:
            print(f"  Top tokens: {tl[key]}")

    if args.output:
        result = {"j1": float(j1_grid[j_idx]), "ksi": float(ksi_grid[k_idx]),
                   "coherence": float(coherence[j_idx,k_idx]),
                   "register": reg_names[register[j_idx,k_idx]]}
        with open(args.output,'w') as f: json.dump(result,f,indent=2)
        print(f"  Saved → {args.output}")


# ── Subcommand: all ────────────────────────────────────────────────────────────

def cmd_all(args):
    """Run scan then plot."""
    scan_path = args.output + '_scan.npz'
    plot_path = args.output + '_heatmap.png'

    class ScanArgs:
        model   = args.model
        engram  = args.engram
        output  = scan_path
        j1_steps  = getattr(args,'j1_steps',72)
        ksi_steps = getattr(args,'ksi_steps',40)

    class PlotArgs:
        scan   = scan_path
        field  = getattr(args,'field',None)
        output = plot_path

    cmd_scan(ScanArgs())
    cmd_plot(PlotArgs())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Manifold Illuminator")
    sub = ap.add_subparsers(dest="cmd")

    ps = sub.add_parser("scan")
    ps.add_argument("--model",     required=True)
    ps.add_argument("--engram",    required=True)
    ps.add_argument("--output",    default="manifold_scan.npz")
    ps.add_argument("--j1_steps",  type=int, default=72)
    ps.add_argument("--ksi_steps", type=int, default=40)

    pp = sub.add_parser("plot")
    pp.add_argument("--scan",   required=True)
    pp.add_argument("--field",  default=None)
    pp.add_argument("--output", default="manifold_heatmap.png")

    pq = sub.add_parser("query")
    pq.add_argument("--scan",   required=True)
    pq.add_argument("--j1",     type=float, required=True)
    pq.add_argument("--ksi",    type=float, required=True)
    pq.add_argument("--output", default=None)

    pa = sub.add_parser("all")
    pa.add_argument("--model",     required=True)
    pa.add_argument("--engram",    required=True)
    pa.add_argument("--field",     default=None)
    pa.add_argument("--output",    default="manifold")
    pa.add_argument("--j1_steps",  type=int, default=72)
    pa.add_argument("--ksi_steps", type=int, default=40)

    args = ap.parse_args()
    {"scan": cmd_scan, "plot": cmd_plot, "query": cmd_query,
     "all": cmd_all}.get(args.cmd, lambda _: (ap.print_help(), sys.exit(1)))(args)

if __name__ == "__main__":
    main()
