"""
lmhead_geometry_survey.py
LM_Head Geometric Survey Instrument
Pirouette Framework Volume 8 · CORE-003 · ML-058-PRE

PURPOSE
=======
The LM_Head is the bottleneck between geometric navigation and language output.
This instrument probes it as a geometric object on the HH manifold.

THREE SUBCOMMANDS
=================

  lmhead_survey   — Bird's-eye IFS spectral map of the LM_Head itself.
                    Where does it sit? What basins does it cover?
                    Answers: "What is the LM_Head's geometry?"

  lmhead_align    — For a set of (Ksi, J1) probe addresses, measure cosine
                    alignment between synthetic hidden states and each LM_Head
                    row (token vector). Returns top-K tokens per address.
                    Answers: "Which tokens live at my target coordinates?"

  lmhead_formula  — Test the spectral basin matching hypothesis:
                    does token alpha ~ hidden state alpha predict logit rank?
                    If TRUE: a formula replaces forward passes for candidate
                    filtering. No LM_Head needed for the shortlist.
                    Answers: "Can I predict good tokens geometrically?"

HYPOTHESES (pre-registered)
============================
  H-LMH-001: The LM_Head has the two-regime IFS structure (head + tail)
              seen in all other weight matrices. alpha_tail ≈ 0.12.
              PASS if: IFS k=3 error < 5%, tail segment alpha ∈ [0.10, 0.17].

  H-LMH-002: Token vectors cluster in (Ksi, J1) space — the engram_curve.json
              distribution has internal IFS structure (non-uniform coverage).
              PASS if: token density in (Ksi, J1) space has measurable
              fractal dimension d_f ∈ [1.2, 1.8].

  H-LMH-003: Spectral basin matching formula — tokens whose IFS alpha is
              closest to the hidden state's Ksi score systematically higher
              than random tokens. PASS if: Spearman rank correlation between
              |alpha_token - ksi_h| and cosine_score < -0.3 (p < 0.05).
              This is the "no-LM_Head" shortlist formula.

  H-LMH-004: Geometric blind spots — addresses with Ksi > 0.85 (Wada zone)
              have LOWER mean token alignment than addresses with Ksi ∈ [0.5, 0.8].
              The Wada zone is a coverage gap in the LM_Head.
              PASS if: mean cosine alignment drops > 15% in Wada zone.

Usage:
  # Bird's-eye IFS map of the LM_Head:
  python lmhead_geometry_survey.py lmhead_survey ^
    --model models\\gpt2-large-cycle3-crust-arc1 ^
    --output lmhead_survey.json

  # Token alignment at probe addresses:
  python lmhead_geometry_survey.py lmhead_align ^
    --model models\\gpt2-large-cycle3-crust-arc1 ^
    --engram engram_curve.json ^
    --output lmhead_align.json

  # Spectral basin matching formula test:
  python lmhead_geometry_survey.py lmhead_formula ^
    --model models\\gpt2-large-cycle3-crust-arc1 ^
    --engram engram_curve.json ^
    --output lmhead_formula.json
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

K_PROJ    = 16          # hidden state projection dimension (from v14)
VOCAB_SIZE = 50257      # GPT-2-Large vocab size

# IFS compression params (from CORE-021)
IFS_K     = 3           # number of piecewise log-log segments
IFS_K_FINE = 7          # fine-grained for head decomposition

# Probe addresses for alignment test — spans the manifold systematically
# Format: (label, Ksi, J1_deg)
PROBE_ADDRESSES = [
    ("native_english",    0.585,  90.0),   # Ksi_EN certified attractor
    ("delta_c_synthesis", 0.585, 196.0),   # Δ_c = +0.16 synthesis address
    ("wada_boundary",     0.880, 270.0),   # Wada zone (Ksi ≈ 0.88)
    ("basin_interior",    0.400, 180.0),   # deep basin interior
    ("high_ksi_basin1",   0.700,  45.0),   # mid-high Ksi basin A
    ("high_ksi_basin2",   0.700, 165.0),   # mid-high Ksi basin B
    ("high_ksi_basin3",   0.700, 285.0),   # mid-high Ksi basin C (Z3)
    ("abstract_zone",     0.730, 180.0),   # abstract/abstract zone (helicity peak)
    ("low_ksi_stable",    0.250,  90.0),   # very stable deep basin
    ("near_escape",       0.950, 135.0),   # near escape energy
]

# ── Utilities ──────────────────────────────────────────────────────────────────

def load_model_lmhead(model_path: str):
    """Load GPT-2 model, return lm_head weight matrix [vocab, hidden]."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"  Loading model: {model_path}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(model_path)
    W = model.lm_head.weight.detach().float().numpy()   # [50257, 1280]
    print(f"  LM_Head shape: {W.shape}", flush=True)
    return W, tok, model

def load_engram(path: str) -> dict:
    """Load engram_curve.json — token (Ksi, J1) addresses."""
    with open(path) as f:
        e = json.load(f)
    e["ksi_vals"] = np.array(e["ksi_vals"], dtype=np.float32)
    e["j1_360"]   = np.array(e["j1_pca"],   dtype=np.float32) % 360.0
    e["pc1"]      = np.array(e["pc1"],       dtype=np.float32)
    e["pc2"]      = np.array(e["pc2"],       dtype=np.float32)
    e["U_ref"]    = np.array(e["U_ref"],     dtype=np.float32)
    return e

def ifs_fit_spectrum(sv: np.ndarray, k: int = IFS_K):
    """
    Fit IFS piecewise log-log power law to singular value spectrum.
    Returns: list of (alpha_i, intercept_i, rank_start, rank_end) per segment.
    Overall alpha = slope of full log-log regression.
    """
    n = len(sv)
    sv = sv[sv > 0]
    ranks = np.arange(1, len(sv) + 1, dtype=float)
    log_r = np.log(ranks)
    log_s = np.log(sv)

    # Overall fit
    c_all = np.polyfit(log_r, log_s, 1)
    alpha_all = -c_all[0]

    # Piecewise segments (equal-width in log-rank space)
    breakpoints = np.linspace(0, len(sv), k + 1, dtype=int)
    segments = []
    for i in range(k):
        s, e = breakpoints[i], breakpoints[i + 1]
        if e - s < 2:
            segments.append(None)
            continue
        lr = log_r[s:e]; ls = log_s[s:e]
        c = np.polyfit(lr, ls, 1)
        alpha_seg = -c[0]
        pred = np.polyval(c, lr)
        err_pct = float(np.mean(np.abs(np.exp(pred) - np.exp(ls)) / (np.exp(ls) + 1e-12)) * 100)
        segments.append({
            "seg_idx": i,
            "rank_start": int(s),
            "rank_end":   int(e),
            "alpha":      float(alpha_seg),
            "error_pct":  err_pct
        })

    # Reconstruct top-20 singular values from IFS
    sv1_recon = float(np.exp(np.polyval(np.polyfit(log_r[:20], log_s[:20], 1), log_r[0])))
    sv1_orig  = float(sv[0])

    return {
        "alpha":      float(alpha_all),
        "n_sv":       len(sv),
        "segments":   segments,
        "sv1_orig":   sv1_orig,
        "sv1_recon":  sv1_recon,
        "sv1_err_pct": abs(sv1_orig - sv1_recon) / (sv1_orig + 1e-12) * 100,
        "tail_alpha": float(segments[-1]["alpha"]) if segments[-1] else float(alpha_all),
        "head_alpha": float(segments[0]["alpha"])  if segments[0]  else float(alpha_all),
    }

def hidden_state_from_address(ksi_target: float, j1_deg: float,
                               engram: dict, hidden_dim: int = 1280) -> np.ndarray:
    """
    Synthesize a hidden state vector that sits at (Ksi, J1) in engram space.
    Method:
      1. Find the token whose (ksi, j1) is closest to the target.
      2. Use its LM_Head row direction as the primary direction.
      3. Add a small random component for breadth.
    This is a probe vector — not a true forward-pass hidden state,
    but lives at the correct geometric address.
    """
    ksi_v = engram["ksi_vals"]
    j1_v  = engram["j1_360"]
    pc1   = engram["pc1"]    # [hidden_dim]
    pc2   = engram["pc2"]    # [hidden_dim]

    # Circular distance in J1, euclidean in Ksi
    d_ksi = np.abs(ksi_v - ksi_target)
    d_j1  = np.abs(j1_v - j1_deg) % 360.0
    d_j1  = np.minimum(d_j1, 360.0 - d_j1) / 180.0   # normalize to [0,1]
    dist  = d_ksi + 0.5 * d_j1
    seed_tok = int(np.argmin(dist))

    # Build hidden state in (pc1, pc2) plane at (J1, Ksi) angle
    j1_rad = np.radians(j1_deg)
    h = (np.cos(j1_rad) * pc1 + np.sin(j1_rad) * pc2)
    h = h / (np.linalg.norm(h) + 1e-12)

    # Scale by Ksi-encoded norm (higher Ksi = more diffuse = smaller norm)
    scale = 1.0 - 0.3 * ksi_target
    h = h * scale

    return h.astype(np.float32), seed_tok

def compute_token_alphas(W_lm: np.ndarray, batch_size: int = 500) -> np.ndarray:
    """
    Compute per-token spectral 'alpha' approximation.
    For each token row w_i ∈ R^hidden_dim, treat |w_i| sorted descending
    as a pseudo-spectrum and fit power law slope.
    This is the within-row spectral structure — the token's own IFS coordinate.
    Returns: alpha per token [vocab_size]
    """
    vocab, hidden = W_lm.shape
    alphas = np.zeros(vocab, dtype=np.float32)
    log_ranks = np.log(np.arange(1, hidden + 1, dtype=float))

    for start in range(0, vocab, batch_size):
        end = min(start + batch_size, vocab)
        batch = W_lm[start:end]                          # [B, hidden]
        batch_sorted = np.sort(np.abs(batch), axis=1)[:, ::-1]  # descending
        # Log-log fit: log(sv) ~ alpha * log(rank) + const
        # Use vectorized polyfit across all rows in batch
        log_sv = np.log(batch_sorted + 1e-12)
        # alpha = -slope
        # polyfit via formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        n = hidden
        sum_x  = log_ranks.sum()
        sum_x2 = (log_ranks**2).sum()
        sum_y  = log_sv.sum(axis=1)                      # [B]
        sum_xy = (log_sv * log_ranks[None, :]).sum(axis=1)  # [B]
        slope  = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-12)
        alphas[start:end] = -slope.astype(np.float32)

        if start % 10000 == 0:
            print(f"    token alpha: {start}/{vocab}", flush=True)

    return alphas

# ── Subcommand: lmhead_survey ──────────────────────────────────────────────────

def cmd_lmhead_survey(args):
    """
    Bird's-eye IFS spectral survey of the LM_Head weight matrix.
    Treats lm_head.weight as a weight tensor and applies the CORE-021 IFS pipeline.
    """
    print("\n" + "="*70)
    print("LMHEAD_SURVEY — IFS Spectral Map of LM_Head")
    print("="*70)

    W, tok, model = load_model_lmhead(args.model)

    print("\n[1/4] Computing SVD of LM_Head [50257 × 1280]...")
    # Full SVD is expensive — use randomized SVD
    from sklearn.utils.extmath import randomized_svd
    n_sv = min(200, min(W.shape))
    U, sv, Vt = randomized_svd(W, n_components=n_sv, random_state=42)
    print(f"  Top singular value: {sv[0]:.4f}")
    print(f"  sv[1]/sv[0] ratio:  {sv[1]/sv[0]:.4f}")
    print(f"  sv[10]/sv[0] ratio: {sv[10]/sv[0]:.4f}")

    print("\n[2/4] IFS piecewise log-log fit (k=3 and k=7)...")
    ifs3 = ifs_fit_spectrum(sv, k=3)
    ifs7 = ifs_fit_spectrum(sv, k=7)

    print(f"\n  IFS k=3:")
    print(f"    overall alpha:    {ifs3['alpha']:.4f}")
    print(f"    head alpha (s1):  {ifs3['head_alpha']:.4f}")
    print(f"    tail alpha (s3):  {ifs3['tail_alpha']:.4f}")
    print(f"    sv1 recon err:    {ifs3['sv1_err_pct']:.2f}%")

    print(f"\n  IFS k=7:")
    print(f"    overall alpha:    {ifs7['alpha']:.4f}")
    print(f"    head alpha (s1):  {ifs7['head_alpha']:.4f}")
    print(f"    tail alpha (s7):  {ifs7['tail_alpha']:.4f}")

    # Basin assignment (from CORE-021 alpha thresholds)
    alpha = ifs3['alpha']
    if alpha < 0.43:
        basin = "MLP-DOMAIN (alpha < 0.43, low fractal dim)"
    elif alpha < 0.53:
        basin = "WADA-ZONE (alpha 0.43–0.53, physical fractal constant boundary)"
    elif alpha < 0.65:
        basin = "TRANSITION (alpha 0.53–0.65)"
    else:
        basin = "ATTENTION-DOMAIN (alpha > 0.65, high fractal dim)"
    print(f"\n  Basin assignment: {basin}")

    # H-LMH-001 verdict
    tail_ok = 0.10 <= ifs3['tail_alpha'] <= 0.17
    sv1_ok  = ifs3['sv1_err_pct'] < 5.0
    h001 = "PASS" if (tail_ok and sv1_ok) else "FAIL"
    print(f"\n  H-LMH-001 verdict: {h001}")
    print(f"    tail_alpha ∈ [0.10, 0.17]: {tail_ok} (got {ifs3['tail_alpha']:.4f})")
    print(f"    sv1 err < 5%:              {sv1_ok} (got {ifs3['sv1_err_pct']:.2f}%)")

    print("\n[3/4] Token row norms and coverage statistics...")
    row_norms = np.linalg.norm(W, axis=1)    # [vocab]
    print(f"  Mean row norm:   {row_norms.mean():.4f}")
    print(f"  Std  row norm:   {row_norms.std():.4f}")
    print(f"  Min  row norm:   {row_norms.min():.4f}")
    print(f"  Max  row norm:   {row_norms.max():.4f}")
    print(f"  % rows > 1.0:    {(row_norms > 1.0).mean()*100:.1f}%")
    print(f"  % rows < 0.1:    {(row_norms < 0.1).mean()*100:.1f}%")

    print("\n[4/4] Top 10 and bottom 10 singular values...")
    for i, s in enumerate(sv[:10]):
        print(f"  sv[{i:3d}] = {s:.6f}")
    print("  ...")
    for i, s in enumerate(sv[-5:]):
        print(f"  sv[{n_sv-5+i:3d}] = {s:.6f}")

    results = {
        "lmhead_shape": list(W.shape),
        "sv_top_20": [float(x) for x in sv[:20]],
        "ifs3": ifs3,
        "ifs7": ifs7,
        "basin": basin,
        "h001_verdict": h001,
        "row_norm_mean": float(row_norms.mean()),
        "row_norm_std":  float(row_norms.std()),
        "row_norm_min":  float(row_norms.min()),
        "row_norm_max":  float(row_norms.max()),
    }

    _save(results, args.output)
    print(f"\n  Saved → {args.output}")

# ── Subcommand: lmhead_align ──────────────────────────────────────────────────

def cmd_lmhead_align(args):
    """
    Token alignment test: for each probe address (Ksi, J1),
    synthesize a hidden state and measure cosine similarity to all LM_Head rows.
    Returns top-K tokens per address and coverage statistics.
    Tests H-LMH-002 (fractal token distribution) and H-LMH-004 (Wada blind spot).
    """
    print("\n" + "="*70)
    print("LMHEAD_ALIGN — Token Alignment at Probe Addresses")
    print("="*70)

    W, tok, model = load_model_lmhead(args.model)
    engram = load_engram(args.engram)
    top_k = getattr(args, 'top_k', 30)

    # Normalize LM_Head rows for cosine similarity
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    print(f"\n  Probing {len(PROBE_ADDRESSES)} addresses (top_k={top_k})...\n")

    address_results = []
    wada_cosines = []
    normal_cosines = []

    for label, ksi_target, j1_deg in PROBE_ADDRESSES:
        h, seed_tok = hidden_state_from_address(ksi_target, j1_deg, engram)
        h_norm = h / (np.linalg.norm(h) + 1e-12)

        cos_scores = W_norm @ h_norm.astype(np.float64)    # [vocab]

        top_ids = np.argsort(cos_scores)[::-1][:top_k]
        top_scores = cos_scores[top_ids]
        top_tokens = [tok.decode([int(i)], skip_special_tokens=False)
                      for i in top_ids]

        # Ksi distribution of top tokens
        top_ksi = engram["ksi_vals"][top_ids]
        top_j1  = engram["j1_360"][top_ids]

        res = {
            "label":        label,
            "ksi_target":   float(ksi_target),
            "j1_target":    float(j1_deg),
            "seed_token_id": int(seed_tok),
            "top_tokens":   top_tokens[:15],
            "top_scores":   [float(x) for x in top_scores[:15]],
            "mean_cos_top10": float(np.mean(top_scores[:10])),
            "top_ksi_mean":  float(top_ksi.mean()),
            "top_ksi_std":   float(top_ksi.std()),
            "top_j1_mean":   float(top_j1.mean()),
        }
        address_results.append(res)

        if ksi_target >= 0.85:
            wada_cosines.append(float(np.mean(top_scores[:10])))
        else:
            normal_cosines.append(float(np.mean(top_scores[:10])))

        print(f"  [{label}] Ksi={ksi_target:.3f} J1={j1_deg:.0f}°")
        print(f"    mean_cos(top10): {np.mean(top_scores[:10]):.4f}")
        print(f"    top-5 tokens:   {' | '.join(top_tokens[:5])}")
        print()

    # H-LMH-002: fractal dimension of token distribution in (Ksi, J1) space
    print("\n  H-LMH-002: Fractal dimension of token (Ksi, J1) distribution...")
    ksi_all = engram["ksi_vals"]
    j1_all  = engram["j1_360"]
    # Box-counting on a 20×20 grid in (Ksi, J1_normalized) space
    ksi_norm = (ksi_all - ksi_all.min()) / (ksi_all.max() - ksi_all.min() + 1e-12)
    j1_norm  = j1_all / 360.0
    box_counts = []
    grid_sizes = [4, 8, 16, 32, 64]
    for g in grid_sizes:
        grid_ksi = np.floor(ksi_norm * g).astype(int)
        grid_j1  = np.floor(j1_norm  * g).astype(int)
        occupied = len(set(zip(grid_ksi.tolist(), grid_j1.tolist())))
        box_counts.append(occupied)
    log_g   = np.log(grid_sizes)
    log_bc  = np.log(box_counts)
    d_f     = float(np.polyfit(log_g, log_bc, 1)[0])
    h002    = "PASS" if 1.2 <= d_f <= 1.8 else "FAIL"
    print(f"    box counts:    {box_counts}")
    print(f"    d_f (slope):   {d_f:.4f}")
    print(f"    H-LMH-002:     {h002}")

    # H-LMH-004: Wada blind spot
    h004 = "NOT TESTABLE (need both wada and normal addresses)"
    if wada_cosines and normal_cosines:
        wada_mean   = float(np.mean(wada_cosines))
        normal_mean = float(np.mean(normal_cosines))
        drop_pct    = (normal_mean - wada_mean) / (normal_mean + 1e-12) * 100
        h004 = "PASS" if drop_pct > 15.0 else "FAIL"
        print(f"\n  H-LMH-004: Wada blind spot")
        print(f"    normal mean cos(top10): {normal_mean:.4f}")
        print(f"    wada   mean cos(top10): {wada_mean:.4f}")
        print(f"    drop:                   {drop_pct:.1f}%")
        print(f"    H-LMH-004:              {h004}")

    results = {
        "addresses": address_results,
        "h002_verdict": h002,
        "h002_d_f":     d_f,
        "h002_box_counts": box_counts,
        "h004_verdict": h004,
        "wada_cosines":   wada_cosines,
        "normal_cosines": normal_cosines,
    }
    _save(results, args.output)
    print(f"\n  Saved → {args.output}")

# ── Subcommand: lmhead_formula ─────────────────────────────────────────────────

def cmd_lmhead_formula(args):
    """
    Spectral basin matching formula test.
    H-LMH-003: tokens whose IFS alpha is closest to hidden state Ksi score higher.
    If confirmed, this is the LM_Head shortcut — no forward pass needed for filtering.

    Also tests a secondary formula: Ksi_token (from engram) ~ Ksi_hidden,
    since the engram already provides per-token Ksi coordinates.
    """
    print("\n" + "="*70)
    print("LMHEAD_FORMULA — Spectral Basin Matching Hypothesis")
    print("="*70)

    W, tok, model = load_model_lmhead(args.model)
    engram = load_engram(args.engram)
    n_sample = getattr(args, 'n_sample', 5000)    # tokens to sample for alpha compute

    # Normalize LM_Head rows
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    print(f"\n[1/4] Computing per-token row alphas (sample={n_sample} tokens)...")
    # Sample n_sample tokens uniformly
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(VOCAB_SIZE, size=n_sample, replace=False)
    sample_ids.sort()
    W_sample   = W[sample_ids]

    # Per-row alpha (spectral decay of sorted weight magnitudes)
    hidden_dim = W.shape[1]
    log_ranks  = np.log(np.arange(1, hidden_dim + 1, dtype=float))
    W_sorted   = np.sort(np.abs(W_sample), axis=1)[:, ::-1]
    log_sv     = np.log(W_sorted + 1e-12)
    n = hidden_dim
    sum_x  = log_ranks.sum(); sum_x2 = (log_ranks**2).sum()
    sum_y  = log_sv.sum(axis=1)
    sum_xy = (log_sv * log_ranks[None, :]).sum(axis=1)
    slopes  = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-12)
    token_alphas = -slopes.astype(np.float32)
    print(f"  Token alpha: mean={token_alphas.mean():.4f} std={token_alphas.std():.4f}")
    print(f"               min={token_alphas.min():.4f}  max={token_alphas.max():.4f}")

    print("\n[2/4] Testing formula at each probe address...")
    from scipy.stats import spearmanr, pearsonr

    formula_results = []
    h003_verdicts   = []

    for label, ksi_target, j1_deg in PROBE_ADDRESSES:
        h, seed_tok = hidden_state_from_address(ksi_target, j1_deg, engram)
        h_norm = h / (np.linalg.norm(h) + 1e-12)

        # True cosine scores for the sampled tokens
        cos_scores = W_norm[sample_ids] @ h_norm.astype(np.float64)

        # Formula 1: |alpha_token - ksi_h| should anti-correlate with cos_score
        alpha_dist  = np.abs(token_alphas - ksi_target)
        rho1, p1    = spearmanr(alpha_dist, cos_scores)

        # Formula 2: |ksi_token - ksi_h| from engram should also anti-correlate
        ksi_tokens  = engram["ksi_vals"][sample_ids]
        ksi_dist    = np.abs(ksi_tokens - ksi_target)
        rho2, p2    = spearmanr(ksi_dist, cos_scores)

        # Formula 3: angular distance in J1 space
        j1_tokens   = engram["j1_360"][sample_ids]
        j1_dist     = np.abs(j1_tokens - j1_deg) % 360.0
        j1_dist     = np.minimum(j1_dist, 360.0 - j1_dist)
        rho3, p3    = spearmanr(j1_dist, cos_scores)

        # Combined formula: weighted by both Ksi and J1 proximity
        combined_dist = ksi_dist + 0.3 * j1_dist / 180.0
        rho_comb, p_comb = spearmanr(combined_dist, cos_scores)

        # H-LMH-003 verdict (primary formula: alpha distance)
        h003 = "PASS" if (rho1 < -0.3 and p1 < 0.05) else "FAIL"
        h003_verdicts.append(h003)

        print(f"\n  [{label}] Ksi={ksi_target:.3f} J1={j1_deg:.0f}°")
        print(f"    Spearman(|alpha-ksi|, cos): rho={rho1:.3f}  p={p1:.3e}  → {h003}")
        print(f"    Spearman(|ksi_tok-ksi|, cos): rho={rho2:.3f}  p={p2:.3e}")
        print(f"    Spearman(J1_dist, cos):       rho={rho3:.3f}  p={p3:.3e}")
        print(f"    Spearman(combined, cos):      rho={rho_comb:.3f}  p={p_comb:.3e}")

        formula_results.append({
            "label": label, "ksi_target": float(ksi_target), "j1_target": float(j1_deg),
            "rho_alpha": float(rho1), "p_alpha": float(p1), "h003": h003,
            "rho_ksi":   float(rho2), "p_ksi":   float(p2),
            "rho_j1":    float(rho3), "p_j1":    float(p3),
            "rho_comb":  float(rho_comb), "p_comb": float(p_comb),
        })

    print("\n[3/4] Cross-address formula consistency...")
    pass_count = h003_verdicts.count("PASS")
    total      = len(h003_verdicts)
    print(f"  H-LMH-003 pass rate: {pass_count}/{total}")
    overall_h003 = "PASS" if pass_count >= total // 2 else "FAIL"
    print(f"  Overall H-LMH-003: {overall_h003}")

    print("\n[4/4] Formula shortlist efficiency estimate...")
    # If formula works: how many tokens pass |alpha - ksi| < threshold?
    # Compare against beam_base in orbital generator
    for ksi_target in [0.585, 0.880, 0.730]:
        alpha_dist  = np.abs(token_alphas - ksi_target)
        for thresh in [0.1, 0.2, 0.3, 0.5]:
            n_pass = (alpha_dist < thresh).sum()
            frac   = n_pass / n_sample * 100
            print(f"  ksi={ksi_target:.3f} |alpha-ksi|<{thresh:.1f}: {n_pass} tokens ({frac:.1f}%)")

    results = {
        "token_alpha_mean": float(token_alphas.mean()),
        "token_alpha_std":  float(token_alphas.std()),
        "n_sample": n_sample,
        "formula_by_address": formula_results,
        "h003_pass_rate": f"{pass_count}/{total}",
        "h003_overall":   overall_h003,
    }
    _save(results, args.output)
    print(f"\n  Saved → {args.output}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(obj, path: str):
    """JSON-safe serialization."""
    def _convert(o):
        if isinstance(o, np.ndarray):    return o.tolist()
        if isinstance(o, (np.int64, np.int32)):   return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o)
        raise TypeError(f"unserializable: {type(o)}")
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_convert)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LM_Head Geometric Survey Instrument")
    sub = ap.add_subparsers(dest="cmd")

    # lmhead_survey
    p1 = sub.add_parser("lmhead_survey")
    p1.add_argument("--model",  required=True)
    p1.add_argument("--output", default="lmhead_survey.json")

    # lmhead_align
    p2 = sub.add_parser("lmhead_align")
    p2.add_argument("--model",  required=True)
    p2.add_argument("--engram", required=True)
    p2.add_argument("--output", default="lmhead_align.json")
    p2.add_argument("--top_k",  type=int, default=30)

    # lmhead_formula
    p3 = sub.add_parser("lmhead_formula")
    p3.add_argument("--model",    required=True)
    p3.add_argument("--engram",   required=True)
    p3.add_argument("--output",   default="lmhead_formula.json")
    p3.add_argument("--n_sample", type=int, default=5000,
                    help="Number of tokens to sample for alpha computation")

    args = ap.parse_args()
    if args.cmd == "lmhead_survey":
        cmd_lmhead_survey(args)
    elif args.cmd == "lmhead_align":
        cmd_lmhead_align(args)
    elif args.cmd == "lmhead_formula":
        cmd_lmhead_formula(args)
    else:
        ap.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
