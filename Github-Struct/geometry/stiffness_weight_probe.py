"""
stiffness_weight_probe.py
Vacuum Stiffness × Weight Matrix Correlation Instrument
Pirouette Framework Volume 8 · CORE-003 · ML-059

PURPOSE
=======
Three independent probes:

1. STIFFNESS_CORR — Does the HH vacuum stiffness field correlate with the
   LM_Head weight structure? The stiffness field is a 360-point LUT over phi.
   The LM_Head token addresses are in (J1, Ksi) space. Test: does token
   cosine-to-hidden-state correlate with stiffness at the token's J1 address?
   This is the "other piece of the fractal" test.

2. CHAOS_DETECT — Distinguish structured chaos from plain chaos using three
   instruments that work without full Lyapunov computation:
     a. Correlation dimension (box-counting on orbit): structured chaos has
        finite fractal dimension; plain chaos fills [0,1] (dim=1)
     b. Recurrence plot diagonal density: structured chaos has long diagonal
        lines (determinism); plain chaos has short diagonal runs
     c. Permutation entropy: structured chaos has lower PE than plain chaos

3. FEIGENBAUM_OPERATOR — Test whether the Feigenbaum scaling constants
   define a useful operator on IFS manifold coordinates. If W_next ≈
   (1/alpha_F) * W_current in spectral space, the weight matrices across
   transformer depth follow Feigenbaum scaling.

PRE-REGISTERED HYPOTHESES
==========================
H-SW-001: STIFFNESS-TOKEN ALIGNMENT
  Tokens at J1 positions where vacuum stiffness is HIGH show higher cosine
  alignment with hidden states at those same J1 positions than tokens at
  LOW-stiffness J1 positions.
  PASS: Spearman rho(stiffness_at_j1, cosine_score) > 0.2, p < 0.05
  at 5+ of 10 probe addresses.

H-SW-002: STRUCTURED CHAOS DETECTION
  The logistic map at r=3.91 (Ksi=0.585, native English) shows lower
  permutation entropy and higher recurrence plot determinism than the map
  at r=3.977 (Ksi=0.88, Wada boundary) — confirming that native English
  sits in more structured chaos than the Wada zone.
  PASS: PE(0.585) < PE(0.880) AND DET(0.585) > DET(0.880)

H-SW-003: FEIGENBAUM LAYER SCALING
  Across transformer depth, the ratio of consecutive layer IFS alpha values
  clusters near 1/Feigenbaum_alpha (= 0.400) or Feigenbaum_alpha (= 2.502)
  with std < 0.3. This would mean transformer depth is a Feigenbaum cascade.
  PASS: |mean(alpha_l / alpha_{l+1}) - 0.400| < 0.1  OR
        |mean(alpha_l / alpha_{l+1}) - 2.502| < 0.5
  NOTE: This requires model weight access. Tested on MLP c_proj layers only.

Usage:
  python stiffness_weight_probe.py stiffness_corr ^
    --model models\\gpt2-large-cycle3-crust-arc1 ^
    --engram engram_curve.json ^
    --output sw_stiffness.json

  python stiffness_weight_probe.py chaos_detect ^
    --output sw_chaos.json

  python stiffness_weight_probe.py feigenbaum_op ^
    --model models\\gpt2-large-cycle3-crust-arc1 ^
    --output sw_feigenbaum.json

  python stiffness_weight_probe.py full_run ^
    --model models\\gpt2-large-cycle3-crust-arc1 ^
    --engram engram_curve.json ^
    --output sw_full.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats

# ── Constants ──────────────────────────────────────────────────────────────────
FEIGENBAUM_ALPHA = 2.50290
FEIGENBAUM_DELTA = 4.66920
FEIGENBAUM_ALPHA_INV = 1.0 / FEIGENBAUM_ALPHA   # 0.3995
VOCAB_SIZE = 50257
TWIST = 3.8   # from hh_orbital_generator_v14 — certified vacuum field TWIST

PROBE_ADDRESSES = [
    ("native_english",    0.585,  90.0),
    ("delta_c_synthesis", 0.585, 196.0),
    ("wada_boundary",     0.880, 270.0),
    ("basin_interior",    0.400, 180.0),
    ("high_ksi_basin1",   0.700,  45.0),
    ("high_ksi_basin2",   0.700, 165.0),
    ("high_ksi_basin3",   0.700, 285.0),
    ("abstract_zone",     0.730, 180.0),
    ("low_ksi_stable",    0.250,  90.0),
    ("near_escape",       0.950, 135.0),
]

# ── Vacuum stiffness LUT (from space_fractal.py physics) ─────────────────────

def compute_stiffness_lut(twist=TWIST, n_phi=360):
    """
    Compute the vacuum stiffness field at n_phi angles on the unit circle.
    Reproduces the force law from space_fractal.py / hh_orbital_generator_v14.

    Physical model: three-basin HH force field with three attractor zones
    (gold at 30°, teal at 150°, red at 270°) and TWIST=3.8 coupling.
    Stiffness = eigenvalue of local force gradient = |dF/dphi| at each phi.

    Returns: stiffness[n_phi] normalized to [0, 1].
    """
    print(f"  Computing vacuum stiffness LUT (n_phi={n_phi}, TWIST={twist})...",
          flush=True)
    phi_arr = np.linspace(0, 360, n_phi, endpoint=False)
    stiffness = np.zeros(n_phi)

    for i, phi in enumerate(phi_arr):
        rad = np.radians(phi)
        m   = np.cos(rad)
        lam = np.sin(rad)

        # Force components (from space_fractal.py)
        F_tm = -(m + 0.866)
        F_tl = -(lam - 0.5)
        F_rm = -m
        F_rl = -(lam + 1.0) + twist * np.sin(m * 2.5)

        sm = F_tm + F_rm
        sl = F_tl + F_rl
        mag = np.sqrt(sm**2 + sl**2)
        sf  = np.sqrt(max(float(mag), 0.0))
        gm  = sm * sf
        gl  = sl * sf

        # Basin weighting (Gaussian gates at 30°, 150°, 270°)
        def gw(a, c):
            diff = min(abs(a - c), 360 - abs(a - c))
            return np.exp(-((diff) / 80)**2)

        wg = gw(phi, 30); wt = gw(phi, 150); wr = gw(phi, 270)
        tot = wg + wt + wr + 1e-6
        nwr = wr / tot; nwt = wt / tot; nwg = wg / tot

        # Effective force magnitude at this phi
        Fm = nwt * F_tm + nwr * F_rm + nwg * gm
        Fl = nwt * F_tl + nwr * F_rl + nwg * gl

        # Stiffness = magnitude of effective force (proxy for |dF/dphi|)
        stiffness[i] = float(np.sqrt(Fm**2 + Fl**2))

    # Normalize
    s_min, s_max = stiffness.min(), stiffness.max()
    stiffness_norm = (stiffness - s_min) / (s_max - s_min + 1e-12)
    return phi_arr, stiffness_norm

def stiffness_at_j1(j1_deg: float, phi_arr: np.ndarray,
                    stiffness: np.ndarray) -> float:
    """Interpolate stiffness at a given J1 angle."""
    idx = np.argmin(np.abs(phi_arr - (j1_deg % 360)))
    return float(stiffness[idx])

# ── Logistic utilities ─────────────────────────────────────────────────────────

def logistic_orbit(r: float, x0: float = 0.5, n: int = 5000,
                   n_burn: int = 500) -> np.ndarray:
    x = float(x0)
    for _ in range(n_burn):
        x = r * x * (1.0 - x)
    orbit = np.empty(n - n_burn)
    for i in range(len(orbit)):
        x = r * x * (1.0 - x)
        orbit[i] = x
    return orbit

# ── Structured vs plain chaos detectors ───────────────────────────────────────

def permutation_entropy(x: np.ndarray, m: int = 5, tau: int = 1) -> float:
    """
    Permutation entropy of order m with delay tau.
    Lower PE = more structured/predictable.
    PE = 1.0 for white noise, < 1 for structured signals.
    """
    n = len(x)
    patterns = {}
    count = 0
    for i in range(n - (m - 1) * tau):
        window = x[i:i + m * tau:tau]
        perm   = tuple(np.argsort(window))
        patterns[perm] = patterns.get(perm, 0) + 1
        count += 1
    probs = np.array(list(patterns.values()), dtype=float) / count
    # Normalize by log(m!) for [0,1] range
    import math
    h = -np.sum(probs * np.log(probs + 1e-12)) / np.log(math.factorial(m))
    return float(h)

def recurrence_determinism(x: np.ndarray, eps: float = 0.05,
                            min_diag: int = 2, n_sample: int = 500) -> float:
    """
    Recurrence plot determinism (DET): fraction of recurrence points
    forming diagonal lines of length >= min_diag.
    Higher DET = more structured/periodic behavior.

    Uses a sample of n_sample points to keep computation tractable.
    """
    sample = x[:n_sample]
    n = len(sample)
    # Distance matrix
    diff = np.abs(sample[:, None] - sample[None, :])
    R    = (diff < eps).astype(np.int8)

    # Count diagonal lengths
    total_recurrence = R.sum()
    if total_recurrence == 0:
        return 0.0

    diag_points = 0
    for k in range(-(n-1), n):
        diag = np.diagonal(R, k)
        if len(diag) < min_diag:
            continue
        # Find runs of 1s
        run = 0
        for v in diag:
            if v:
                run += 1
                if run >= min_diag:
                    diag_points += 1
            else:
                run = 0

    return float(diag_points) / (total_recurrence + 1e-12)

def correlation_dimension(x: np.ndarray, n_sample: int = 1000,
                           r_vals: int = 20) -> float:
    """
    Correlation dimension D2 via Grassberger-Procaccia.
    D2 ≈ 0 for fixed point, D2 ≈ 1 for filled [0,1], D2 < 1 for Cantor-like sets.
    Uses embedding dimension 1 (scalar time series).
    """
    sample = x[:n_sample]
    n = len(sample)
    diff = np.abs(sample[:, None] - sample[None, :])

    # Log-log slope of C(r) vs r
    r_min, r_max = 1e-4, 0.5
    r_arr = np.logspace(np.log10(r_min), np.log10(r_max), r_vals)
    c_arr = np.array([(diff < r).sum() / (n * n) for r in r_arr])

    # Fit log-log slope on the middle portion (avoid saturation)
    mid = slice(r_vals // 4, 3 * r_vals // 4)
    log_r = np.log(r_arr[mid])
    log_c = np.log(c_arr[mid] + 1e-12)
    slope = float(np.polyfit(log_r, log_c, 1)[0])
    return slope   # This is D2

# ── Subcommand: stiffness_corr ─────────────────────────────────────────────────

def cmd_stiffness_corr(args):
    print("\n" + "="*70)
    print("STIFFNESS_CORR — Vacuum Stiffness × Token Alignment")
    print("="*70)

    # Load stiffness LUT
    phi_arr, stiffness = compute_stiffness_lut()

    # Load model and engram
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"\n  Loading model: {args.model}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()
    tok   = GPT2Tokenizer.from_pretrained(args.model)
    W     = model.lm_head.weight.detach().float().numpy()
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    with open(args.engram) as f:
        e = json.load(f)
    ksi_v = np.array(e["ksi_vals"], dtype=np.float32)
    j1_v  = np.array(e["j1_pca"],   dtype=np.float32) % 360.0
    pc1   = np.array(e["pc1"],       dtype=np.float32)
    pc2   = np.array(e["pc2"],       dtype=np.float32)

    print("\n  Stiffness field summary:")
    print(f"    Peak stiffness at phi = {phi_arr[np.argmax(stiffness)]:.1f}°")
    print(f"    Min stiffness at phi  = {phi_arr[np.argmin(stiffness)]:.1f}°")
    print(f"    Global attractor phi  = {phi_arr[np.argmin(np.abs(stiffness - stiffness.max()))]:.1f}°")

    # H-SW-001: for each probe, compute cosine scores and correlate with
    # stiffness at each token's J1 address
    print("\n  H-SW-001: Stiffness-token alignment per probe address")
    print(f"  {'Address':<22} {'stiff_j1':>9} {'rho':>7} {'p':>12} {'verdict':>8}")
    results = []
    h001_passes = 0

    for label, ksi_t, j1_t in PROBE_ADDRESSES:
        # Build probe hidden state
        j1_rad = np.radians(j1_t)
        h = np.cos(j1_rad) * pc1 + np.sin(j1_rad) * pc2
        h = h / (np.linalg.norm(h) + 1e-12)
        h_f64 = h.astype(np.float64)

        # Cosine scores for all tokens
        cos_all = W_norm @ h_f64    # [vocab]

        # Stiffness at each token's J1 address
        stiff_per_token = np.array([stiffness_at_j1(float(j1_v[i]), phi_arr, stiffness)
                                     for i in range(len(j1_v))], dtype=np.float32)

        # Correlation: stiffness at token J1 vs cosine score
        rho, p = stats.spearmanr(stiff_per_token, cos_all[:len(stiff_per_token)])
        stiff_here = stiffness_at_j1(j1_t, phi_arr, stiffness)

        h001 = "PASS" if (rho > 0.2 and p < 0.05) else "FAIL"
        if h001 == "PASS": h001_passes += 1

        print(f"  {label:<22} {stiff_here:>9.3f} {rho:>7.3f} {p:>12.3e} {h001:>8}")
        results.append({
            "label": label, "ksi_target": float(ksi_t), "j1_target": float(j1_t),
            "stiffness_at_j1": float(stiff_here),
            "rho": float(rho), "p_val": float(p), "h001": h001,
        })

    h001_overall = "PASS" if h001_passes >= 5 else "FAIL"
    print(f"\n  H-SW-001: {h001_passes}/10 → {h001_overall}")

    # Stiffness field: find peaks and troughs
    peaks  = [(phi_arr[i], float(stiffness[i]))
              for i in range(1, len(stiffness)-1)
              if stiffness[i] > stiffness[i-1] and stiffness[i] > stiffness[i+1]]
    troughs = [(phi_arr[i], float(stiffness[i]))
               for i in range(1, len(stiffness)-1)
               if stiffness[i] < stiffness[i-1] and stiffness[i] < stiffness[i+1]]
    peaks.sort(key=lambda x: -x[1])
    troughs.sort(key=lambda x: x[1])
    print(f"\n  Top 3 stiffness peaks: {[(f'{p[0]:.0f}°', f'{p[1]:.3f}') for p in peaks[:3]]}")
    print(f"  Top 3 stiffness troughs: {[(f'{t[0]:.0f}°', f'{t[1]:.3f}') for t in troughs[:3]]}")

    output = {
        "h001_overall": h001_overall, "h001_pass_rate": f"{h001_passes}/10",
        "results": results,
        "stiffness_peaks":  [(float(p[0]), float(p[1])) for p in peaks[:5]],
        "stiffness_troughs": [(float(t[0]), float(t[1])) for t in troughs[:5]],
        "stiffness_lut_sample": {
            "phi": [float(p) for p in phi_arr[::30]],
            "stiffness": [float(s) for s in stiffness[::30]],
        }
    }
    _save(output, args.output)
    print(f"\n  Saved → {args.output}")

# ── Subcommand: chaos_detect ───────────────────────────────────────────────────

def cmd_chaos_detect(args):
    """
    H-SW-002: Structured vs plain chaos at logistic r values corresponding
    to key Ksi addresses. Tests permutation entropy, recurrence determinism,
    and correlation dimension.
    """
    print("\n" + "="*70)
    print("CHAOS_DETECT — Structured vs Plain Chaos Fingerprints")
    print("="*70)

    def r_from_ksi(ksi, beta=0.438):
        return 3.56995 + (float(np.clip(ksi, 0, 1))**beta) * (4.0 - 3.56995)

    # Test points spanning the manifold
    test_cases = [
        ("period6 (deep basin)",   0.010, "period-6"),
        ("period5 (basin stable)", 0.119, "period-5"),
        ("period3 (outer basin)",  0.312, "period-3 adjacent"),
        ("alpha_physical",         0.397, "chaos onset"),
        ("native_english",         0.585, "structured chaos"),
        ("abstract_zone",          0.730, "structured chaos"),
        ("wada_boundary",          0.880, "deep chaos"),
        ("near_escape",            0.950, "deep chaos"),
        ("full_chaos",             0.999, "r=4 arcsine"),
    ]

    print(f"\n  {'Label':<28} {'r':>7} {'PE':>7} {'DET':>7} {'D2':>7} {'regime'}")
    print("  " + "-"*80)

    results = []
    n_orbit = 8000

    pe_native = None; pe_wada = None
    det_native = None; det_wada = None

    for label, ksi, regime in test_cases:
        r  = r_from_ksi(ksi)
        orbit = logistic_orbit(r, x0=0.5, n=n_orbit, n_burn=1000)

        pe  = permutation_entropy(orbit, m=5, tau=1)
        det = recurrence_determinism(orbit, eps=0.05, min_diag=2, n_sample=800)
        d2  = correlation_dimension(orbit, n_sample=800)

        if "native" in label: pe_native = pe; det_native = det
        if "wada"   in label: pe_wada   = pe; det_wada   = det

        print(f"  {label:<28} {r:>7.4f} {pe:>7.4f} {det:>7.4f} {d2:>7.4f}  {regime}")
        results.append({
            "label": label, "ksi": float(ksi), "r": float(r),
            "regime": regime, "PE": float(pe), "DET": float(det), "D2": float(d2),
        })

    # H-SW-002 verdict
    h002 = "NOT_TESTABLE"
    if pe_native is not None and pe_wada is not None:
        pe_ok  = pe_native < pe_wada
        det_ok = det_native > det_wada
        h002   = "PASS" if (pe_ok and det_ok) else "FAIL"
        print(f"\n  H-SW-002:")
        print(f"    PE: native={pe_native:.4f}, wada={pe_wada:.4f}, native<wada: {pe_ok}")
        print(f"    DET: native={det_native:.4f}, wada={det_wada:.4f}, native>wada: {det_ok}")
        print(f"    H-SW-002: {h002}")

    # Key insight: what makes structured chaos "structured"
    print("""
  INTERPRETATION:
  ─────────────────────────────────────────────────────────────────
  PE < 0.95 = structured (predictable symbol order in orbit)
  DET > 0.2 = structured (long diagonal runs in recurrence plot)
  D2 < 0.9  = not fully space-filling (finite attractor dimension)

  Period windows (Ksi < 0.4) have PE ≈ 0 and DET ≈ 1 (perfectly periodic).
  Structured chaos (Ksi 0.4-0.8) has intermediate PE, DET, D2.
  Wada boundary (Ksi > 0.85) approaches white noise (PE→1, DET→0, D2→1).
    """)

    output = {
        "h002_verdict": h002,
        "results": results,
    }
    _save(output, args.output)
    print(f"\n  Saved → {args.output}")

# ── Subcommand: feigenbaum_op ──────────────────────────────────────────────────

def cmd_feigenbaum_op(args):
    """
    H-SW-003: Does transformer depth follow Feigenbaum scaling?
    Measure alpha ratios between consecutive MLP c_proj layers.
    """
    print("\n" + "="*70)
    print("FEIGENBAUM_OP — Feigenbaum Scaling Across Transformer Depth")
    print("="*70)

    import torch
    from transformers import GPT2LMHeadModel
    from sklearn.utils.extmath import randomized_svd

    print(f"\n  Loading model: {args.model}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()

    n_layers = model.config.n_layer   # 36 for GPT-2-Large
    n_sv = 50

    print(f"  Extracting IFS alphas for {n_layers} MLP c_proj layers...", flush=True)
    alphas = []
    for l in range(n_layers):
        W = model.transformer.h[l].mlp.c_proj.weight.detach().float().numpy()
        _, sv, _ = randomized_svd(W, n_components=n_sv, random_state=42)
        # Overall alpha (log-log slope)
        log_r = np.log(np.arange(1, n_sv+1, dtype=float))
        log_s = np.log(sv + 1e-12)
        slope = float(np.polyfit(log_r, log_s, 1)[0])
        alphas.append(-slope)
        if l % 12 == 0:
            print(f"    layer {l:2d}: alpha={-slope:.4f}", flush=True)

    alphas = np.array(alphas)

    # Consecutive ratios
    ratios_forward = alphas[:-1] / (alphas[1:] + 1e-12)   # alpha_l / alpha_{l+1}
    ratios_inverse = alphas[1:]  / (alphas[:-1] + 1e-12)  # alpha_{l+1} / alpha_l

    mean_fwd = float(ratios_forward.mean())
    std_fwd  = float(ratios_forward.std())
    mean_inv = float(ratios_inverse.mean())

    print(f"\n  Alpha sequence (first 6 and last 6):")
    for i, a in enumerate(alphas[:6]):
        print(f"    layer {i:2d}: alpha={a:.4f}")
    print("    ...")
    for i, a in enumerate(alphas[-6:]):
        print(f"    layer {n_layers-6+i:2d}: alpha={a:.4f}")

    print(f"\n  Consecutive ratio alpha_l / alpha_{{l+1}}:")
    print(f"    mean = {mean_fwd:.4f}, std = {std_fwd:.4f}")
    print(f"    Feigenbaum alpha_inv = {FEIGENBAUM_ALPHA_INV:.4f}")
    print(f"    Feigenbaum alpha     = {FEIGENBAUM_ALPHA:.4f}")
    print(f"    Gap to alpha_inv: {abs(mean_fwd - FEIGENBAUM_ALPHA_INV):.4f}")
    print(f"    Gap to alpha:     {abs(mean_fwd - FEIGENBAUM_ALPHA):.4f}")

    # H-SW-003 verdict
    gap_inv = abs(mean_fwd - FEIGENBAUM_ALPHA_INV)
    gap_fwd = abs(mean_fwd - FEIGENBAUM_ALPHA)
    h003 = "PASS_INV" if gap_inv < 0.1 else ("PASS_FWD" if gap_fwd < 0.5 else "FAIL")
    print(f"\n  H-SW-003: {h003}")

    # Feigenbaum vector in IFS space
    # If the ratio is meaningful, we can define a "Feigenbaum direction" in
    # IFS alpha space: d_F = [1, 1/alpha_F, 1/alpha_F^2, ...]
    # This vector points from any layer toward the next self-similar layer.
    feig_vec = np.array([FEIGENBAUM_ALPHA_INV ** i for i in range(n_layers)])
    feig_vec /= np.linalg.norm(feig_vec)

    # Project actual alpha sequence onto Feigenbaum direction
    alpha_norm = (alphas - alphas.mean()) / (alphas.std() + 1e-12)
    feig_projection = float(np.dot(alpha_norm, feig_vec[:len(alpha_norm)]))
    print(f"\n  Projection of actual alpha sequence onto Feigenbaum direction:")
    print(f"    dot product = {feig_projection:.4f}")
    print(f"    (0 = orthogonal, ±1 = aligned)")

    output = {
        "n_layers": int(n_layers),
        "alphas": [float(a) for a in alphas],
        "ratios_forward": [float(r) for r in ratios_forward],
        "mean_ratio": float(mean_fwd), "std_ratio": float(std_fwd),
        "feigenbaum_alpha_inv": float(FEIGENBAUM_ALPHA_INV),
        "feigenbaum_alpha":     float(FEIGENBAUM_ALPHA),
        "gap_to_alpha_inv": float(gap_inv),
        "gap_to_alpha":     float(gap_fwd),
        "h003_verdict": h003,
        "feig_vec_projection": float(feig_projection),
    }
    _save(output, args.output)
    print(f"\n  Saved → {args.output}")

# ── Subcommand: full_run ───────────────────────────────────────────────────────

def cmd_full_run(args):
    import os

    class NS:
        def __init__(self, **kw): self.__dict__.update(kw)

    t1 = args.output.replace(".json", "_stiffness.json")
    t2 = args.output.replace(".json", "_chaos.json")
    t3 = args.output.replace(".json", "_feig.json")

    cmd_stiffness_corr(NS(model=args.model, engram=args.engram, output=t1))
    cmd_chaos_detect(NS(output=t2))
    cmd_feigenbaum_op(NS(model=args.model, output=t3))

    with open(t1) as f: d1 = json.load(f)
    with open(t2) as f: d2 = json.load(f)
    with open(t3) as f: d3 = json.load(f)

    report = {
        "checkpoint": "SW-001",
        "title": "Vacuum Stiffness × Weight Probe",
        "summary": {
            "H-SW-001": d1["h001_overall"],
            "H-SW-002": d2["h002_verdict"],
            "H-SW-003": d3["h003_verdict"],
        },
        "stiffness": d1, "chaos": d2, "feigenbaum": d3
    }
    _save(report, args.output)
    for f in [t1, t2, t3]:
        try: os.remove(f)
        except: pass
    print(f"\n  Full report saved → {args.output}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(obj, path):
    def _cvt(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.int32, np.int64)): return int(o)
        if isinstance(o, (np.float32, np.float64)): return float(o)
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_cvt)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Vacuum Stiffness × Weight Probe")
    sub = ap.add_subparsers(dest="cmd")

    p1 = sub.add_parser("stiffness_corr")
    p1.add_argument("--model",  required=True)
    p1.add_argument("--engram", required=True)
    p1.add_argument("--output", default="sw_stiffness.json")

    p2 = sub.add_parser("chaos_detect")
    p2.add_argument("--output", default="sw_chaos.json")

    p3 = sub.add_parser("feigenbaum_op")
    p3.add_argument("--model",  required=True)
    p3.add_argument("--output", default="sw_feigenbaum.json")

    p4 = sub.add_parser("full_run")
    p4.add_argument("--model",  required=True)
    p4.add_argument("--engram", required=True)
    p4.add_argument("--output", default="sw_full.json")

    args = ap.parse_args()
    dispatch = {
        "stiffness_corr": cmd_stiffness_corr,
        "chaos_detect":   cmd_chaos_detect,
        "feigenbaum_op":  cmd_feigenbaum_op,
        "full_run":       cmd_full_run,
    }
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
