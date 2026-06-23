"""
vessel_v1_gpt2.py
GPT-2 Hook for vessel_v1.py
Pirouette Framework Volume 8 · CORE-003 · ML-065

WHAT THIS DOES
==============
Replaces the three toy stubs in vessel_v1.py with real GPT-2-Large calls:

  toy_token_addrs → real_token_addrs(engram)
    Each of the 50,257 vocab tokens gets its real (J1, Ksi) address
    from the engram_curve.json. This is a one-time load.

  toy_logits → real_logits(h, W_n)
    The biased softmax operates over real LM_Head dot products:
    logits[v] = W_n[v] @ h  for the current hidden state h.

  eval_h001_h002 → real_eval(generated_tokens, tok, ksi_traj)
    h001: mean word length > 3.5 AND single-letter fraction < 0.20
    h002: alpha_std > 0.15 (genuine basin navigation)
    These match the leyline_prime definitions — comparable across runs.

The Vessel class, null battery, and leyline subcommand run UNCHANGED from
vessel_v1.py. This file imports vessel_v1 and monkey-patches the run function.

NEW SUBCOMMAND: gpt2
  Runs a single generation with the real GPT-2-Large model, vessel-steered.
  Starts from a certified seed token (default: ' of' at Zone A address).
  Outputs text + all vessel diagnostics.

NEW SUBCOMMAND: gpt2_null
  Runs the three null conditions on real GPT-2-Large:
    canonical     : full vessel (beta_wet=1.0, logprob deposits, real LUTs)
    N_vessel_vs_state : base overwritten by hidden-state address each step
    N_substructure    : cloud reshuffled (centroid preserved, links destroyed)
  This is the definitive test of whether vessel history helps real generation.

NEW SUBCOMMAND: gpt2_leyline
  Three conditions on real GPT-2-Large:
    A: dual_free baseline (leyline_gain=0)
    B: logistic itinerary (certified orbit drives kick-scheduling)
    C: scrambled itinerary (sequence destroyed, marginal preserved)
  Headline result: does B beat C? (structure_is_loadbearing)

Usage:
  python vessel_v1_gpt2.py gpt2 ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --helicity-lut helicity_lut.npy ^
    --stiffness-lut stiffness_lut.npy ^
    --steps 120 --out vessel_gpt2_run.json

  python vessel_v1_gpt2.py gpt2_null ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --helicity-lut helicity_lut.npy ^
    --stiffness-lut stiffness_lut.npy ^
    --steps 120 --null-reps 3 --out vessel_gpt2_null.json

  python vessel_v1_gpt2.py gpt2_leyline ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --helicity-lut helicity_lut.npy ^
    --stiffness-lut stiffness_lut.npy ^
    --steps 120 --null-reps 3 --leyline-gain 0.15 --out vessel_gpt2_leyline.json
"""

import argparse
import json
import sys
import os
import time
import numpy as np

# ── Import vessel_v1 ──────────────────────────────────────────────────────────
# vessel_v1.py must be in the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vessel_v1 as V

# ── Constants ──────────────────────────────────────────────────────────────────
VOCAB_SIZE = 50257
K_PROJ     = 16
TEMPERATURE = 0.82

# Certified seed: ' of' at Zone A address (Ksi=0.17, J1=72°)
# Best combined score from scan_map.json
SEED_HINT  = " of"
SEED_KSI   = 0.17
SEED_J1    = 72.0

# ── Model loading (cached globally so multi-run null doesn't reload) ──────────
_model_cache = {}

def load_model(model_path: str):
    if model_path in _model_cache:
        return _model_cache[model_path]
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"  [model] loading {model_path}...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    tok   = GPT2Tokenizer.from_pretrained(model_path)
    W_lm  = model.lm_head.weight.detach().float().numpy()
    W_n   = W_lm / (np.linalg.norm(W_lm, axis=1, keepdims=True) + 1e-12)
    _model_cache[model_path] = (model, tok, W_n)
    print(f"  [model] loaded. W_n shape: {W_n.shape}", flush=True)
    return model, tok, W_n

# ── Engram loading (cached) ────────────────────────────────────────────────────
_engram_cache = {}

def load_engram(path: str):
    if path in _engram_cache:
        return _engram_cache[path]
    with open(path) as f:
        e = json.load(f)
    out = {
        "ksi_vals": np.array(e["ksi_vals"],  dtype=np.float32),
        "j1_360":   np.array(e["j1_pca"],    dtype=np.float32) % 360.0,
        "pc1":      np.array(e["pc1"],        dtype=np.float32),
        "pc2":      np.array(e["pc2"],        dtype=np.float32),
        "U_ref":    np.array(e["U_ref"],      dtype=np.float32),
    }
    _engram_cache[path] = out
    print(f"  [engram] loaded. vocab={len(out['ksi_vals'])}", flush=True)
    return out

# ── Real replacements for toy stubs ───────────────────────────────────────────

def real_token_addrs(engram) -> np.ndarray:
    """J1 address for every vocab token, from the engram. Shape [50257]."""
    return engram["j1_360"].astype(np.float64)


def real_logits(h: np.ndarray, W_n: np.ndarray) -> np.ndarray:
    """Real LM_Head logits: W_n @ h. Shape [vocab_size]."""
    return (W_n @ h.astype(np.float64)).astype(np.float64)


def real_eval_h001_h002(generated_tokens: list, tok, ksi_traj: list,
                         alpha_traj: list) -> tuple:
    """
    Real h001/h002 matching leyline_prime definitions.
    h001: mean_word_length > 3.5 AND single_letter_fraction < 0.20
    h002: alpha_std > 0.15  (genuine basin navigation)
    """
    word_lens = []
    for tid in generated_tokens:
        try:
            s = tok.decode([int(tid)], skip_special_tokens=False).strip()
            if any(c.isalpha() for c in s):
                word_lens.append(len(s))
        except Exception:
            pass
    mean_wl    = float(np.mean(word_lens)) if word_lens else 0.0
    single_let = sum(1 for l in word_lens if l == 1) / (len(word_lens) + 1e-12)
    alpha_std  = float(np.std(alpha_traj)) if alpha_traj else 0.0
    h001 = bool(mean_wl > 3.5 and single_let < 0.20)
    h002 = bool(alpha_std > 0.15)
    return h001, h002, mean_wl, single_let, alpha_std


def hidden_address(h: np.ndarray, U_ref, pc1, pc2) -> tuple:
    """Extract (Ksi, J1) from a hidden state vector."""
    h_n  = h / (np.linalg.norm(h) + 1e-12)
    proj = U_ref.T @ h_n
    p    = proj**2 / (proj**2).sum() + 1e-12
    ksi  = float(-np.sum(p * np.log(p + 1e-12)) / np.log(K_PROJ))
    j1   = float(np.degrees(np.arctan2(
        float(np.dot(h_n, pc1)), float(np.dot(h_n, pc2)))) % 360)
    return ksi, j1


def get_seed_hidden(model, tok, engram, W_n, hint=SEED_HINT,
                    ksi_t=SEED_KSI, j1_t=SEED_J1):
    """Get a real 1280-dim seed hidden state via one forward pass."""
    import torch
    ksi_v = engram["ksi_vals"]; j1_v = engram["j1_360"]
    sid = None
    if hint:
        try:
            s = hint if hint.startswith(' ') else f' {hint}'
            sid = tok.encode(s)[0]
        except Exception:
            pass
    if sid is None:
        d = np.abs(ksi_v - ksi_t) + 0.3 * np.minimum(
            np.abs(j1_v - j1_t) % 360, 360 - np.abs(j1_v - j1_t) % 360) / 180
        sid = int(np.argmin(d))
    with torch.no_grad():
        out = model(torch.tensor([[sid]]), output_hidden_states=True)
    h_raw = out.hidden_states[-1][0, -1].float().numpy()
    ksi_h, j1_h = hidden_address(h_raw, engram["U_ref"],
                                  engram["pc1"], engram["pc2"])
    seed_str = tok.decode([sid], skip_special_tokens=False)
    return sid, seed_str, h_raw, ksi_h, j1_h

# ── Core GPT-2 generation with Vessel ─────────────────────────────────────────

def run_gpt2_vessel(steps, model, tok, W_n, engram, knobs, k_cloud, seed,
                    helicity, stiffness,
                    reshuffle=False, use_state_addr=False,
                    leyline=None, leyline_scramble=False,
                    seed_hint=SEED_HINT, verbose=True, 
                    arc_prior=None):
    """
    Full GPT-2-Large generation with Vessel steering.
    Direct replacement for vessel_v1.run_generation with real model.

    The Vessel does the same three things as in toy mode:
      1. roll() : base drifts under stiffness potential
      2. softmax_bias(tok_addrs) : wet cloud mass biases logits
      3. kick(logprob_weight) : token quality deposits into base velocity

    The only change: logits come from W_n @ h (real LM_Head), not rng.normal.
    """
    import torch

    rng = np.random.default_rng(seed)
    itinerary = None
    if leyline == "logistic":
        itinerary, _ = V.logistic_itinerary(steps, ksi_seed=0.585, rng=rng,
                                             scramble=leyline_scramble)

    vessel = V.Vessel(k_cloud, helicity, stiffness, knobs, rng,
                      reshuffle_offsets=reshuffle,
                      use_state_addr=use_state_addr,
                      itinerary=itinerary)

    # Real token J1 addresses for all 50,257 vocab tokens
    tok_addrs = real_token_addrs(engram)

    # Seed hidden state
    sid, seed_str, h_raw, ksi_h, j1_h = get_seed_hidden(
        model, tok, engram, W_n, hint=seed_hint)

    # Override vessel base with seed address
    vessel.base[V.J1]  = j1_h
    vessel.base[V.KSI] = ksi_h

    h_ln = h_raw / (np.linalg.norm(h_raw) + 1e-12)

    generated_ids   = []
    generated_text  = ""
    alpha_traj      = []
    ksi_traj        = [ksi_h]
    lockin_list     = []
    garbage_count   = 0
    recent_30       = []

    if verbose:
        print(f"  [vessel] seed='{seed_str}'  Ksi={ksi_h:.3f}, J1={j1_h:.1f}°")
        print(f"  [vessel] beta_wet={knobs['beta_wet']:.2f}  "
              f"decay={knobs['decay']:.2f}  k_cloud={k_cloud}")
        print(f"\n  ", end="", flush=True)

    for s in range(steps):
        vessel.roll()
        vessel.record_basin()

        # Real logits from LM_Head
        base_logits = real_logits(h_ln, W_n)   # [50257]

        # Vessel wet-cloud bias (additive in logit space)
        bias = vessel.softmax_bias(tok_addrs)    # [50257]
        biased = base_logits + bias

        # ── NEW: ARC PRIOR INTEGRATION ──
        if arc_prior is not None:
            # The vessel's target arc is derived from its current heading:
            # J1 target is velocity * dt. Ksi target is the drift toward the vessel base.
            target_dj1 = float(vessel.vel[V.J1] * knobs["dt"])
            target_dksi = float(vessel.base[V.KSI] - ksi_h)

            sigma_j1 = 20.0
            sigma_ksi = 0.1

            # Calculate compatibility (ignoring NaNs for tokens without arc data)
            arc_compat = np.exp(-((arc_prior[:, 0] - target_dj1) / sigma_j1)**2 
                                - ((arc_prior[:, 1] - target_dksi) / sigma_ksi)**2)
            arc_compat = np.nan_to_num(arc_compat, nan=0.0)

            # Apply beta_arc
            biased += knobs["beta_arc"] * np.log1p(arc_compat)
        # ────────────────────────────────

        # Repetition penalty
        for tid in set(recent_30):
            biased[tid] *= 0.80

        # Sample
        b_shift = biased - biased.max()
        probs   = np.exp(b_shift / TEMPERATURE); probs /= probs.sum()
        chosen  = int(rng.choice(VOCAB_SIZE, p=probs))

        # Logprob weight for deposit (sigmoid of token log-probability)
        log_z  = np.log(np.sum(np.exp(b_shift))) + biased.max()
        logprob = float(biased[chosen]) - float(log_z)
        logprob_weight = float(1.0 / (1.0 + np.exp(-logprob)))

        # Decode
        token_str = tok.decode([chosen], skip_special_tokens=False)
        generated_ids.append(chosen)
        generated_text += token_str
        recent_30.append(chosen)
        if len(recent_30) > 30: recent_30.pop(0)
        if '\ufffd' in token_str: garbage_count += 1

        if verbose:
            print(token_str, end="", flush=True)

        # Update vessel state
        if use_state_addr:
            vessel.set_base_from_state(float(tok_addrs[chosen]), ksi_h)
        vessel.kick(logprob_weight)
        addrs, w = vessel.wetness()
        vessel.reinforce_and_link(addrs, w)

        # Alpha from vessel base position (matches vessel_v1 proxy formula)
        alpha = float(0.45 + 0.30 * V.lut_at(stiffness, vessel.base[V.J1]))
        alpha_traj.append(alpha)
        lockin_list.append(vessel.lockin_entropy())

        # Update hidden state via forward pass
        with torch.no_grad():
            out  = model(torch.tensor([[chosen]]), output_hidden_states=True)
        h_raw = out.hidden_states[-1][0, -1].float().numpy()
        h_ln  = h_raw / (np.linalg.norm(h_raw) + 1e-12)
        ksi_h, j1_h = hidden_address(h_raw, engram["U_ref"],
                                      engram["pc1"], engram["pc2"])
        ksi_traj.append(ksi_h)

        # Also update vessel Ksi (drifts toward current hidden-state Ksi)
        vessel.base[V.KSI] += 0.05 * (ksi_h - vessel.base[V.KSI])

    if verbose:
        print("\n")

    h001, h002, mean_wl, single_let, alpha_std = real_eval_h001_h002(
        generated_ids, tok, ksi_traj, alpha_traj)

    coh = sum(
        1 for tid in generated_ids
        if '\ufffd' not in tok.decode([tid])
        and any(c.isalpha() for c in tok.decode([tid]))
        and all(ord(c) < 256 for c in tok.decode([tid]))
    ) / (steps + 1e-12)

    return {
        "generated_text":        generated_text,
        "n_tokens":              steps,
        "garbage_count":         int(garbage_count),
        "coherence":             float(coh),
        "mean_word_length":      float(mean_wl),
        "single_letter_fraction": float(single_let),
        "fertile_hit_rate":      float(sum(
            1 for a in [float(tok_addrs[t]) for t in generated_ids]
            if 60 <= a <= 140) / (steps + 1e-12)),
        "mean_lockin_entropy":   float(np.nanmean(
            [x for x in lockin_list if x is not None]) or 0.0),
        "final_lockin_entropy":  lockin_list[-1],
        "n_origin_links":        len(vessel.links),
        "alpha_mean":            float(np.mean(alpha_traj)),
        "alpha_std":             float(alpha_std),
        "alpha_trajectory":      [float(a) for a in alpha_traj],
        "ksi_mean":              float(np.mean(ksi_traj)),
        "ksi_std":               float(np.std(ksi_traj)),
        "ksi_trajectory":        [float(k) for k in ksi_traj],
        "h001_pass":             bool(h001),
        "h002_pass":             bool(h002),
        "dual_pass":             bool(h001 and h002),
    }

# ── Subcommand: gpt2 ──────────────────────────────────────────────────────────

def cmd_gpt2(args):
    print("\n" + "="*70)
    print("VESSEL GPT2 — Single Generation Run")
    print("="*70)
    model, tok, W_n = load_model(args.model)
    engram = load_engram(args.engram)
    hel = V.load_lut(args.helicity_lut, "helicity")
    stf = V.load_lut(args.stiffness_lut, "stiffness")
    knobs = _make_knobs(args)
    print(f"  knobs: {knobs}", flush=True)

    # ── NEW: Load arc prior ──
    arc_prior = None
    if getattr(args, "arc_prior", None) and os.path.exists(args.arc_prior):
        arc_prior = np.load(args.arc_prior)
        print(f"  [arc] loaded prior from {args.arc_prior}", flush=True)

    t0  = time.time()
    res = run_gpt2_vessel(
        args.steps, model, tok, W_n, engram, knobs, args.k_cloud,
        args.seed, hel, stf,
        leyline=getattr(args, "leyline", None),
        leyline_scramble=False,
        verbose=True,
        arc_prior=arc_prior
    )
    res["knobs"]     = knobs
    res["elapsed_s"] = float(time.time() - t0)

    print(f"  coherence={res['coherence']:.3f}  "
          f"garbage={res['garbage_count']}  "
          f"mean_wl={res['mean_word_length']:.2f}  "
          f"links={res['n_origin_links']}  "
          f"lockin={res['mean_lockin_entropy']:.3f}")
    print(f"  h001={res['h001_pass']}  h002={res['h002_pass']}  "
          f"DUAL={res['dual_pass']}")
    V.dump_json(res, args.out)

# ── Subcommand: gpt2_null ─────────────────────────────────────────────────────

def cmd_gpt2_null(args):
    """
    Real null battery: canonical vs N_vessel_vs_state vs N_substructure.
    The key question: does vessel HISTORY help real generation?
    """
    print("\n" + "="*70)
    print("VESSEL GPT2 NULL — Three Conditions on Real GPT-2-Large")
    print("="*70)
    print("  Conditions:")
    print("  canonical          : full vessel (beta_wet=1.0, real deposits)")
    print("  N_vessel_vs_state  : base overwritten by hidden-state each step")
    print("  N_substructure     : cloud reshuffled (centroid preserved)")
    print()

    model, tok, W_n = load_model(args.model)
    engram = load_engram(args.engram)
    hel = V.load_lut(args.helicity_lut, "helicity")
    stf = V.load_lut(args.stiffness_lut, "stiffness")
    knobs = _make_knobs(args)
    print(f"  knobs: {knobs}", flush=True)

    # ── NEW: Load arc prior ──
    arc_prior = None
    if getattr(args, "arc_prior", None) and os.path.exists(args.arc_prior):
        arc_prior = np.load(args.arc_prior)
        print(f"  [arc] loaded prior from {args.arc_prior}", flush=True)
    reps  = args.null_reps

    def avg_condition(reshuffle=False, use_state=False, label=""):
        print(f"\n  --- {label} (reps={reps}) ---")
        hits, h1s, h2s, duals, links, lockouts, wls = [], [], [], [], [], [], []
        for sd in range(args.seed, args.seed + reps):
            r = run_gpt2_vessel(
                args.steps, model, tok, W_n, engram, knobs, args.k_cloud,
                sd, hel, stf,
                reshuffle=reshuffle, use_state_addr=use_state,
                verbose=(sd == args.seed)   # verbose on first rep only
            )
            hits.append(r["fertile_hit_rate"])
            h1s.append(int(r["h001_pass"]))
            h2s.append(int(r["h002_pass"]))
            duals.append(int(r["dual_pass"]))
            links.append(r["n_origin_links"])
            lockouts.append(r["mean_lockin_entropy"])
            wls.append(r["mean_word_length"])
            print(f"    seed={sd}  coh={r['coherence']:.3f}  "
                  f"wl={r['mean_word_length']:.2f}  "
                  f"links={r['n_origin_links']}  "
                  f"lockin={r['mean_lockin_entropy']:.3f}  "
                  f"h001={r['h001_pass']}  h002={r['h002_pass']}", flush=True)
        return {
            "label":           label,
            "fertile_hit":     float(np.mean(hits)),
            "fertile_hit_std": float(np.std(hits)),
            "h001_frac":       float(np.mean(h1s)),
            "h002_frac":       float(np.mean(h2s)),
            "dual_frac":       float(np.mean(duals)),
            "mean_links":      float(np.mean(links)),
            "mean_lockin":     float(np.mean(lockouts)),
            "mean_wl":         float(np.mean(wls)),
        }

    canonical  = avg_condition(label="canonical")
    n_state    = avg_condition(use_state=True, label="N_vessel_vs_state")
    n_sub      = avg_condition(reshuffle=True, label="N_substructure")

    verdicts = {
        "vessel_beats_state":   bool(canonical["dual_frac"] >
                                     n_state["dual_frac"] + canonical["fertile_hit_std"]),
        "substructure_matters": bool(canonical["dual_frac"] >
                                     n_sub["dual_frac"] + canonical["fertile_hit_std"]),
    }

    print("\n" + "="*70)
    print("NULL BATTERY RESULTS")
    print(f"  {'Condition':<24} {'fertile':>8} {'h001':>6} {'h002':>6} "
          f"{'DUAL':>6} {'links':>7} {'lockin':>7} {'wl':>6}")
    print("  " + "-"*72)
    for r in [canonical, n_state, n_sub]:
        print(f"  {r['label']:<24} {r['fertile_hit']:>8.3f} "
              f"{r['h001_frac']:>6.2f} {r['h002_frac']:>6.2f} "
              f"{r['dual_frac']:>6.2f} {r['mean_links']:>7.1f} "
              f"{r['mean_lockin']:>7.3f} {r['mean_wl']:>6.2f}")
    print()
    print("  Verdicts:")
    for k, v in verdicts.items():
        print(f"    {k:<28}: {'PASS' if v else 'FAIL'}")

    out = {
        "knobs": knobs, "null_reps": reps,
        "canonical":         canonical,
        "N_vessel_vs_state": n_state,
        "N_substructure":    n_sub,
        "verdicts":          verdicts,
    }
    V.dump_json(out, args.out)

# ── Subcommand: gpt2_leyline ──────────────────────────────────────────────────

def cmd_gpt2_leyline(args):
    """
    Lock-and-key test on real GPT-2-Large.
    A: dual_free (leyline_gain=0)
    B: logistic itinerary + leyline pull
    C: scrambled itinerary (null — same values, sequence destroyed)

    THE LOAD-BEARING QUESTION:
    Does B beat C? If yes, the logistic SEQUENCE (period windows / chaos structure)
    is doing real work, not just the marginal distribution of target addresses.
    This is what 'May those we' was pointing at: the period-5 window holding
    register long enough for grammar to emerge.
    """
    print("\n" + "="*70)
    print("VESSEL GPT2 LEYLINE — Lock-and-Key Test")
    print("="*70)
    print("  A: dual_free baseline  (leyline_gain=0)")
    print("  B: logistic itinerary  (certified orbit)")
    print("  C: scrambled itinerary (sequence destroyed = null)")
    print()

    model, tok, W_n = load_model(args.model)
    engram = load_engram(args.engram)
    hel = V.load_lut(args.helicity_lut, "helicity")
    stf = V.load_lut(args.stiffness_lut, "stiffness")
    knobs_base = _make_knobs(args)
    reps = args.null_reps
    gain = getattr(args, "leyline_gain", 0.15)

    def avg(leyline, gain_val, scramble=False, label=""):
        print(f"\n  --- {label} (reps={reps}, gain={gain_val:.2f}) ---")
        knobs = dict(knobs_base); knobs["leyline_gain"] = float(gain_val)
        # ── NEW: Load arc prior ──
        arc_prior = None
        if getattr(args, "arc_prior", None) and os.path.exists(args.arc_prior):
            arc_prior = np.load(args.arc_prior)
            print(f"  [arc] loaded prior from {args.arc_prior}", flush=True)
        h1s, h2s, duals = [], [], []
        ams, astds, wls, texts = [], [], [], []
        for sd in range(args.seed, args.seed + reps):
            r = run_gpt2_vessel(
                args.steps, model, tok, W_n, engram, knobs, args.k_cloud,
                sd, hel, stf,
                leyline=leyline, leyline_scramble=scramble,
                verbose=(sd == args.seed)
            )
            h1s.append(int(r["h001_pass"]))
            h2s.append(int(r["h002_pass"]))
            duals.append(int(r["dual_pass"]))
            ams.append(r["alpha_mean"])
            astds.append(r["alpha_std"])
            wls.append(r["mean_word_length"])
            texts.append(r["generated_text"])
            print(f"    seed={sd}  wl={r['mean_word_length']:.2f}  "
                  f"alpha={r['alpha_mean']:.3f}±{r['alpha_std']:.3f}  "
                  f"dual={r['dual_pass']}", flush=True)
        return {
            "label":      label,
            "h001_frac":  float(np.mean(h1s)),
            "h002_frac":  float(np.mean(h2s)),
            "dual_frac":  float(np.mean(duals)),
            "alpha_mean": float(np.mean(ams)),
            "alpha_std":  float(np.mean(astds)),
            "mean_wl":    float(np.mean(wls)),
            "sample_text": texts[0][:200] if texts else "",
        }

    A = avg(None, 0.0, label="A_dual_free")
    B = avg("logistic", gain, label="B_logistic")
    C = avg("logistic", gain, scramble=True, label="C_scrambled")

    verdicts = {
        "B_achieves_dual_pass":     bool(B["dual_frac"] > 0.5),
        "B_beats_baseline":         bool(B["dual_frac"] > A["dual_frac"]),
        "structure_is_loadbearing": bool(B["dual_frac"] > C["dual_frac"] + 0.1),
    }

    print("\n" + "="*70)
    print("LEYLINE RESULTS")
    print(f"  {'Label':<20} {'h001':>6} {'h002':>6} {'DUAL':>6} "
          f"{'alpha':>8} {'alpha_std':>10} {'wl':>6}")
    print("  " + "-"*65)
    for r in [A, B, C]:
        print(f"  {r['label']:<20} {r['h001_frac']:>6.2f} {r['h002_frac']:>6.2f} "
              f"{r['dual_frac']:>6.2f} {r['alpha_mean']:>8.3f} "
              f"{r['alpha_std']:>10.4f} {r['mean_wl']:>6.2f}")
    print()
    print("  Sample outputs:")
    for r in [A, B, C]:
        print(f"  [{r['label']}]: {r['sample_text'][:120]}")
    print()
    print("  Verdicts:")
    for k, v in verdicts.items():
        print(f"    {k:<30}: {'PASS' if v else 'FAIL'}")
    print()
    print("  NOTE: structure_is_loadbearing (B > C+0.1) is the")
    print("  definitive test. Speed/wl improvement alone is not evidence.")

    out = {
        "knobs": knobs_base, "leyline_gain": float(gain), "reps": reps,
        "A_dual_free": A, "B_logistic": B, "C_scrambled": C,
        "verdicts": verdicts,
    }
    V.dump_json(out, args.out)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_knobs(args):
    return {
        "diffusivity_mode":  getattr(args, "diffusivity_mode", "inverse"),
        "beta_wet":          getattr(args, "beta_wet",         1.0),
        "decay":             getattr(args, "decay",            0.1),
        "deposit_gain":      getattr(args, "deposit_gain",     1.0),
        "reinforce_lr":      getattr(args, "reinforce_lr",     0.2),
        "cloud_sigma":       getattr(args, "cloud_sigma",      8.0),
        "bias_bandwidth_deg":getattr(args, "bias_bandwidth_deg", 12.0),
        "dt":                getattr(args, "dt",               0.5),
        "leyline_gain":      getattr(args, "leyline_gain",     0.0),
        "beta_arc":          getattr(args, "beta_arc",         0.5), # <-- Add this
    }

def main():
    ap = argparse.ArgumentParser(
        description="Vessel v1 — GPT-2-Large Hook")
    ap.add_argument("cmd", choices=["gpt2", "gpt2_null", "gpt2_leyline"])
    ap.add_argument("--model",             required=True)
    ap.add_argument("--engram",            required=True)
    ap.add_argument("--helicity-lut",      default="helicity_lut.npy",
                    dest="helicity_lut")
    ap.add_argument("--stiffness-lut",     default="stiffness_lut.npy",
                    dest="stiffness_lut")
    ap.add_argument("--steps",             type=int,   default=120)
    ap.add_argument("--k-cloud",           type=int,   default=12, dest="k_cloud")
    ap.add_argument("--seed",              type=int,   default=0)
    ap.add_argument("--null-reps",         type=int,   default=3,  dest="null_reps")
    ap.add_argument("--beta-wet",          type=float, default=1.0, dest="beta_wet")
    ap.add_argument("--decay",             type=float, default=0.1)
    ap.add_argument("--deposit-gain",      type=float, default=1.0, dest="deposit_gain")
    ap.add_argument("--reinforce-lr",      type=float, default=0.2, dest="reinforce_lr")
    ap.add_argument("--cloud-sigma",       type=float, default=8.0, dest="cloud_sigma")
    ap.add_argument("--bias-bandwidth-deg",type=float, default=12.0,
                    dest="bias_bandwidth_deg")
    ap.add_argument("--dt",                type=float, default=0.5)
    ap.add_argument("--diffusivity-mode",  default="inverse",
                    choices=["inverse","inverse_sq","exp","flat"],
                    dest="diffusivity_mode")
    ap.add_argument("--leyline",           default=None,
                    choices=["logistic"])
    ap.add_argument("--leyline-gain",      type=float, default=0.15,
                    dest="leyline_gain")
    ap.add_argument("--arc-prior",         default=None, 
                    help="Path to arc_prior.npy")                     # <-- Add this
    ap.add_argument("--beta-arc",          type=float, default=0.5, 
                    dest="beta_arc")                                  # <-- Add this
    ap.add_argument("--out",               default="vessel_gpt2_run.json")

    args = ap.parse_args()
    {"gpt2":         cmd_gpt2,
     "gpt2_null":    cmd_gpt2_null,
     "gpt2_leyline": cmd_gpt2_leyline}[args.cmd](args)

if __name__ == "__main__":
    main()
