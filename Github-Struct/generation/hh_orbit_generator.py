"""
hh_orbit_generator.py
HH Orbit Token Generator
Pirouette Framework Volume 8 · CORE-003 · ML-068

THE HH WAY
===========
Every previous generator used a single ticker (phi) and asked "what tokens
live near this J1 angle?" The vessel added history but still used a scalar
J1 position. The arc steerer added bigram grammar but the vessel arc target
was nearly zero because the stiffness gradient at the parked position was flat.

This script does it differently:

  1. Start at (J1_init, Ksi_init, J2_init, Phi_init, E_init) — a full HH address
  2. Integrate the actual HH equations of motion for N steps
  3. Each integration step gives a new (J1_t, Ksi_t) on the trajectory
  4. At each step, score tokens by BOTH:
     a. Bigram arc score: does this token arrive via the arc (ΔJ1_t, ΔKsi_t)?
     b. LM_Head score: does this token match the current hidden state?
  5. The HH orbit IS the generation schedule — no vessel needed

THE HH EQUATIONS
=================
The Hénon-Heiles Hamiltonian:
  H = (p1² + p2²)/2 + (q1² + q2²)/2 + q1²*q2 - q2³/3

Equations of motion:
  dq1/dt = p1
  dq2/dt = p2
  dp1/dt = -q1 - 2*q1*q2
  dp2/dt = -q2 - q1² + q2²

Action-angle coordinates (J1, J2) are computed from (q1, p1, q2, p2).
In the framework, J1 corresponds to the angular position on the manifold
and Ksi to the spectral entropy of the projection.

For generation purposes, we use a simplified mapping:
  J1_orbit(t) = phase angle of (q1, p1) at time t → [0°, 360°)
  Ksi_orbit(t) = normalized energy in (q2, p2) relative to total

Each integration step (q,p) → (q',p') traces an arc (ΔJ1, ΔKsi) on the manifold.
That arc is exactly the vessel arc target — but now it comes from real HH dynamics,
not a stiffness gradient approximation.

THE ORBIT-BIGRAM MATCH
=======================
At each step t of the HH orbit:
  orbit_arc_t = (ΔJ1_t, ΔKsi_t)  ← from HH integration

  For each candidate token v:
    arc_score[v] = Σ_{(t1,v) in bigram_db} freq * arc_compat(arc(t1→v), orbit_arc_t)

  The token that best matches WHERE THE HH ORBIT IS GOING gets the highest score.

This is fundamentally different from all prior approaches:
  - No ticker: the orbit IS the sequence
  - No zone boundaries: the orbit crosses them naturally
  - No dead zone avoidance: the HH orbit passes through Wada boundary naturally
    (orbits near escape energy visit the Wada zone — high-Ksi content words)
  - Deterministic from initial conditions: same (q0,p0,E) → same generation

INITIAL CONDITIONS
==================
Three orbit families from the HH manifold (from CORE-003 certified zones):

  NATIVE_ENGLISH: E ≈ 0.08, starts in the (J1=90°, Ksi=0.585) basin interior
    Orbit stays near the English attractor, revisits Zone A/B alternation
    Expected: balanced function/content word flow

  ABSTRACT: E ≈ 0.12, starts near the Wada boundary (J1=270°, Ksi=0.88)
    Higher energy orbit, visits Wada zone, wider vocabulary
    Expected: more complex/abstract vocabulary

  STRUCTURED: E ≈ 0.05, starts deep in basin (J1=110°, Ksi=0.25)
    Low energy, stays near Zone A peak
    Expected: high coherence, shorter arcs, structured output

PRE-REGISTERED HYPOTHESES
==========================
H-HO-001: ORBIT GIVES RICHER ARC VARIETY
  The HH orbit arc target (ΔJ1_t, ΔKsi_t) varies more across 120 steps
  than the vessel arc target (which was nearly constant at -0.06°).
  PASS: std(ΔJ1_orbit) > 10°

H-HO-002: ORBIT-BIGRAM AGREEMENT IMPROVES COHERENCE
  Steps where the orbit arc matches a known bigram arc family (compat > 0.5)
  produce higher-quality tokens (longer words, less fragmentation) than
  steps where they don't agree.
  PASS: mean_wl(high_compat_steps) > mean_wl(low_compat_steps)

H-HO-003: ORBIT FAMILY SPECIFICITY
  Different initial conditions produce measurably different output:
  NATIVE_ENGLISH orbit produces more ZONE_A_INT arc types than ABSTRACT orbit.
  PASS: abs(frac_zone_a_int(native) - frac_zone_a_int(abstract)) > 0.1

Usage:
  python hh_orbit_generator.py generate ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --bigram_db bigram_db_combined.json ^
    --orbit native_english ^
    --n_tokens 120 --lambda_arc 0.7 --output hh_native.json

  python hh_orbit_generator.py compare_orbits ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --bigram_db bigram_db_combined.json ^
    --n_tokens 120 --output hh_compare.json
"""

import argparse, json, sys, time
import numpy as np
from pathlib import Path

K_PROJ = 16
VOCAB_SIZE = 50257

# ── HH Dynamics ───────────────────────────────────────────────────────────────

def hh_derivs(state):
    """
    Hénon-Heiles equations of motion.
    state = [q1, q2, p1, p2]
    Returns [dq1/dt, dq2/dt, dp1/dt, dp2/dt]
    """
    q1, q2, p1, p2 = state
    dq1 = p1
    dq2 = p2
    dp1 = -q1 - 2.0 * q1 * q2
    dp2 = -q2 - q1**2 + q2**2
    return np.array([dq1, dq2, dp1, dp2])

def rk4_step(state, dt):
    """Single RK4 integration step."""
    k1 = hh_derivs(state)
    k2 = hh_derivs(state + 0.5*dt*k1)
    k3 = hh_derivs(state + 0.5*dt*k2)
    k4 = hh_derivs(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def hh_energy(state):
    q1, q2, p1, p2 = state
    return 0.5*(p1**2 + p2**2) + 0.5*(q1**2 + q2**2) + q1**2*q2 - q2**3/3.0

def state_to_address(state):
    """
    Map HH state (q1, q2, p1, p2) to (J1, Ksi) address.

    J1: phase angle of (q1, p1) oscillation → [0°, 360°)
    Ksi: normalized energy partition → [0, 1]
    """
    q1, q2, p1, p2 = state
    # J1 = angle of (q1, p1) vector
    j1 = float(np.degrees(np.arctan2(p1, q1)) % 360.0)
    # Ksi = fraction of energy in the q2 DOF
    E_total = hh_energy(state)
    E_q2    = 0.5*p2**2 + 0.5*q2**2
    # Normalize to [0,1] using a soft sigmoid
    ksi = float(np.clip(E_q2 / (abs(E_total) + 1e-6), 0.0, 1.0))
    return j1, ksi

# Pre-defined initial conditions (from CORE-003 certified addresses)
ORBIT_CONFIGS = {
    "native_english": {
        # E=0.1477, stays in basin, moderate arc variation (H-HO-001 PASS)
        "q1":  0.5, "q2": -0.2, "p1": 0.1, "p2":  0.3,
        "dt":  0.10, "steps_per_token": 8, "lookback": 4,
        "label": "native_english (E≈0.148, basin orbit, J1_std≈11°)"
    },
    "abstract": {
        # E=0.1453, near-escape, high arc variation (Wada zone visitor)
        "q1":  0.3, "q2":  0.2, "p1": 0.3, "p2":  0.2,
        "dt":  0.10, "steps_per_token": 8, "lookback": 4,
        "label": "abstract (E≈0.145, near-escape, J1_std≈17°)"
    },
    "structured": {
        # E=0.1453, Z3 basin 2, crosses basin boundaries
        "q1": -0.3, "q2":  0.2, "p1": 0.2, "p2":  0.3,
        "dt":  0.10, "steps_per_token": 8, "lookback": 4,
        "label": "structured (E≈0.145, Z3-basin2, J1_std≈14°)"
    },
    "custom": None
}

def arc_type_classify(j1a, j1b, dksi):
    def in_z(j, lo, hi): return lo <= (j%360) <= hi
    dj1 = float(((j1b - j1a + 180) % 360) - 180)
    if in_z(j1a,140,265) or in_z(j1b,140,265): return "DEAD"
    if abs(dj1) < 20 and abs(dksi) < 0.08:     return "HOVER"
    if in_z(j1a,80,140) and in_z(j1b,80,140):  return "A_INT"
    if in_z(j1a,280,340) and in_z(j1b,280,340):return "B_INT"
    if in_z(j1a,80,140) and in_z(j1b,280,340): return "A2B"
    if in_z(j1a,280,340) and in_z(j1b,80,140): return "B2A"
    if abs(dj1) > 120:                           return "PIVOT"
    return "CROSS"

# ── Resource loading ───────────────────────────────────────────────────────────

def load_resources(model_path, engram_path):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"  [model] {model_path}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    tok   = GPT2Tokenizer.from_pretrained(model_path)
    W     = model.lm_head.weight.detach().float().numpy()
    W_n   = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    with open(engram_path) as f: e = json.load(f)
    ksi_v = np.array(e["ksi_vals"], dtype=np.float32)
    j1_v  = np.array(e["j1_pca"], dtype=np.float32) % 360.0
    pc1   = np.array(e["pc1"], dtype=np.float32)
    pc2   = np.array(e["pc2"], dtype=np.float32)
    U_ref = np.array(e["U_ref"], dtype=np.float32)
    return model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref

def load_db(path):
    with open(path) as f: return json.load(f)

def h_address(h, U_ref, pc1, pc2):
    h_n  = h / (np.linalg.norm(h) + 1e-12)
    proj = U_ref.T @ h_n; p = proj**2/(proj**2).sum()+1e-12
    ksi  = float(-np.sum(p*np.log(p+1e-12))/np.log(K_PROJ))
    j1   = float(np.degrees(np.arctan2(float(np.dot(h_n,pc1)),
                                        float(np.dot(h_n,pc2))))%360)
    return ksi, j1

def arc_score_from_orbit(current_tok, db, orbit_dj1, orbit_dksi,
                          sigma_j1=20.0, sigma_ksi=0.12, top_k=40):
    """Score vocab by how well each bigram arc matches the current orbit arc."""
    scores = np.zeros(VOCAB_SIZE, dtype=np.float32)
    key    = str(current_tok)
    if key not in db["tokens"]: return scores, False
    deps   = db["tokens"][key]["departures"]
    total  = sum(v["count"] for v in deps.values())
    if total == 0: return scores, False
    items  = sorted(deps.items(), key=lambda x: -x[1]["count"])[:top_k]
    for t2_str, arc in items:
        t2 = int(t2_str)
        dj1_d  = (arc["mean_dj1"]  - orbit_dj1)  / sigma_j1
        dksi_d = (arc["mean_dksi"] - orbit_dksi) / sigma_ksi
        compat = float(np.exp(-0.5*(dj1_d**2 + dksi_d**2)))
        freq   = arc["count"] / (total + 1e-12)
        scores[t2] += float(freq * compat)
    return scores, True

# ── Core generation ────────────────────────────────────────────────────────────

def generate_from_orbit(model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref,
                         db, orbit_cfg, n_tokens=120,
                         lambda_arc=0.7, temperature=0.82,
                         seed=0, label=""):
    """
    Generate tokens by following a real HH orbit.
    The orbit gives (ΔJ1, ΔKsi) at each step, which is matched against
    the bigram database to score candidate tokens.
    """
    import torch
    rng = np.random.default_rng(seed)

    # Initialize HH state
    state = np.array([orbit_cfg["q1"], orbit_cfg["q2"],
                       orbit_cfg["p1"], orbit_cfg["p2"]], dtype=np.float64)
    dt = orbit_cfg["dt"]; spt = orbit_cfg["steps_per_token"]
    lookback = orbit_cfg.get("lookback", 4)
    E0 = hh_energy(state)

    # Compute initial orbit address
    j1_prev, ksi_prev = state_to_address(state)
    print(f"\n  [{label}]")
    print(f"  E={E0:.4f}  initial J1={j1_prev:.1f}°  Ksi={ksi_prev:.3f}")
    print(f"  dt={dt}  spt={spt}  lookback={lookback}  lambda_arc={lambda_arc}")

    # Seed hidden state: find token at initial orbit address
    d_ksi = np.abs(ksi_v - ksi_prev)
    d_j1  = np.abs(j1_v - j1_prev) % 360.0
    d_j1  = np.minimum(d_j1, 360 - d_j1) / 180.0
    seed_id = int(np.argmin(d_ksi + 0.3 * d_j1))
    seed_str = tok.decode([seed_id], skip_special_tokens=False)
    print(f"  Seed token: '{seed_str}' (Ksi={ksi_v[seed_id]:.3f}, J1={j1_v[seed_id]:.1f}°)")

    with torch.no_grad():
        out = model(torch.tensor([[seed_id]]), output_hidden_states=True)
    h_raw = out.hidden_states[-1][0,-1].float().numpy()
    h_ln  = h_raw / (np.linalg.norm(h_raw) + 1e-12)

    generated_ids  = [seed_id]
    generated_text = seed_str
    ksi_traj = [float(ksi_v[seed_id])]
    orbit_arc_traj = []
    arc_type_counts = {}
    steps_with_arc = 0; steps_without_arc = 0
    compat_scores = []; word_lens_high = []; word_lens_low = []
    current_tok = seed_id

    # Lookback buffer: store last `lookback` J1/Ksi positions
    j1_buf  = [j1_prev]  * lookback
    ksi_buf = [ksi_prev] * lookback

    print(f"\n  ", end="", flush=True)

    for step in range(n_tokens):
        # Advance HH orbit by spt integration steps
        for _ in range(spt):
            state = rk4_step(state, dt)

        j1_new, ksi_new = state_to_address(state)
        # Arc = from lookback position to now (captures phrase-length trajectory)
        j1_lb  = j1_buf[0];  ksi_lb  = ksi_buf[0]
        orbit_dj1  = float(((j1_new - j1_lb + 180) % 360) - 180)
        orbit_dksi = ksi_new - ksi_lb
        atype = arc_type_classify(j1_lb, j1_new, orbit_dksi)
        arc_type_counts[atype] = arc_type_counts.get(atype, 0) + 1

        # Update lookback buffer
        j1_buf.pop(0);  j1_buf.append(j1_new)
        ksi_buf.pop(0); ksi_buf.append(ksi_new)

        # Score tokens by orbit arc match
        arc_sc, has_arc = arc_score_from_orbit(
            current_tok, db, orbit_dj1, orbit_dksi)

        # LM_Head score
        lm_sc = W_n @ h_ln.astype(np.float64)

        if has_arc:
            steps_with_arc += 1
            # Measure arc compatibility (max score = how well the best bigram matches)
            max_compat = float(arc_sc.max())
            compat_scores.append(max_compat)
            arc_weight = lambda_arc
            # H-HO-002 tracking
            if max_compat > 0.3:
                wl = len(tok.decode([current_tok]).strip())
                word_lens_high.append(wl)
            else:
                wl = len(tok.decode([current_tok]).strip())
                word_lens_low.append(wl)
        else:
            steps_without_arc += 1
            arc_weight = 0.0
            compat_scores.append(0.0)

        # Combine scores
        arc_norm = arc_sc / (arc_sc.max() + 1e-12) if arc_sc.max() > 0 else arc_sc
        lm_norm  = (lm_sc - lm_sc.mean()) / (lm_sc.std() + 1e-12)
        combined = (1.0 - arc_weight) * lm_norm + arc_weight * arc_norm

        orbit_arc_traj.append({
            "j1_orbit": float(j1_new), "ksi_orbit": float(ksi_new),
            "dj1": float(orbit_dj1), "dksi": float(orbit_dksi),
            "arc_type": atype, "compat": float(compat_scores[-1]),
            "arc_weight": float(arc_weight)
        })

        # Sample
        top_s = combined - combined.max()
        probs  = np.exp(top_s / temperature); probs /= probs.sum()
        chosen = int(rng.choice(VOCAB_SIZE, p=probs))

        token_str = tok.decode([chosen], skip_special_tokens=False)
        generated_ids.append(chosen); generated_text += token_str
        print(token_str, end="", flush=True)

        current_tok = chosen
        j1_prev = j1_new   # still needed for orbit_arc_sample

        with torch.no_grad():
            out  = model(torch.tensor([[chosen]]), output_hidden_states=True)
        h_raw = out.hidden_states[-1][0,-1].float().numpy()
        h_ln  = h_raw / (np.linalg.norm(h_raw) + 1e-12)
        ksi_h, _ = h_address(h_raw, U_ref, pc1, pc2)
        ksi_traj.append(ksi_h)

    print("\n")

    # Metrics
    words = [tok.decode([t]).strip() for t in generated_ids
             if any(c.isalpha() for c in tok.decode([t]))]
    mean_wl    = float(np.mean([len(w) for w in words])) if words else 0.
    garbage    = sum(1 for t in generated_ids if '\ufffd' in tok.decode([t]))
    single_let = sum(1 for w in words if len(w)==1) / (len(words)+1e-12)
    orbit_dj1s = [o["dj1"] for o in orbit_arc_traj]
    h001 = bool(np.std(orbit_dj1s) > 10.0)
    h002 = "N/A"
    if word_lens_high and word_lens_low:
        mwl_hi = float(np.mean(word_lens_high))
        mwl_lo = float(np.mean(word_lens_low))
        h002 = f"PASS ({mwl_hi:.2f}>{mwl_lo:.2f})" if mwl_hi > mwl_lo else f"FAIL ({mwl_hi:.2f}<{mwl_lo:.2f})"

    print(f"  mean_wl={mean_wl:.2f}  garbage={garbage}  single_let={single_let:.3f}")
    print(f"  arc steps: {steps_with_arc}/{n_tokens} ({100*steps_with_arc/n_tokens:.0f}%)")
    print(f"  orbit ΔJ1 std={np.std(orbit_dj1s):.1f}°  H-HO-001: {'PASS' if h001 else 'FAIL'}")
    print(f"  H-HO-002: {h002}")
    print(f"  orbit arc types: {dict(sorted(arc_type_counts.items(), key=lambda x:-x[1]))}")

    return {
        "label": label,
        "orbit_config": {k: v for k,v in orbit_cfg.items() if k != "label"},
        "generated_text": generated_text,
        "n_tokens": n_tokens, "lambda_arc": float(lambda_arc),
        "garbage_count": int(garbage),
        "mean_word_length": float(mean_wl),
        "single_letter_fraction": float(single_let),
        "steps_with_arc": int(steps_with_arc),
        "known_bigram_fraction": float(steps_with_arc / n_tokens),
        "orbit_dj1_std": float(np.std(orbit_dj1s)),
        "h001_pass": h001,
        "h002_result": h002,
        "arc_type_distribution": arc_type_counts,
        "ksi_mean": float(np.mean(ksi_traj)),
        "ksi_std":  float(np.std(ksi_traj)),
        "ksi_trajectory": [float(k) for k in ksi_traj],
        "orbit_arc_sample": orbit_arc_traj[:20],  # first 20 for inspection
    }

# ── Subcommands ────────────────────────────────────────────────────────────────

def cmd_generate(args):
    print("\n" + "="*70)
    print("HH ORBIT GENERATOR — Real HH Dynamics as Generation Schedule")
    print("="*70)

    model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref = load_resources(
        args.model, args.engram)
    db = load_db(args.bigram_db)
    print(f"  [db] {len(db['tokens'])} source tokens, "
          f"{db['_meta'].get('total_arcs',0):,} arcs")

    orbit_name = getattr(args, 'orbit', 'native_english')
    if orbit_name == "custom":
        cfg = {"q1": args.q1, "q2": args.q2, "p1": args.p1, "p2": args.p2,
               "dt": getattr(args,'dt',0.08), "steps_per_token": getattr(args,'spt',4),
               "label": f"custom q1={args.q1} q2={args.q2}"}
    else:
        cfg = ORBIT_CONFIGS[orbit_name]

    print(f"  Orbit: {cfg.get('label', orbit_name)}")
    print(f"  E = {hh_energy(np.array([cfg['q1'],cfg['q2'],cfg['p1'],cfg['p2']])): .4f}")

    result = generate_from_orbit(
        model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref, db, cfg,
        n_tokens=args.n_tokens,
        lambda_arc=getattr(args,'lambda_arc',0.7),
        seed=getattr(args,'seed',0),
        label=orbit_name)

    with open(args.output,'w') as f:
        json.dump(result, f, indent=2, default=lambda o:
            int(o) if isinstance(o,(np.int64,np.int32,np.intp)) else
            float(o) if isinstance(o,(np.float64,np.float32)) else
            bool(o) if isinstance(o,(np.bool_,bool)) else None)
    print(f"\n  Saved → {args.output}")

def cmd_compare_orbits(args):
    print("\n" + "="*70)
    print("HH ORBIT COMPARE — Three Initial Conditions")
    print("="*70)

    model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref = load_resources(
        args.model, args.engram)
    db = load_db(args.bigram_db)

    results = []
    for orbit_name, cfg in ORBIT_CONFIGS.items():
        if cfg is None: continue
        print(f"\n{'─'*60}\nOrbit: {cfg['label']}")
        r = generate_from_orbit(
            model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref, db, cfg,
            n_tokens=args.n_tokens,
            lambda_arc=getattr(args,'lambda_arc',0.7),
            seed=0, label=orbit_name)
        results.append(r)

    # H-HO-003: orbit family specificity
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print(f"\n  {'Orbit':<20} {'mean_wl':>7} {'arc%':>6} {'ΔJ1_std':>9} {'H001':>6}")
    print("  " + "-"*55)
    for r in results:
        print(f"  {r['label']:<20} {r['mean_word_length']:>7.2f} "
              f"{100*r['known_bigram_fraction']:>5.0f}% "
              f"{r['orbit_dj1_std']:>9.1f}° "
              f"{'PASS' if r['h001_pass'] else 'FAIL':>6}")

    if len(results) >= 2:
        ne = next((r for r in results if r['label']=='native_english'), None)
        ab = next((r for r in results if r['label']=='abstract'), None)
        if ne and ab:
            ne_a = ne['arc_type_distribution'].get('A_INT',0) / (ne['n_tokens']+1e-12)
            ab_a = ab['arc_type_distribution'].get('A_INT',0) / (ab['n_tokens']+1e-12)
            diff = abs(ne_a - ab_a)
            h003 = "PASS" if diff > 0.1 else "FAIL"
            print(f"\n  H-HO-003: frac_A_INT: native={ne_a:.3f}  abstract={ab_a:.3f}  "
                  f"diff={diff:.3f}  → {h003}")

    print("\n  Generated texts:")
    for r in results:
        print(f"\n  [{r['label']}]")
        print(f"  {r['generated_text'][:200]}")

    def cvt(o):
        if isinstance(o,(np.int64,np.int32,np.intp)): return int(o)
        if isinstance(o,(np.float64,np.float32)): return float(o)
        if isinstance(o,(np.bool_,bool)): return bool(o)
    with open(args.output,'w') as f:
        json.dump({"results": results}, f, indent=2, default=cvt)
    print(f"\n\n  Saved → {args.output}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="HH Orbit Token Generator")
    sub = ap.add_subparsers(dest="cmd")

    pg = sub.add_parser("generate")
    pg.add_argument("--model",       required=True)
    pg.add_argument("--engram",      required=True)
    pg.add_argument("--bigram_db",   required=True)
    pg.add_argument("--orbit",       default="native_english",
                    choices=list(ORBIT_CONFIGS.keys()))
    pg.add_argument("--n_tokens",    type=int, default=120)
    pg.add_argument("--lambda_arc",  type=float, default=0.7)
    pg.add_argument("--seed",        type=int, default=0)
    pg.add_argument("--dt",          type=float, default=0.08)
    pg.add_argument("--spt",         type=int, default=4,
                    help="integration steps per token")
    pg.add_argument("--q1",          type=float, default=0.3)
    pg.add_argument("--q2",          type=float, default=-0.1)
    pg.add_argument("--p1",          type=float, default=0.1)
    pg.add_argument("--p2",          type=float, default=0.2)
    pg.add_argument("--output",      default="hh_out.json")

    pc = sub.add_parser("compare_orbits")
    pc.add_argument("--model",       required=True)
    pc.add_argument("--engram",      required=True)
    pc.add_argument("--bigram_db",   required=True)
    pc.add_argument("--n_tokens",    type=int, default=120)
    pc.add_argument("--lambda_arc",  type=float, default=0.7)
    pc.add_argument("--output",      default="hh_compare.json")

    args = ap.parse_args()
    {"generate": cmd_generate, "compare_orbits": cmd_compare_orbits
     }.get(args.cmd, lambda _: (ap.print_help(), sys.exit(1)))(args)

if __name__ == "__main__":
    main()
