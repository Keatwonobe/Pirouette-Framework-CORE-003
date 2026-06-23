"""
wandering_model_arc.py
Wandering Model + Bigram Arc Augmentation
Pirouette Framework Volume 8 · CORE-003 · ML-069

WHAT THIS DOES
==============
The wandering model (wandering_model.py / wandering_sweep.py) produces
coherent paragraph-length English by correcting each layer's hidden state
Ksi toward a scheduled target. It works well. The text it produces is
grammatical, on-topic, and structured.

The bigram arc database (bigram_db_combined.json) encodes which token
transitions are grammatically natural on the manifold. It gives us
statistics like "after the token ',', the next token typically arrives
via a +11° arc in Zone A with ΔKsi=+0.065."

This script augments the wandering model with the bigram arc:

  In each layer hook, BEFORE Ksi correction:
    1. Look up current-token successors in bigram database
    2. Find the most probable next arc at this layer
    3. Nudge h toward the J1 direction predicted by that arc
    4. THEN apply Ksi correction (unchanged from wandering_model.py)

  Result: Ksi correction shapes the epistemic register (wandering model)
          bigram arc shapes the syntactic direction (arc grammar)
          Together: coherent paragraphs with better grammatical structure

THE NUDGE MECHANISM
====================
The hidden state h lives in R^1280. Its J1 coordinate is the angle in
the (pc1, pc2) plane. To nudge h toward a target J1 direction:

  1. Project h onto (pc1, pc2): a = h·pc1, b = h·pc2
  2. Current J1 = atan2(a, b)
  3. Target J1 = current J1 + predicted ΔJ1 from bigram arc
  4. Nudge: rotate (a,b) toward target J1 by fraction α
     a_new = cos(target)*|(a,b)| + (1-α)*a
     Normalize, add back residual
  5. Scale α by layer position: more at early layers (structure),
     less at late layers (don't disrupt content selection)

PRE-REGISTERED HYPOTHESES
==========================
H-WA-001: ARC AUGMENTATION PRESERVES COHERENCE
  The augmented model produces text with mean_ksi close to (within 0.02
  of) the un-augmented wandering model for the same schedule.
  PASS: |mean_ksi_aug - mean_ksi_base| < 0.02

H-WA-002: ARC AUGMENTATION ADDS GRAMMATICAL MARKERS
  The augmented model produces more comma/period/newline tokens per 100
  tokens than the un-augmented wandering model.
  PASS: punct_rate_aug > punct_rate_base * 1.2

H-WA-003: ARC TYPE DISTRIBUTION MATCHES CORPUS STATISTICS
  The arc types in generated text (computed from hidden-state J1 trajectory)
  should show higher ZONE_A_INT fraction than the HH orbit generator, which
  was dominated by DEAD arcs (64%) due to high orbit energy.
  PASS: ZONE_A_INT fraction > 0.15 AND DEAD fraction < 0.40

Usage:
  # Basic run — augment the wandering model with arc grammar:
  python wandering_model_arc.py ^
    --model "...\\models\\gpt2-large-cycle3-cust-arc1" ^
    --baseline_file baseline_math.json ^
    --bigram_db bigram_db_combined.json ^
    --engram engram_curve.json ^
    --prompt "The cause of altruistic behavior is" ^
    --delta_end 0.08 --arc_strength 0.15 ^
    --output wma_altruism.json

  # Sweep delta_end and arc_strength together:
  python wandering_model_arc.py ^
    --model "...\\models\\gpt2-large-cycle3-cust-arc1" ^
    --baseline_file baseline_math.json ^
    --bigram_db bigram_db_combined.json ^
    --engram engram_curve.json ^
    --prompt "The relationship between mathematical constants and information" ^
    --sweep --output wma_sweep.json

  # Comparison: base wandering vs augmented:
  python wandering_model_arc.py ^
    --model "...\\models\\gpt2-large-cycle3-cust-arc1" ^
    --baseline_file baseline_math.json ^
    --bigram_db bigram_db_combined.json ^
    --engram engram_curve.json ^
    --prompt "The cause of altruistic behavior is" ^
    --delta_end 0.08 --compare --output wma_compare.json
"""

import argparse, json, time, sys
import numpy as np
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

K_PROJ = 16

# ── Ksi arithmetic (from wandering_model.py — unchanged) ──────────────────────

def measure_ksi(h, V_top):
    z = V_top.T @ h; z2 = z*z; S = float(z2.sum())
    if S < 1e-12: return 0.5, z
    p = z2/S; ps = np.where(p>1e-15, p, 1e-15)
    return float(np.clip(-np.sum(p*np.log(ps))/np.log(K_PROJ), 0, 1)), z

def ksi_gradient(z):
    z2=z*z; S=float(z2.sum())
    if S < 1e-12: return np.zeros_like(z)
    p=z2/S; ps=np.where(p>1e-15,p,1e-15); H=float(-np.sum(p*np.log(ps)))
    return -(2.*z)/(S*np.log(K_PROJ))*(np.log(ps)+H)

def correct_ksi(h, V_top, tgt, cur, step_size=0.05, n_steps=3):
    d = tgt - cur
    if abs(d) < 5e-5: return h, cur
    z = V_top.T@h; hp = V_top@z; hr = h-hp; pn = float(np.linalg.norm(hp))
    sign = float(np.sign(d))
    for _ in range(n_steps):
        g = ksi_gradient(z); gn = float(np.linalg.norm(g))
        if gn < 1e-12: break
        z = z + sign*step_size*g/gn
    np_ = V_top@z; nn = float(np.linalg.norm(np_))
    if nn > 1e-8 and pn > 1e-8: z = z*(pn/nn)
    ho = (V_top@z+hr).astype(np.float32)
    ksi_out, _ = measure_ksi(ho, V_top)
    return ho, ksi_out

def build_relative_bezier(baseline, delta_start, delta_end):
    n = len(baseline); t = np.linspace(0,1,n)
    dp = (delta_start + delta_end) / 2.
    deltas = (1-t)**2*delta_start + 2*(1-t)*t*dp + t**2*delta_end
    return (np.array(baseline) + deltas).tolist()

def extract_gate_svds(model, n_layers):
    svds = []
    for i in range(n_layers):
        w = model.transformer.h[i].mlp.c_fc.weight.data.float().numpy()
        U, _, _ = np.linalg.svd(w, full_matrices=False)
        svds.append(U[:, :K_PROJ].astype(np.float32))
    return svds

# ── Arc augmentation utilities ─────────────────────────────────────────────────

def load_bigram_db(path):
    with open(path) as f: return json.load(f)

def load_engram(path):
    with open(path) as f: e = json.load(f)
    return {
        "pc1":  np.array(e["pc1"], dtype=np.float32),
        "pc2":  np.array(e["pc2"], dtype=np.float32),
    }

def get_predicted_arc(current_tok, db, top_k=10):
    """
    Return the predicted (ΔJ1, ΔKsi) for the next token given the current.
    Uses the frequency-weighted mean arc from the top-k successors.
    Returns (0, 0) if the token is not in the database.
    """
    key = str(current_tok)
    if key not in db["tokens"]: return 0.0, 0.0
    deps = db["tokens"][key]["departures"]
    if not deps: return 0.0, 0.0
    items = sorted(deps.items(), key=lambda x: -x[1]["count"])[:top_k]
    total = sum(v["count"] for _, v in items)
    dj1  = sum(v["mean_dj1"]  * v["count"] for _, v in items) / (total + 1e-12)
    dksi = sum(v["mean_dksi"] * v["count"] for _, v in items) / (total + 1e-12)
    return float(dj1), float(dksi)

def nudge_h_toward_arc(h, pc1, pc2, target_dj1, alpha=0.15):
    """
    Nudge hidden state h's J1 coordinate by target_dj1 degrees.
    alpha: how strongly to rotate (0=no nudge, 1=full rotation).
    Returns: modified h (float32), current_j1, target_j1
    """
    if abs(target_dj1) < 0.5 or alpha < 1e-4:
        j1 = float(np.degrees(np.arctan2(np.dot(h,pc1), np.dot(h,pc2))) % 360)
        return h, j1, j1

    h_n = h / (np.linalg.norm(h) + 1e-12)
    a   = float(np.dot(h_n, pc1))
    b   = float(np.dot(h_n, pc2))
    j1  = float(np.degrees(np.arctan2(a, b)) % 360)

    # Target J1
    j1_target = (j1 + target_dj1) % 360
    rad_t = np.radians(j1_target)
    a_t   = float(np.cos(rad_t))
    b_t   = float(np.sin(rad_t))

    # Interpolate direction in (pc1, pc2) plane
    a_new = (1 - alpha) * a + alpha * a_t
    b_new = (1 - alpha) * b + alpha * b_t
    norm_ab = np.sqrt(a_new**2 + b_new**2)
    if norm_ab < 1e-12: return h, j1, j1_target

    # Original plane norm
    orig_norm = np.sqrt(a**2 + b**2)

    # Reconstruct: remove old plane component, add new
    h_plane    = a*pc1 + b*pc2
    h_residual = h - h_plane   # everything NOT in (pc1, pc2) plane

    h_plane_new = (a_new/norm_ab*orig_norm) * pc1 + (b_new/norm_ab*orig_norm) * pc2
    h_new       = (h_plane_new + h_residual).astype(np.float32)
    return h_new, j1, j1_target

# ── Core generation ────────────────────────────────────────────────────────────

def run_generation_arc(
    model, tok, prompt, schedule, svds, engram, db,
    apply_correction=True, apply_arc=True,
    arc_strength=0.15, arc_layer_decay=True,
    step_size=0.05, correction_steps=3,
    max_tokens=120, temperature=0.9, top_k=50,
):
    """
    Wandering model generation, optionally augmented with bigram arc nudge.

    Each layer hook:
      1. (if apply_arc) Nudge h toward predicted next arc
      2. (if apply_correction) Apply Ksi correction
    """
    n_layers = len(schedule)
    token_counter = [0]
    current_tok   = [None]
    ksi_pre_all = []; ksi_post_all = []
    j1_traj     = []
    arc_usage   = []   # (predicted_dj1, actual_j1_before, actual_j1_after)

    # pc1, pc2 for arc nudge
    pc1 = engram["pc1"].astype(np.float64)
    pc2 = engram["pc2"].astype(np.float64)

    def make_hook(li):
        def hook_fn(module, args, output):
            h_batch = output[0]
            h = h_batch[0,-1,:].float().cpu().numpy().astype(np.float64)
            V_top = svds[li]
            tgt = schedule[li]

            ksi_pre, _ = measure_ksi(h.astype(np.float32), V_top)
            ksi_pre_all.append(ksi_pre)

            # Layer-dependent arc strength (stronger at early layers)
            if arc_layer_decay:
                layer_alpha = arc_strength * (1.0 - li / (n_layers * 1.5))
                layer_alpha = max(0.0, layer_alpha)
            else:
                layer_alpha = arc_strength

            # Arc nudge (at first few layers only, to shape direction early)
            predicted_dj1 = 0.0; predicted_dksi = 0.0
            j1_before = 0.0; j1_after = 0.0
            if apply_arc and current_tok[0] is not None and layer_alpha > 0.01:
                predicted_dj1, predicted_dksi = get_predicted_arc(
                    current_tok[0], db)
                if abs(predicted_dj1) > 0.5:
                    h_nudged, j1_before, j1_after = nudge_h_toward_arc(
                        h.astype(np.float32), pc1.astype(np.float32),
                        pc2.astype(np.float32), predicted_dj1, layer_alpha)
                    h = h_nudged.astype(np.float64)

            # Ksi correction
            if apply_correction:
                h_new, ksi_post = correct_ksi(
                    h.astype(np.float32), V_top, tgt, ksi_pre,
                    step_size, correction_steps)
                ksi_post_all.append(ksi_post)
                h_out = torch.tensor(h_new, dtype=h_batch.dtype,
                                     device=h_batch.device)
            else:
                ksi_post_all.append(ksi_pre)
                h_out = torch.tensor(h.astype(np.float32),
                                     dtype=h_batch.dtype, device=h_batch.device)

            if li == 0:
                j1_traj.append(j1_after if apply_arc else j1_before)
            if li == 0 and apply_arc:
                arc_usage.append((float(predicted_dj1), float(j1_before),
                                   float(j1_after)))

            h_batch = h_batch.clone(); h_batch[0,-1,:] = h_out
            return (h_batch,) + output[1:] if isinstance(output, tuple) else h_batch
        return hook_fn

    hooks = [model.transformer.h[i].register_forward_hook(make_hook(i))
             for i in range(n_layers)]

    input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    generated = input_ids.clone()
    n_tokens  = 0

    # Initialize current_tok from last prompt token
    current_tok[0] = int(input_ids[0,-1])

    with torch.no_grad():
        for step in range(max_tokens):
            token_counter[0] = step
            out = model(generated)
            logits = out.logits[0,-1,:] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).unsqueeze(0)
            generated = torch.cat([generated, next_tok], dim=1)
            current_tok[0] = int(next_tok[0,0])
            n_tokens += 1
            if current_tok[0] == tok.eos_token_id: break

    for h in hooks: h.remove()

    text = tok.decode(generated[0], skip_special_tokens=True)
    mean_ksi_pre  = float(np.mean(ksi_pre_all))  if ksi_pre_all  else 0.
    mean_ksi_post = float(np.mean(ksi_post_all)) if ksi_post_all else 0.
    return text, mean_ksi_pre, mean_ksi_post, n_tokens, j1_traj, arc_usage

# ── Metrics ────────────────────────────────────────────────────────────────────

def punct_rate(text):
    punct = sum(1 for c in text if c in '.,;:!?()[]{}"\'-–—\n')
    return punct / (len(text) + 1e-12) * 100

def arc_type_from_j1_traj(j1_traj):
    types = {}
    for i in range(1, len(j1_traj)):
        dj1 = float(((j1_traj[i] - j1_traj[i-1] + 180) % 360) - 180)
        def in_z(j, lo, hi): return lo <= (j%360) <= hi
        j1a, j1b = j1_traj[i-1], j1_traj[i]
        if in_z(j1a,140,265) or in_z(j1b,140,265): t = "DEAD"
        elif abs(dj1) < 20: t = "HOVER"
        elif in_z(j1a,80,140) and in_z(j1b,80,140): t = "A_INT"
        elif in_z(j1a,280,340) and in_z(j1b,280,340): t = "B_INT"
        elif abs(dj1) > 120: t = "PIVOT"
        else: t = "CROSS"
        types[t] = types.get(t, 0) + 1
    return types

# ── Subcommands ────────────────────────────────────────────────────────────────

def run_one(model, tok, svds, engram, db, baseline, prompt,
            delta_start, delta_end, arc_strength, apply_arc,
            step_size, max_tokens, temperature, top_k, label=""):
    schedule = build_relative_bezier(baseline, delta_start, delta_end)
    text, kp, kpost, n, j1t, arcu = run_generation_arc(
        model, tok, prompt, schedule, svds, engram, db,
        apply_correction=True, apply_arc=apply_arc,
        arc_strength=arc_strength, arc_layer_decay=True,
        step_size=step_size, correction_steps=3,
        max_tokens=max_tokens, temperature=temperature, top_k=top_k)
    arc_dist = arc_type_from_j1_traj(j1t) if j1t else {}
    pr = punct_rate(text)
    total = sum(arc_dist.values()) + 1e-12
    return {
        "label": label, "apply_arc": apply_arc,
        "delta_end": float(delta_end), "arc_strength": float(arc_strength),
        "mean_ksi_pre": float(kp), "mean_ksi_post": float(kpost),
        "n_tokens": int(n), "punct_rate": float(pr),
        "arc_type_distribution": arc_dist,
        "frac_zone_a_int": float(arc_dist.get("A_INT",0)/total),
        "frac_dead": float(arc_dist.get("DEAD",0)/total),
        "generated_text": text,
        "arc_usage_sample": arcu[:10],
    }

def cmd_run(args):
    print("\n" + "="*70)
    print("WANDERING MODEL + ARC AUGMENTATION")
    print("="*70)

    print(f"\nLoading model: {args.model}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model, local_files_only=True,
                                             low_cpu_mem_usage=True)
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    n_layers = model.config.n_layer
    print(f"  Extracting SVDs...", flush=True)
    svds = extract_gate_svds(model, n_layers)

    with open(args.baseline_file) as f: bl = json.load(f)
    baseline = bl["baseline"]
    engram = load_engram(args.engram)
    db     = load_bigram_db(args.bigram_db)
    print(f"  [db] {len(db['tokens'])} tokens, "
          f"{db['_meta'].get('total_arcs',0):,} arcs")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  delta_end={args.delta_end}  arc_strength={args.arc_strength}")

    compare = getattr(args, 'compare', False)
    results = []

    if compare:
        print(f"\n--- BASELINE (no arc augmentation) ---")
        r_base = run_one(model, tok, svds, engram, db, baseline, args.prompt,
                          -0.02, args.delta_end, arc_strength=0.0,
                          apply_arc=False,
                          step_size=0.05, max_tokens=args.max_tokens,
                          temperature=args.temperature, top_k=50,
                          label="wandering_base")
        print(f"  Ksi_post={r_base['mean_ksi_post']:.4f}  "
              f"punct={r_base['punct_rate']:.2f}%  "
              f"arc_A_INT={r_base['frac_zone_a_int']:.3f}")
        print(f"  {r_base['generated_text'][:300]}")
        results.append(r_base)

    print(f"\n--- ARC AUGMENTED (arc_strength={args.arc_strength}) ---")
    r_aug = run_one(model, tok, svds, engram, db, baseline, args.prompt,
                     -0.02, args.delta_end, arc_strength=args.arc_strength,
                     apply_arc=True,
                     step_size=0.05, max_tokens=args.max_tokens,
                     temperature=args.temperature, top_k=50,
                     label="wandering_arc")
    print(f"  Ksi_post={r_aug['mean_ksi_post']:.4f}  "
          f"punct={r_aug['punct_rate']:.2f}%  "
          f"arc_A_INT={r_aug['frac_zone_a_int']:.3f}")
    print(f"  H-WA-003: A_INT={r_aug['frac_zone_a_int']:.3f} (>0.15?)  "
          f"DEAD={r_aug['frac_dead']:.3f} (<0.40?)")
    print(f"\n  Generated text:")
    print(f"  {r_aug['generated_text'][:600]}")
    results.append(r_aug)

    if compare and len(results) == 2:
        b, a = results[0], results[1]
        h001 = abs(a['mean_ksi_post'] - b['mean_ksi_post']) < 0.02
        h002 = a['punct_rate'] > b['punct_rate'] * 1.2
        h003 = a['frac_zone_a_int'] > 0.15 and a['frac_dead'] < 0.40
        print(f"\n  H-WA-001 (Ksi preserved): {'PASS' if h001 else 'FAIL'}  "
              f"Δ={abs(a['mean_ksi_post']-b['mean_ksi_post']):.4f}")
        print(f"  H-WA-002 (more punct):    {'PASS' if h002 else 'FAIL'}  "
              f"base={b['punct_rate']:.2f}%  aug={a['punct_rate']:.2f}%")
        print(f"  H-WA-003 (arc dist):      {'PASS' if h003 else 'FAIL'}")

    out = {"results": results, "prompt": args.prompt,
           "delta_end": args.delta_end, "arc_strength": args.arc_strength}
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2, default=lambda o:
            int(o) if isinstance(o,(int,np.int64,np.int32)) else
            float(o) if isinstance(o,(float,np.float64,np.float32)) else
            bool(o) if isinstance(o,(bool,np.bool_)) else None)
    print(f"\n  Saved → {args.output}")

def cmd_sweep(args):
    """Sweep both delta_end and arc_strength."""
    print("\n" + "="*70)
    print("WANDERING ARC SWEEP")
    print("="*70)

    print(f"Loading model...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model, local_files_only=True,
                                             low_cpu_mem_usage=True)
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    svds = extract_gate_svds(model, model.config.n_layer)

    with open(args.baseline_file) as f: bl = json.load(f)
    baseline = bl["baseline"]
    engram   = load_engram(args.engram)
    db       = load_bigram_db(args.bigram_db)

    delta_ends    = [-0.04, 0.0, 0.04, 0.08]
    arc_strengths = [0.0, 0.10, 0.20]

    results = []
    for delta_end in delta_ends:
        for arc_str in arc_strengths:
            lbl = f"d{delta_end:+.2f}_a{arc_str:.2f}"
            print(f"\n  [{lbl}]", flush=True)
            r = run_one(model, tok, svds, engram, db, baseline, args.prompt,
                         -0.02, delta_end, arc_strength=arc_str,
                         apply_arc=(arc_str > 0),
                         step_size=0.05, max_tokens=args.max_tokens,
                         temperature=args.temperature, top_k=50, label=lbl)
            results.append(r)
            print(f"    Ksi={r['mean_ksi_post']:.4f}  "
                  f"punct={r['punct_rate']:.2f}%  "
                  f"A_INT={r['frac_zone_a_int']:.3f}")
            print(f"    {r['generated_text'][:150]}")

    print("\n" + "="*70 + "\nSWEEP SUMMARY")
    print(f"  {'Label':<20} {'Ksi':>7} {'punct%':>8} {'A_INT':>7}  Text snippet")
    print("  " + "-"*70)
    for r in results:
        print(f"  {r['label']:<20} {r['mean_ksi_post']:>7.4f} "
              f"{r['punct_rate']:>7.2f}% {r['frac_zone_a_int']:>7.3f}  "
              f"{r['generated_text'][len(args.prompt):len(args.prompt)+60].replace(chr(10),'↵')}")

    with open(args.output, 'w') as f:
        json.dump({"results": results, "prompt": args.prompt}, f, indent=2,
                   default=lambda o:
                       int(o) if isinstance(o,(int,np.int64,np.int32)) else
                       float(o) if isinstance(o,(float,np.float64,np.float32)) else
                       bool(o) if isinstance(o,(bool,np.bool_)) else None)
    print(f"\n  Saved → {args.output}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Wandering Model + Arc Augmentation")
    ap.add_argument("--model",         required=True)
    ap.add_argument("--baseline_file", required=True)
    ap.add_argument("--bigram_db",     required=True)
    ap.add_argument("--engram",        required=True)
    ap.add_argument("--prompt",        default="The cause of altruistic behavior is")
    ap.add_argument("--delta_end",     type=float, default=0.08)
    ap.add_argument("--arc_strength",  type=float, default=0.15)
    ap.add_argument("--max_tokens",    type=int,   default=120)
    ap.add_argument("--temperature",   type=float, default=0.9)
    ap.add_argument("--compare",       action="store_true")
    ap.add_argument("--sweep",         action="store_true")
    ap.add_argument("--output",        default="wma_out.json")

    args = ap.parse_args()
    if args.sweep:
        cmd_sweep(args)
    else:
        cmd_run(args)

if __name__ == "__main__":
    main()
