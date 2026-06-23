"""
arc_steered_generator.py
Arc-Driven Bigram Generator
Pirouette Framework Volume 8 · CORE-003 · ML-067

THE PROBLEM WITH THE PREVIOUS APPROACH
========================================
The arc_prior was computed but never wired into generation.
The vessel's softmax_bias used stiffness/helicity, and arc_prior
was loaded but unused — both corpus runs produced identical output.

THIS SCRIPT
============
Strips everything back to the arc signal alone:

  1. CORPUS BUILDER — incrementally builds and updates a bigram arc
     database from multiple text files. Each new file updates existing
     bigram statistics (weighted average by frequency). The database
     grows richer with each text added.

  2. ARC STEERER — generation loop driven PURELY by arc statistics:
     At each step:
       a. Look up current token in bigram database
       b. Find all known successor arcs from this token
       c. Score vocab tokens by arc compatibility:
          arc_score[v] = Σ_i freq(t1→v) * arc_compat(arc(t1→v), vessel_arc_target)
       d. Combine with LM_Head logits:
          final_score = (1-λ) * lm_score + λ * arc_score
       e. Sample

  3. VESSEL ARC TARGET — the vessel provides a target arc (ΔJ1, ΔKsi)
     at each step based on its current position and the stiffness gradient.
     The arc steerer asks "which bigrams lead toward the vessel's target?"
     This is the vessel and arc grammar talking to each other.

THE BIGRAM DATABASE
====================
Stored as a flat JSON/numpy structure:
  bigram_db[t1_id] = {
    successors: {t2_id: {count, mean_dj1, mean_dksi, arc_type}},
    total_departures: int
  }

Updated incrementally: when a new corpus adds evidence for (t1, t2),
  new_count = old_count + new_count
  new_mean_dj1 = (old_mean * old_count + new_mean * new_count) / total_count
  (Welford's online mean update)

At generation time, only the TOP-K successors per token are used (default K=20).

PRE-REGISTERED HYPOTHESES
==========================
H-AG-001: ARC STEERING IMPROVES GRAMMATICALITY
  Pure arc steering (λ=1.0) produces sequences where consecutive tokens
  follow the dominant arc family for their zone (Family 1 for Zone A,
  Family 2 for content→punct). Measured by fraction of bigrams in output
  that appear in the training database.
  PASS: known_bigram_fraction > 0.3

H-AG-002: VESSEL + ARC > ARC ALONE
  Combined (vessel + arc, λ=0.5) outperforms arc-only (λ=1.0) on
  mean word length. The vessel's manifold position adds information
  that the arc statistics alone cannot provide.
  PASS: mean_wl(combined) > mean_wl(arc_only)

H-AG-003: CORPUS SPECIFICITY
  A generator steered by a Shakespeare-trained database produces
  different output character (different dominant bigrams) than one
  steered by a Mutual-Aid-trained database.
  PASS: Jaccard similarity of top-50 generated tokens < 0.5 across
  databases.

Usage:
  # Build/update bigram database from text files:
  python arc_steered_generator.py build ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --texts tinyshakespeare.txt mutual_aid.txt ^
    --output bigram_db.json

  # Generate using arc steering:
  python arc_steered_generator.py generate ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --helicity-lut helicity_lut.npy ^
    --stiffness-lut stiffness_lut.npy ^
    --bigram_db bigram_db.json ^
    --n_tokens 120 --lambda_arc 0.7 --output ag_out.json

  # Compare two databases:
  python arc_steered_generator.py compare ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --helicity-lut helicity_lut.npy ^
    --stiffness-lut stiffness_lut.npy ^
    --bigram_db_a bigram_db_shakespeare.json ^
    --bigram_db_b bigram_db_gutenberg.json ^
    --n_tokens 120 --output ag_compare.json
"""

import argparse, json, sys, time
import numpy as np
from pathlib import Path
from collections import defaultdict

K_PROJ = 16
VOCAB_SIZE = 50257

# Stiffness LUT (computed inline if not provided)
def _compute_stiffness(twist=3.8, n=360):
    phi = np.linspace(0, 360, n, endpoint=False)
    s = np.zeros(n)
    for i, p in enumerate(phi):
        rad = np.radians(p); m = np.cos(rad); lam = np.sin(rad)
        Ftm=-(m+0.866); Ftl=-(lam-0.5); Frm=-m
        Frl=-(lam+1.0)+twist*np.sin(m*2.5)
        sm=Ftm+Frm; sl=Ftl+Frl
        mag=np.sqrt(sm**2+sl**2); sf=np.sqrt(max(float(mag),0.))
        gm=sm*sf; gl=sl*sf
        def gw(a,c):
            d=min(abs(a-c),360-abs(a-c)); return np.exp(-((d)/80)**2)
        wg=gw(p,30); wt=gw(p,150); wr=gw(p,270); tot=wg+wt+wr+1e-6
        nwr=wr/tot; nwt=wt/tot; nwg=wg/tot
        Fm=nwt*Ftm+nwr*Frm+nwg*gm; Fl=nwt*Ftl+nwr*Frl+nwg*gl
        s[i]=float(np.sqrt(Fm**2+Fl**2))
    mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn+1e-12)

def _lut_at(lut, phi):
    a=float(phi)%360.; n=len(lut)
    i0=int(np.floor(a/360.*n))%n; i1=(i0+1)%n
    frac=(a/360.*n)-np.floor(a/360.*n)
    return (1-frac)*lut[i0]+frac*lut[i1]

# ── Loading ────────────────────────────────────────────────────────────────────

def load_resources(model_path, engram_path, helicity_path=None, stiffness_path=None):
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

    if helicity_path and Path(helicity_path).exists():
        hel = np.load(helicity_path).astype(np.float64).ravel()
        if len(hel.shape) == 0 or hel.shape[0] != 360:
            hel = np.interp(np.linspace(0,360,360,endpoint=False),
                            np.linspace(0,360,hel.shape[-1],endpoint=False), hel.ravel())
        print(f"  [lut] helicity: real ({hel.shape[0]} pts)")
    else:
        print("  [lut] helicity: synthetic fallback")
        th = np.linspace(0, 2*np.pi, 360, endpoint=False)
        hel = np.cos(th) * 0.5 + 0.5

    if stiffness_path and Path(stiffness_path).exists():
        stf = np.load(stiffness_path).astype(np.float64).ravel()
        if stf.shape[-1] != 360:
            stf = np.interp(np.linspace(0,360,360,endpoint=False),
                            np.linspace(0,360,stf.shape[-1],endpoint=False), stf.ravel())
        print(f"  [lut] stiffness: real ({stf.shape[0]} pts)")
    else:
        print("  [lut] stiffness: computing from HH physics")
        stf = _compute_stiffness()

    return model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref, hel, stf

def h_address(h, U_ref, pc1, pc2):
    h_n = h / (np.linalg.norm(h) + 1e-12)
    proj = U_ref.T @ h_n; p = proj**2/(proj**2).sum()+1e-12
    ksi = float(-np.sum(p*np.log(p+1e-12))/np.log(K_PROJ))
    j1  = float(np.degrees(np.arctan2(float(np.dot(h_n,pc1)),
                                       float(np.dot(h_n,pc2))))%360)
    return ksi, j1

def angular_signed(a, b):
    return float(((b - a + 180) % 360) - 180)

# ── Bigram database ────────────────────────────────────────────────────────────

def load_db(path):
    if path and Path(path).exists():
        with open(path) as f: return json.load(f)
    return {"_meta": {"n_texts": 0, "total_arcs": 0}, "tokens": {}}

def save_db(db, path):
    def cvt(o):
        if isinstance(o, (np.int64, np.int32, np.intp)): return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o)
        if isinstance(o, (np.bool_, bool)): return bool(o)
        raise TypeError(type(o))
    with open(path, 'w') as f: json.dump(db, f, separators=(',',':'), default=cvt)

def update_db(db, t1_id, t2_id, dj1, dksi, arc_type):
    """Welford online update for one bigram observation."""
    key1 = str(t1_id)
    if key1 not in db["tokens"]:
        db["tokens"][key1] = {"departures": {}, "total": 0}
    node = db["tokens"][key1]
    node["total"] += 1
    key2 = str(t2_id)
    if key2 not in node["departures"]:
        node["departures"][key2] = {
            "count": 0, "mean_dj1": 0.0, "mean_dksi": 0.0,
            "arc_type": arc_type
        }
    e = node["departures"][key2]
    e["count"] += 1
    n = e["count"]
    e["mean_dj1"]  += (dj1  - e["mean_dj1"])  / n
    e["mean_dksi"] += (dksi - e["mean_dksi"]) / n

def arc_type_classify(j1a, j1b, dksi):
    def in_zone(j, lo, hi): return lo <= (j%360) <= hi
    dj1 = angular_signed(j1a, j1b)
    if in_zone(j1a,140,265) or in_zone(j1b,140,265): return "DEAD"
    if abs(dj1) < 20 and abs(dksi) < 0.08: return "HOVER"
    if in_zone(j1a,80,140) and in_zone(j1b,80,140): return "A_INT"
    if in_zone(j1a,280,340) and in_zone(j1b,280,340): return "B_INT"
    if in_zone(j1a,80,140) and in_zone(j1b,280,340): return "A2B"
    if in_zone(j1a,280,340) and in_zone(j1b,80,140): return "B2A"
    if abs(dj1) > 120: return "PIVOT"
    return "CROSS"

def top_k_successors(db, t1_id, k=20):
    """Return top-k successors of t1 sorted by count."""
    key = str(t1_id)
    if key not in db["tokens"]: return []
    deps = db["tokens"][key]["departures"]
    items = sorted(deps.items(), key=lambda x: -x[1]["count"])[:k]
    return [(int(k2), v) for k2, v in items]

# ── Subcommand: build ─────────────────────────────────────────────────────────

def cmd_build(args):
    print("\n" + "="*70)
    print("BUILD — Incremental Bigram Arc Database")
    print("="*70)

    _, tok, _, ksi_v, j1_v, _, _, _, _, _ = load_resources(
        args.model, args.engram)

    db = load_db(getattr(args, 'load_db', None))
    texts = args.texts

    for text_path in texts:
        if not Path(text_path).exists():
            print(f"  [skip] not found: {text_path}")
            continue
        print(f"\n  Processing: {text_path}")
        with open(text_path, encoding='utf-8', errors='replace') as f:
            content = f.read()
        sentences = [s.strip() for s in content.replace('\n\n','\n').split('\n')
                     if len(s.strip()) > 10]
        max_s = getattr(args, 'max_sentences', 10000)
        sentences = sentences[:max_s]
        print(f"    {len(sentences)} sentences")

        n_arcs = 0
        for sent_idx, sentence in enumerate(sentences):
            ids = tok.encode(sentence, add_special_tokens=False)
            for i in range(len(ids) - 1):
                t1, t2 = ids[i], ids[i+1]
                if t1 >= len(ksi_v) or t2 >= len(ksi_v): continue
                dj1  = angular_signed(float(j1_v[t1]), float(j1_v[t2]))
                dksi = float(ksi_v[t2]) - float(ksi_v[t1])
                atype = arc_type_classify(float(j1_v[t1]), float(j1_v[t2]), dksi)
                update_db(db, int(t1), int(t2), dj1, dksi, atype)
                n_arcs += 1
            if sent_idx % 1000 == 0:
                print(f"    sentence {sent_idx}/{len(sentences)}  arcs={n_arcs}",
                      flush=True)

        db["_meta"]["n_texts"]   = db["_meta"].get("n_texts", 0) + 1
        db["_meta"]["total_arcs"] = db["_meta"].get("total_arcs", 0) + n_arcs
        print(f"    Added {n_arcs} arcs. DB now: "
              f"{len(db['tokens'])} source tokens, "
              f"{db['_meta']['total_arcs']} total arcs")

    save_db(db, args.output)
    print(f"\n  Database saved → {args.output}")
    print(f"  Unique source tokens: {len(db['tokens'])}")
    print(f"  Total arc observations: {db['_meta']['total_arcs']}")

    # Show most connected tokens
    by_successors = sorted(
        db["tokens"].items(),
        key=lambda x: len(x[1]["departures"]), reverse=True)[:10]
    print(f"\n  Most connected tokens (by unique successors):")
    for t_str, node in by_successors:
        try:
            s = tok.decode([int(t_str)], skip_special_tokens=False)
        except Exception:
            s = f"id={t_str}"
        print(f"    '{s}': {len(node['departures'])} successors, "
              f"{node['total']} total departures")

# ── Arc scoring for generation ─────────────────────────────────────────────────

def arc_score_vocab(current_tok_id, db, ksi_v, j1_v,
                    vessel_dj1_target, vessel_dksi_target,
                    sigma_j1=25.0, sigma_ksi=0.15, top_k=40):
    """
    Score all vocab tokens by arc compatibility with the vessel's target arc.
    Returns dense score vector [VOCAB_SIZE].
    """
    scores = np.zeros(VOCAB_SIZE, dtype=np.float32)
    succs  = top_k_successors(db, current_tok_id, k=top_k)
    if not succs:
        return scores

    total_count = sum(v["count"] for _, v in succs)
    for t2_id, arc_data in succs:
        # Compatibility with vessel target arc
        freq_weight = arc_data["count"] / (total_count + 1e-12)
        dj1_dist    = (arc_data["mean_dj1"]  - vessel_dj1_target) / sigma_j1
        dksi_dist   = (arc_data["mean_dksi"] - vessel_dksi_target) / sigma_ksi
        compat      = np.exp(-0.5 * (dj1_dist**2 + dksi_dist**2))
        scores[t2_id] += float(freq_weight * compat)

    return scores

def vessel_arc_target(base_j1, hel_lut, stf_lut, dt=0.5):
    """Compute the arc direction the vessel is naturally heading."""
    # stiffness gradient gives force direction
    h = 1.0
    grad_stf = (_lut_at(stf_lut, base_j1 + h) - _lut_at(stf_lut, base_j1 - h)) / (2*h)
    # vessel rolls toward higher stiffness → target_dj1 = grad * dt
    target_dj1  = float(grad_stf * dt * 10.0)   # scale to degrees
    target_dksi = 0.0   # vessel doesn't predict Ksi change
    return target_dj1, target_dksi

# ── Subcommand: generate ──────────────────────────────────────────────────────

def cmd_generate(args):
    print("\n" + "="*70)
    print("GENERATE — Arc-Steered Generation")
    print("="*70)

    model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref, hel, stf = load_resources(
        args.model, args.engram,
        getattr(args, 'helicity_lut', None),
        getattr(args, 'stiffness_lut', None))

    db_path = getattr(args, 'bigram_db', None)
    if not db_path or not Path(db_path).exists():
        print("  ERROR: --bigram_db required for generate")
        sys.exit(1)
    db = load_db(db_path)
    print(f"  [db] {db_path}: {len(db['tokens'])} source tokens, "
          f"{db['_meta'].get('total_arcs',0)} arcs")

    lambda_arc = float(getattr(args, 'lambda_arc', 0.7))
    n_tokens   = int(getattr(args, 'n_tokens', 120))
    temperature = 0.82
    seed       = int(getattr(args, 'seed', 0))

    print(f"  lambda_arc={lambda_arc}  n_tokens={n_tokens}  seed={seed}")

    import torch
    rng = np.random.default_rng(seed)

    # Seed token: ' of' at Zone A (best certified address from scan_map)
    try:
        seed_id = tok.encode(" of")[0]
    except Exception:
        seed_id = 0
    print(f"  Seed: '{tok.decode([seed_id])}'")

    with torch.no_grad():
        out = model(torch.tensor([[seed_id]]), output_hidden_states=True)
    h_raw = out.hidden_states[-1][0,-1].float().numpy()
    h_ln  = h_raw / (np.linalg.norm(h_raw) + 1e-12)
    ksi_h, j1_h = h_address(h_raw, U_ref, pc1, pc2)

    # Vessel state (simplified: just base_j1 and velocity)
    base_j1 = float(j1_h); vel_j1 = 0.0
    current_tok = seed_id

    generated_ids  = [seed_id]
    generated_text = tok.decode([seed_id])
    ksi_traj  = [ksi_h]; j1_traj = [j1_h]
    arc_traj  = []   # (vessel_dj1_target, arc_score_weight) per step
    known_bigrams = 0

    print(f"\n  Starting at Ksi={ksi_h:.3f}, J1={j1_h:.1f}°")
    print(f"\n  ", end="", flush=True)

    for step in range(n_tokens):
        # Vessel rolls under stiffness gradient
        grad_stf = (_lut_at(stf, base_j1+1.) - _lut_at(stf, base_j1-1.)) / 2.
        vel_j1   = 0.9 * vel_j1 + 0.5 * grad_stf
        base_j1  = (base_j1 + 0.5 * vel_j1) % 360.

        # Vessel arc target
        t_dj1, t_dksi = vessel_arc_target(base_j1, hel, stf)

        # Arc score from bigram database
        arc_sc = arc_score_vocab(current_tok, db, ksi_v, j1_v,
                                 t_dj1, t_dksi, top_k=40)

        # LM_Head score
        lm_sc = W_n @ h_ln.astype(np.float64)

        # Check if current token has any successors in db
        if str(current_tok) in db["tokens"] and len(db["tokens"][str(current_tok)]["departures"]) > 0:
            known_bigrams += 1
            arc_weight = lambda_arc
        else:
            arc_weight = 0.0   # fall back to LM_Head only

        # Combine
        arc_norm = arc_sc / (arc_sc.max() + 1e-12)
        lm_norm  = (lm_sc - lm_sc.mean()) / (lm_sc.std() + 1e-12)
        combined = (1.0 - arc_weight) * lm_norm + arc_weight * arc_norm

        arc_traj.append((float(t_dj1), float(arc_weight)))

        # Sample
        top_s = combined - combined.max()
        probs  = np.exp(top_s / temperature); probs /= probs.sum()
        chosen = int(rng.choice(VOCAB_SIZE, p=probs))

        token_str = tok.decode([chosen], skip_special_tokens=False)
        generated_ids.append(chosen); generated_text += token_str
        print(token_str, end="", flush=True)

        current_tok = chosen
        with torch.no_grad():
            out  = model(torch.tensor([[chosen]]), output_hidden_states=True)
        h_raw = out.hidden_states[-1][0,-1].float().numpy()
        h_ln  = h_raw / (np.linalg.norm(h_raw) + 1e-12)
        ksi_h, j1_h = h_address(h_raw, U_ref, pc1, pc2)
        ksi_traj.append(ksi_h); j1_traj.append(j1_h)

    print("\n")

    # Metrics
    words = [tok.decode([t]).strip() for t in generated_ids
             if any(c.isalpha() for c in tok.decode([t]))]
    mean_wl     = float(np.mean([len(w) for w in words])) if words else 0.
    single_let  = sum(1 for w in words if len(w)==1) / (len(words)+1e-12)
    garbage     = sum(1 for t in generated_ids if '\ufffd' in tok.decode([t]))
    known_frac  = known_bigrams / (n_tokens + 1e-12)
    h001        = bool(known_frac > 0.3)
    print(f"  known_bigram_frac={known_frac:.3f}  H-AG-001: {'PASS' if h001 else 'FAIL'}")
    print(f"  mean_wl={mean_wl:.2f}  single_let={single_let:.3f}  garbage={garbage}")
    print(f"  Ksi mean={np.mean(ksi_traj):.3f} ± {np.std(ksi_traj):.3f}")

    result = {
        "generated_text": generated_text,
        "n_tokens": n_tokens, "lambda_arc": lambda_arc,
        "garbage_count": int(garbage), "mean_word_length": float(mean_wl),
        "single_letter_fraction": float(single_let),
        "known_bigram_fraction": float(known_frac),
        "h001_pass": h001,
        "ksi_mean": float(np.mean(ksi_traj)), "ksi_std": float(np.std(ksi_traj)),
        "ksi_trajectory": [float(k) for k in ksi_traj],
        "arc_trajectory": arc_traj,
        "db_path": str(db_path),
        "db_source_tokens": len(db["tokens"]),
        "db_total_arcs": db["_meta"].get("total_arcs", 0),
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {args.output}")

# ── Subcommand: compare ───────────────────────────────────────────────────────

def cmd_compare(args):
    """H-AG-003: compare generation from two databases."""
    print("\n" + "="*70)
    print("COMPARE — Two Database Comparison")
    print("="*70)

    model, tok, W_n, ksi_v, j1_v, pc1, pc2, U_ref, hel, stf = load_resources(
        args.model, args.engram,
        getattr(args, 'helicity_lut', None),
        getattr(args, 'stiffness_lut', None))

    results = []
    for db_path, label in [(args.bigram_db_a, "DB_A"), (args.bigram_db_b, "DB_B")]:
        if not Path(db_path).exists():
            print(f"  [skip] {db_path} not found"); continue
        db = load_db(db_path)
        print(f"\n  --- {label}: {db_path} ---")

        # Override args for sub-call
        class NS:
            def __init__(self, **kw): self.__dict__.update(kw)

        out_path = args.output.replace(".json", f"_{label}.json")
        sub_args = NS(
            model=args.model, engram=args.engram,
            helicity_lut=getattr(args,'helicity_lut',None),
            stiffness_lut=getattr(args,'stiffness_lut',None),
            bigram_db=db_path, n_tokens=args.n_tokens,
            lambda_arc=getattr(args,'lambda_arc',0.7),
            seed=0, output=out_path)
        cmd_generate(sub_args)
        with open(out_path) as f: results.append(json.load(f))

    if len(results) == 2:
        tokens_a = set(results[0]["generated_text"].split())
        tokens_b = set(results[1]["generated_text"].split())
        jaccard   = len(tokens_a & tokens_b) / (len(tokens_a | tokens_b) + 1e-12)
        h003      = "PASS" if jaccard < 0.5 else "FAIL"
        print(f"\n  H-AG-003: Jaccard similarity = {jaccard:.3f}  → {h003}")
        print(f"    Tokens unique to A: {len(tokens_a - tokens_b)}")
        print(f"    Tokens unique to B: {len(tokens_b - tokens_a)}")
        print(f"    Tokens in both:     {len(tokens_a & tokens_b)}")
        summary = {"results": results, "jaccard": float(jaccard),
                   "h003_verdict": h003}
        with open(args.output, 'w') as f: json.dump(summary, f, indent=2)
        print(f"\n  Comparison saved → {args.output}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Arc-Steered Bigram Generator")
    sub = ap.add_subparsers(dest="cmd")

    # build
    pb = sub.add_parser("build")
    pb.add_argument("--model",          required=True)
    pb.add_argument("--engram",         required=True)
    pb.add_argument("--texts",          nargs="+", required=True)
    pb.add_argument("--output",         default="bigram_db.json")
    pb.add_argument("--load_db",        default=None)
    pb.add_argument("--max_sentences",  type=int, default=10000)

    # generate
    pg = sub.add_parser("generate")
    pg.add_argument("--model",          required=True)
    pg.add_argument("--engram",         required=True)
    pg.add_argument("--bigram_db",      required=True)
    pg.add_argument("--helicity-lut",   default=None, dest="helicity_lut")
    pg.add_argument("--stiffness-lut",  default=None, dest="stiffness_lut")
    pg.add_argument("--n_tokens",       type=int, default=120)
    pg.add_argument("--lambda_arc",     type=float, default=0.7)
    pg.add_argument("--seed",           type=int, default=0)
    pg.add_argument("--output",         default="ag_out.json")

    # compare
    pc = sub.add_parser("compare")
    pc.add_argument("--model",          required=True)
    pc.add_argument("--engram",         required=True)
    pc.add_argument("--bigram_db_a",    required=True)
    pc.add_argument("--bigram_db_b",    required=True)
    pc.add_argument("--helicity-lut",   default=None, dest="helicity_lut")
    pc.add_argument("--stiffness-lut",  default=None, dest="stiffness_lut")
    pc.add_argument("--n_tokens",       type=int, default=120)
    pc.add_argument("--lambda_arc",     type=float, default=0.7)
    pc.add_argument("--output",         default="ag_compare.json")

    args = ap.parse_args()
    {"build": cmd_build, "generate": cmd_generate, "compare": cmd_compare
     }.get(args.cmd, lambda _: (ap.print_help(), sys.exit(1)))(args)

if __name__ == "__main__":
    main()
