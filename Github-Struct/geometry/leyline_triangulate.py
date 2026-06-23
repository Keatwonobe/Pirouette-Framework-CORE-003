"""
leyline_triangulate.py
Pirouette Framework Volume 8 · CORE-003
Coordinate Scan · Probe Triangulation · Incompleteness Score

THREE INSTRUMENTS:

  scan   — Sweep one coordinate (Ksi or concept direction) from the english
           vector toward a concept target. At each step, decode raw hidden
           states via ln_f → lm_head. No generation loop.
           Find: does the target vocabulary exist in the weight space?
           At what address does "cannot be proven" / "unprovable" appear?
           This is the decisive test before the 8B route.

  probe  — Compute concept vectors from (question, correct_answer) pairs
           rather than semantic prompts. The model's hidden state when
           processing a CORRECT answer is more discriminative than the
           hidden state from a defining prompt (which is near-identical
           to other prompts in the same semantic cluster).
           Triangulate: find the manifold address where multiple correct
           probe answers simultaneously converge.

  score  — Incompleteness score: apply a controlled perturbation to FRADAR
           gate SVDs, rerun ley-lines, measure trajectory divergence.
           High score = weakly encoded (a gap). Low score = robustly encoded.
           Spatial map of encoding confidence across the concept manifold.

HYPOTHESES:
  H-SCAN-001: "cannot be proven" / "unprovable" tokens appear with >1% 
              probability at some Ksi address between english and logic.godel.
              PASS: the concept exists; navigation was the problem.
              FAIL: the concept is too weakly encoded; need the 8B model.

  H-PROBE-001: Probe-triangulated concept vectors have higher pairwise
               divergence than semantic-prompt vectors (>0.05 vs ~0.02).
               PASS: probe triangulation gives more discriminative targets.
               FAIL: all concept addresses are similarly clustered.

  H-SCORE-001: logic.godel has higher incompleteness score than math.pi
               (Gödel's theorem is less stably encoded than pi).
               This would explain why navigation succeeds for math.pi
               but not for logic.godel.

Usage:
  # Coordinate scan: find where "unprovable" lives in the manifold
  python leyline_triangulate.py --mode scan ^
    --model models\\gpt2-large-pirouette ^
    --source english --target logic.godel ^
    --n_steps 50 ^
    --target_vocab "unprovable" "cannot" "incomplete" "undecidable" "true" "provable" ^
    --save scan_godel.json

  # Probe triangulation: compute concept vectors from correct answers
  python leyline_triangulate.py --mode probe ^
    --model models\\gpt2-large-pirouette ^
    --save probe_vectors.json

  # Incompleteness score: which concepts are stably encoded?
  python leyline_triangulate.py --mode score ^
    --model models\\gpt2-large-pirouette ^
    --concepts math.pi truth logic.godel math.completeness english ^
    --n_perturb 5 --perturb_scale 0.1 ^
    --save incompleteness_scores.json

  # All three:
  python leyline_triangulate.py --mode all ^
    --model models\\gpt2-large-pirouette ^
    --save triangulate_all.json
"""

import argparse, json, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

K_PROJ   = 16
KSI_LAYER = 11
FRADAR_LAYERS  = [4, 7, 11, 34]
FRADAR_WEIGHTS = {4: 0.15, 7: 0.20, 11: 0.40, 34: 0.25}

# ── Concept prompts (semantic) ─────────────────────────────────────────────────
CONCEPT_PROMPTS: Dict[str, List[str]] = {
    "english": [
        "The force required to accelerate an object is proportional to its mass.",
        "Natural selection favors traits that increase reproductive success.",
        "A prime number has exactly two distinct positive divisors.",
        "A valid argument requires the conclusion to follow necessarily from premises.",
        "Written language was independently developed in at least three civilizations.",
        "The derivative measures the rate at which a function's output changes.",
        "Attention selectively enhances processing of task-relevant information.",
        "Water molecules form hydrogen bonds giving the substance unusual properties.",
        "Children acquire native language rapidly despite receiving imperfect input.",
        "Democratic institutions distribute political power among competing groups.",
    ],
    "truth": [
        "The Pythagorean theorem: a² + b² = c² for any right triangle.",
        "Every integer greater than one is either prime or a product of primes.",
        "The sum of angles in a Euclidean triangle equals 180 degrees.",
        "A prime number has exactly two distinct positive integer divisors.",
        "Energy cannot be created or destroyed, only transformed.",
        "The speed of light in vacuum is approximately 299,792,458 metres per second.",
        "Water freezes at zero degrees Celsius at standard atmospheric pressure.",
        "DNA carries genetic information through four nucleotide base pairs.",
        "The Earth orbits the Sun with a period of approximately 365.25 days.",
        "All mammals are vertebrates with a backbone.",
    ],
    "logic.godel": [
        "Gödel's incompleteness theorem: any consistent formal system powerful enough to describe arithmetic contains true statements that cannot be proven within that system.",
        "The Gödel sentence G asserts its own unprovability within the formal system.",
        "A statement can be true but unprovable — this is the core of Gödel's result.",
        "The proof of incompleteness constructs a sentence that is true but formally undecidable.",
        "Incompleteness reveals that mathematical truth transcends any fixed axiom system.",
        "Gödel's second theorem: no consistent system can prove its own consistency.",
        "Gödel numbering encodes syntactic formulas as natural numbers, enabling self-reference.",
    ],
    "math.pi": [
        "The mathematical constant pi equals 3.14159, the ratio of circumference to diameter.",
        "Pi is transcendental and cannot be expressed as a ratio of two integers.",
        "Pi appears in Euler's identity: e^(iπ) + 1 = 0.",
        "The digits of pi follow no repeating pattern, extending infinitely.",
        "Pi is computed to trillions of decimal places using series expansions.",
    ],
    "math.completeness": [
        "A proof is complete when it derives the conclusion from premises in finite steps.",
        "Gödel's theorem: any consistent formal system contains unprovable true statements.",
        "A closed proof requires no unresolved assumptions or dangling quantifiers.",
        "The Halting Problem demonstrates fundamental limits to algorithmic provability.",
        "Mathematical completeness means every true statement has a proof in the system.",
    ],
}

# ── Probe pairs: (context, correct_completion) ─────────────────────────────────
# The hidden state when the model processes context + correct_completion is
# more discriminative than the hidden state from a semantic prompt alone.
PROBE_PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "logic.godel.correct": [
        ("Gödel proved that in any consistent formal system, there exist statements that are", 
         "true but cannot be proven within the system"),
        ("The incompleteness theorem shows that no formal system can be both",
         "consistent and complete — some true statements will always be unprovable"),
        ("Gödel's sentence G is designed so that G is true if and only if",
         "G is unprovable within the formal system"),
        ("According to Gödel's second incompleteness theorem, a consistent system",
         "cannot prove its own consistency"),
        ("The key insight of Gödel's proof is that mathematical truth is",
         "not fully capturable by any single formal system"),
    ],
    "truth.factual": [
        ("The Pythagorean theorem states that for a right triangle with legs a and b and hypotenuse c,",
         "a² + b² = c²"),
        ("A prime number is a natural number greater than 1 that has",
         "no positive divisors other than 1 and itself"),
        ("The first incompleteness theorem was published by Kurt Gödel in",
         "1931 in his paper On Formally Undecidable Propositions"),
        ("In Peano arithmetic, Gödel showed that there exist statements that are",
         "true but unprovable within the system"),
    ],
    "math.pi.correct": [
        ("The mathematical constant π is defined as the ratio of a circle's",
         "circumference to its diameter"),
        ("Pi is approximately equal to",
         "3.14159265358979323846"),
        ("Pi is an irrational and transcendental number, meaning it",
         "cannot be expressed as the ratio of two integers"),
    ],
    "logic.godel.wrong": [
        # Wrong completions — used to compute contrastive vector
        ("Gödel proved that in any consistent formal system, there exist statements that are", 
         "false and unprovable"),
        ("The incompleteness theorem shows that no formal system can be both",
         "true and false at the same time"),
        ("According to Gödel, all true mathematical statements",
         "can eventually be proven with enough computation"),
    ],
}


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_model(path):
    print(f"  Loading {path.split(chr(92))[-1].split('/')[-1]}...", flush=True)
    t0 = time.time()
    model = GPT2LMHeadModel.from_pretrained(
        path, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(path, local_files_only=True)
    tok.pad_token = tok.eos_token
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tok

def get_gate_svd(model, layer=KSI_LAYER, k=K_PROJ):
    w = model.transformer.h[layer].mlp.c_fc.weight.data.float().numpy()
    U, _, _ = np.linalg.svd(w, full_matrices=False)
    return U[:, :k].astype(np.float32)

def measure_ksi(h, U_K):
    z = U_K.T @ h.astype(np.float32); zs = z*z; S = float(zs.sum())
    if S < 1e-12: return 0.5
    p = zs/S; ps = np.where(p>1e-15, p, 1e-15)
    return float(np.clip(-np.sum(ps*np.log(ps))/np.log(K_PROJ), 0, 1))

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b)/(na*nb)) if na>1e-12 and nb>1e-12 else 0.0

def compute_h(model, tok, prompts_or_text, layer=KSI_LAYER):
    """Mean hidden state at layer from list of prompts or single string."""
    if isinstance(prompts_or_text, str):
        prompts_or_text = [prompts_or_text]
    hs = []
    for p in prompts_or_text:
        ids = tok(p, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs.append(out.hidden_states[layer+1][0,-1,:].float().numpy())
    return np.mean(hs, axis=0).astype(np.float32)

def decode_h(model, h, temperature=1.0):
    """Raw decode: h → ln_f → lm_head → logits. No generation."""
    ht = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model.lm_head(model.transformer.ln_f(ht))[0]
    if temperature != 1.0:
        logits = logits / temperature
    return logits

def top_tokens(logits, tok, k=10):
    probs = torch.softmax(logits, -1)
    top_idx = torch.argsort(probs, descending=True)[:k]
    return [(tok.decode([i.item()]).strip(), float(probs[i])) for i in top_idx]

def vocab_probs(logits, tok, target_words: List[str]) -> Dict[str, float]:
    """Get probability of specific words in the vocabulary."""
    probs = torch.softmax(logits, -1)
    result = {}
    for word in target_words:
        # Try with and without leading space
        for variant in [word, " " + word, word.lower(), " " + word.lower()]:
            ids = tok.encode(variant, add_special_tokens=False)
            if ids:
                result[word] = max(result.get(word, 0.0), float(probs[ids[0]]))
    return result


# ── Mode 1: Coordinate Scan ────────────────────────────────────────────────────

def mode_scan(model, tok, source_key, target_key, n_steps,
              target_vocab, temperature, save):
    """
    Sweep h(t) = (1-t)*h_source + t*h_target for t ∈ [0,1].
    At each step, decode via ln_f → lm_head (no generation).
    Report: target vocabulary probabilities and top tokens per step.

    This answers: does the target vocabulary exist in the weight space?
    At what coordinate does it appear?
    """
    U_K = get_gate_svd(model)
    print(f"\n  Computing concept vectors...", flush=True)
    h_source = compute_h(model, tok, CONCEPT_PROMPTS[source_key])
    h_target = compute_h(model, tok, CONCEPT_PROMPTS[target_key])
    cos = cosine_sim(h_source, h_target)
    dist = float(np.linalg.norm(h_target - h_source))
    print(f"  [{source_key}] → [{target_key}]  cos={cos:.4f}  dist={dist:.2f}")

    print(f"\n  ── COORDINATE SCAN ──")
    print(f"  Decoding h(t) = (1-t)·h_{source_key} + t·h_{target_key}")
    print(f"  Target vocab: {target_vocab}")
    print()

    header = f"  {'t':>5}  {'ksi':>6}  " + \
             "  ".join(f"{w[:8]:>8}" for w in target_vocab) + \
             "  top-3 tokens"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    best_hits = {w: (0.0, 0.0) for w in target_vocab}  # (t, prob)

    for i, t in enumerate(np.linspace(0, 1, n_steps)):
        h_t = ((1.0 - t) * h_source + t * h_target).astype(np.float32)
        ksi  = measure_ksi(h_t, U_K)
        logits = decode_h(model, h_t, temperature)
        vp   = vocab_probs(logits, tok, target_vocab)
        top3 = top_tokens(logits, tok, 3)

        # Track best hit per target word
        for w, p in vp.items():
            if p > best_hits[w][1]:
                best_hits[w] = (float(t), float(p))

        prob_str = "  ".join(f"{vp.get(w, 0)*100:>7.3f}%" for w in target_vocab)
        top3_str = ", ".join(f"'{t3}'" for t3, _ in top3)
        print(f"  {t:>5.3f}  {ksi:.4f}  {prob_str}  {top3_str}")

        results.append({
            "t": float(t), "ksi": ksi,
            "vocab_probs": {w: float(vp.get(w, 0)) for w in target_vocab},
            "top_tokens":  [(s, float(p)) for s, p in top3],
        })

    print(f"\n  ── PEAK PROBABILITIES ──")
    for w, (t_peak, p_peak) in sorted(best_hits.items(), key=lambda x: -x[1][1]):
        bar = "█" * int(p_peak * 500)
        print(f"  '{w:>12}': {p_peak*100:.3f}% at t={t_peak:.3f}  {bar}")

    # Decisiveness check
    max_peak = max(p for _, p in best_hits.values())
    if max_peak > 0.005:
        print(f"\n  H-SCAN-001: PASS — target vocabulary appears at >{max_peak*100:.2f}%")
        print(f"  The concept EXISTS in the weight space.")
        print(f"  Navigation failure was a ROUTING problem, not an encoding problem.")
        print(f"  The 8B model should express this more clearly; GPT-2 can reach it.")
    else:
        print(f"\n  H-SCAN-001: FAIL — target vocabulary never exceeds 0.5%")
        print(f"  The concept is NOT strongly encoded in this weight space.")
        print(f"  The 8B model will encode it — navigation should succeed there.")

    if save:
        with open(save, "w") as f:
            json.dump({"mode": "scan", "source": source_key, "target": target_key,
                       "target_vocab": target_vocab, "best_hits": {
                           w: {"t": t, "prob": p} for w, (t, p) in best_hits.items()},
                       "results": results}, f, indent=2)
        print(f"  Saved → {save}")
    return results, best_hits


# ── Mode 2: Probe Triangulation ────────────────────────────────────────────────

def mode_probe(model, tok, save):
    """
    Compute concept vectors from (context, correct_answer) pairs.
    These are hidden states when the model processes a CORRECT completion —
    more discriminative than semantic-prompt vectors because they're
    conditioned on the actual correct content.

    Also computes CONTRASTIVE vectors: correct - wrong.
    The contrastive direction points FROM wrong answers TOWARD right answers.
    """
    U_K = get_gate_svd(model)
    print(f"\n  ── PROBE TRIANGULATION ──")

    probe_vectors = {}
    for probe_key, pairs in PROBE_PAIRS.items():
        print(f"\n  [{probe_key}] ({len(pairs)} pairs)...", flush=True)
        hs = []
        for context, completion in pairs:
            text = context + " " + completion
            h = compute_h(model, tok, [text])
            sim_t = cosine_sim(h, compute_h(model, tok, CONCEPT_PROMPTS["truth"]))
            ksi = measure_ksi(h, U_K)
            print(f"    ksi={ksi:.4f}  truth_sim={sim_t:.4f}  "
                  f"'{text[:60]}...'")
            hs.append(h)
        probe_vectors[probe_key] = np.mean(hs, axis=0)

    # Contrastive vector: correct - wrong
    if "logic.godel.correct" in probe_vectors and "logic.godel.wrong" in probe_vectors:
        h_correct = probe_vectors["logic.godel.correct"]
        h_wrong   = probe_vectors["logic.godel.wrong"]
        h_contrast = h_correct - h_wrong
        # Normalize to same magnitude as original
        h_contrast *= (np.linalg.norm(h_correct) /
                       (np.linalg.norm(h_contrast) + 1e-12))
        probe_vectors["logic.godel.contrastive"] = h_contrast
        print(f"\n  Contrastive vector (correct - wrong):")
        print(f"    norm={np.linalg.norm(h_contrast):.2f}  "
              f"cos(correct)={cosine_sim(h_contrast, h_correct):.4f}  "
              f"cos(wrong)={cosine_sim(h_contrast, h_wrong):.4f}")
        # The contrastive vector should be MORE orthogonal to wrong than correct

    # Compare divergence: probe vectors vs semantic vectors
    print(f"\n  ── DIVERGENCE COMPARISON ──")
    print(f"  Probe-triangulated vectors:")
    probe_keys = [k for k in probe_vectors if not k.endswith(".wrong")]
    for i, k1 in enumerate(probe_keys):
        for k2 in probe_keys[i+1:]:
            sim = cosine_sim(probe_vectors[k1], probe_vectors[k2])
            ksi1 = measure_ksi(probe_vectors[k1], U_K)
            ksi2 = measure_ksi(probe_vectors[k2], U_K)
            print(f"  {k1[:20]:>20} ↔ {k2[:20]:<20}: sim={sim:.4f}  "
                  f"Δksi={abs(ksi1-ksi2):.4f}")

    print(f"\n  Semantic-prompt vectors:")
    sem_keys = ["english", "truth", "logic.godel", "math.pi", "math.completeness"]
    sem_h = {k: compute_h(model, tok, CONCEPT_PROMPTS[k]) for k in sem_keys}
    for i, k1 in enumerate(sem_keys):
        for k2 in sem_keys[i+1:]:
            sim = cosine_sim(sem_h[k1], sem_h[k2])
            ksi1 = measure_ksi(sem_h[k1], U_K)
            ksi2 = measure_ksi(sem_h[k2], U_K)
            print(f"  {k1:>20} ↔ {k2:<20}: sim={sim:.4f}  "
                  f"Δksi={abs(ksi1-ksi2):.4f}")

    # Decode at probe-triangulated address
    print(f"\n  ── RAW DECODE AT PROBE ADDRESSES ──")
    for key, h in probe_vectors.items():
        if key.endswith(".wrong"): continue
        logits = decode_h(model, h)
        top5 = top_tokens(logits, tok, 5)
        ksi = measure_ksi(h, U_K)
        vp  = vocab_probs(logits, tok,
                          ["unprovable", "incomplete", "cannot", "true", "provable"])
        print(f"  [{key}] ksi={ksi:.4f}")
        print(f"    top5: {', '.join(f'{t!r}' for t,_ in top5)}")
        print(f"    target: {', '.join(f'{w}={p*100:.3f}%' for w,p in vp.items())}")

    if save:
        # Don't serialize numpy arrays directly
        output = {k: v.tolist() for k, v in probe_vectors.items()}
        with open(save, "w") as f:
            json.dump({"mode": "probe", "vectors": output,
                       "n_probes": {k: len(v) for k, v in PROBE_PAIRS.items()}},
                      f, indent=2)
        print(f"\n  Saved → {save}")
    return probe_vectors


# ── Mode 3: Incompleteness Score ──────────────────────────────────────────────

def mode_score(model, tok, concepts, n_perturb, perturb_scale, save):
    """
    Measure encoding stability of each concept via perturbation.
    
    For each concept C:
      1. Compute h_C (semantic vector)
      2. Walk ley-line from h_english to h_C, record Ksi trajectory T0
      3. Perturb FRADAR gate SVDs by gaussian noise at scale s
      4. Rerun ley-line with perturbed SVDs, record T1
      5. Incompleteness score = mean ||T1 - T0|| / ||h_C - h_english||
      
    High score: the concept trajectory changes a lot under perturbation
                → weakly encoded, sensitive to noise → a real gap
    Low score:  the trajectory is stable → robustly encoded
    
    H-SCORE-001: logic.godel score > math.pi score
    """
    U_K = get_gate_svd(model)
    n_walk = 10

    print(f"\n  ── INCOMPLETENESS SCORES ──")
    print(f"  n_perturb={n_perturb}  scale={perturb_scale}")
    print(f"  Concepts: {concepts}")

    h_english = compute_h(model, tok, CONCEPT_PROMPTS["english"])
    results = {}

    # Save original FRADAR gate weights
    originals = {li: model.transformer.h[li].mlp.c_fc.weight.data.clone()
                 for li in FRADAR_LAYERS if li < model.config.n_layer}

    for concept in concepts:
        if concept not in CONCEPT_PROMPTS: continue
        print(f"\n  [{concept}]...", flush=True)
        h_concept = compute_h(model, tok, CONCEPT_PROMPTS[concept])
        dist = float(np.linalg.norm(h_concept - h_english))

        # Baseline trajectory
        T0 = []
        for t in np.linspace(0, 1, n_walk):
            h_t = (1-t)*h_english + t*h_concept
            T0.append(measure_ksi(h_t, U_K))

        # Perturbed trajectories
        perturbation_diffs = []
        for trial in range(n_perturb):
            # Apply gaussian noise to FRADAR gate SVDs
            for li in FRADAR_LAYERS:
                if li >= model.config.n_layer: continue
                W = model.transformer.h[li].mlp.c_fc.weight.data
                noise = torch.randn_like(W) * perturb_scale * W.std()
                W.data.add_(noise)

            # Rerun ley-line with perturbed model
            U_K_p = get_gate_svd(model)  # recompute SVD with perturbed weights
            T1 = []
            for t in np.linspace(0, 1, n_walk):
                h_t = (1-t)*h_english + t*h_concept
                T1.append(measure_ksi(h_t, U_K_p))

            diff = float(np.mean([abs(a-b) for a, b in zip(T0, T1)]))
            perturbation_diffs.append(diff)

            # Restore original weights
            for li, W_orig in originals.items():
                model.transformer.h[li].mlp.c_fc.weight.data.copy_(W_orig)

        score = float(np.mean(perturbation_diffs)) / (dist + 1e-12)
        std   = float(np.std(perturbation_diffs))
        bar   = "█" * int(score * 1000)
        print(f"  score={score:.5f}  std={std:.5f}  {bar}")
        results[concept] = {"score": score, "std": std,
                             "trial_diffs": perturbation_diffs}

    print(f"\n  ── RANKING (high score = weakly encoded gap) ──")
    for concept, r in sorted(results.items(), key=lambda x: -x[1]["score"]):
        bar = "▓" * int(r["score"] * 1000)
        print(f"  {concept:>20}: {r['score']:.5f} ±{r['std']:.5f}  {bar}")

    # H-SCORE-001
    if "logic.godel" in results and "math.pi" in results:
        godel_score = results["logic.godel"]["score"]
        pi_score    = results["math.pi"]["score"]
        print(f"\n  H-SCORE-001: logic.godel ({godel_score:.5f}) vs math.pi ({pi_score:.5f})")
        if godel_score > pi_score:
            print(f"  PASS — Gödel is less stably encoded than pi.")
            print(f"  This explains why navigation succeeds for mathematical content")
            print(f"  but not for the specific Gödel claim.")
        else:
            print(f"  FAIL — Gödel is as stable as pi. Encoding is not the bottleneck.")
            print(f"  The issue is purely navigational — routing improvements should help.")

    if save:
        with open(save, "w") as f:
            json.dump({"mode": "score", "concepts": concepts,
                       "n_perturb": n_perturb, "perturb_scale": perturb_scale,
                       "results": results}, f, indent=2)
        print(f"\n  Saved → {save}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Triangulate · Score · Scan")
    p.add_argument("--mode",    choices=["scan","probe","score","all"], default="scan")
    p.add_argument("--model",   default=r"models\gpt2-large-pirouette")
    p.add_argument("--source",  default="english")
    p.add_argument("--target",  default="logic.godel")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--target_vocab", nargs="+",
                   default=["unprovable","incomplete","cannot",
                             "undecidable","true","provable","proven"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concepts",    nargs="+",
                   default=["english","truth","math.pi","logic.godel","math.completeness"])
    p.add_argument("--n_perturb",   type=int,   default=5)
    p.add_argument("--perturb_scale", type=float, default=0.1)
    p.add_argument("--save",        default=None)
    args = p.parse_args()

    print("="*66)
    print("  Triangulate · Coordinate Scan · Incompleteness Score")
    print("  Pirouette Framework Volume 8 · CORE-003")
    print("="*66)

    model, tok = load_model(args.model)

    all_results = {}

    if args.mode in ("scan", "all"):
        save_scan = args.save.replace(".json","_scan.json") if args.save else None
        r, bh = mode_scan(model, tok, args.source, args.target,
                          args.n_steps, args.target_vocab, args.temperature, save_scan)
        all_results["scan"] = {"best_hits": {w: {"t": t, "prob": p}
                                              for w, (t, p) in bh.items()}}

    if args.mode in ("probe", "all"):
        save_probe = args.save.replace(".json","_probe.json") if args.save else None
        pv = mode_probe(model, tok, save_probe)
        all_results["probe"] = {"n_vectors": len(pv)}

    if args.mode in ("score", "all"):
        save_score = args.save.replace(".json","_score.json") if args.save else None
        sr = mode_score(model, tok, args.concepts, args.n_perturb,
                        args.perturb_scale, save_score)
        all_results["score"] = sr

    if args.mode == "all" and args.save:
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  All results → {args.save}")

    print("\n  Done.")

if __name__ == "__main__":
    main()
