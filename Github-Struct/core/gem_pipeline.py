"""
gem_pipeline.py
ML-041 — Geometric Mixture of Experts (GeM)
Pirouette Framework Volume 8 · CORE-003

Architecture:
  1. CRAWL: N expert agents, each navigating to a distinct Ksi address.
     Experts are configurable as TIGHT (specialists, low step_size, flat arc,
     stay close to assigned delta) or LOOSE (explorers, higher step_size,
     bezier arc, traverse the space before settling at target).

  2. COLLECT: Expert outputs form the corpus — a set of (text, address,
     ksi_lift) tuples. Each text is grounded in the epistemic register at
     its Ksi address.

  3. SYNTHESIZE: The same model, navigated to the integration address
     (default Δ=+0.08, peak naming capacity from Phase 1), receives all
     expert outputs as context and produces a synthesis.

  4. ENGRAM: The full output — expert corpus + synthesis + address vector —
     is the Geometric Research Engram. Its address is the vector of all
     expert deltas + synthesis delta. A thought that required N+1 coordinate
     positions to construct.

Pre-registration:
  H-041-A: GeM synthesis scores higher on multifaceted coverage than any
  single-address output from the same model.
  H-041-B: Tight specialists + loose explorers produces richer synthesis than
  uniform expert configuration.
  H-041-C: The engram address vector is reproducible: same question + same
  expert config + same baseline produces semantically similar engram.

Usage:
  # Default expert config, altruism question:
  python gem_pipeline.py --baseline_file baseline_math.json \\
    --question "The cause of altruistic behavior is" \\
    --save engram_altruism.json

  # Math question:
  python gem_pipeline.py --baseline_file baseline_math.json \\
    --question "The relationship between mathematical constants and information" \\
    --save engram_math.json

  # Custom expert config file:
  python gem_pipeline.py --baseline_file baseline_math.json \\
    --question "Consciousness arises from" \\
    --expert_config experts_consciousness.json \\
    --save engram_consciousness.json

  # Print the engram summary only (from saved JSON):
  python gem_pipeline.py --print_engram engram_altruism.json
"""

import argparse
import json
import sys
import time
import textwrap
from collections import defaultdict

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

K_PROJ = 16

# ── Default expert profiles ───────────────────────────────────────────────────
# Drawn from Phase 1 sweep results. Each expert targets a Ksi address where
# a distinct epistemic register was observed.
#
# TIGHT experts: step_size=0.03, correction_steps=5, flat arc
#   → Specialists. Stay close to their assigned address.
#   → Produce outputs deep in their native register.
#
# LOOSE experts: step_size=0.08, correction_steps=2, bezier arc
#   → Explorers. Higher correction tolerance, traverse more of the space.
#   → Produce bridging/hybrid outputs that span multiple registers.

DEFAULT_EXPERTS = [
    # Tight specialists — certified epistemic registers from Phase 1
    {"name": "empirical",    "delta": -0.04, "tight": True,
     "note": "Peer-reviewed / multi-factor causal register"},
    {"name": "critical",     "delta": -0.02, "tight": True,
     "note": "Critical theory / interrogative register"},
    {"name": "journalistic", "delta":  0.00, "tight": True,
     "note": "Science journalism / survey register"},
    {"name": "aphoristic",   "delta":  0.02, "tight": False,
     "note": "Philosophical / aphoristic register (loose — finds local basin)"},
    {"name": "theological",  "delta":  0.04, "tight": True,
     "note": "Cultural / theological register (nearest native basin)"},
    {"name": "economic",     "delta":  0.06, "tight": False,
     "note": "Consequentialist / economic register (loose — bridges domains)"},
    {"name": "mechanistic",  "delta":  0.08, "tight": True,
     "note": "Neuroscience / mechanism register (peak naming address)"},
    {"name": "evolutionary", "delta":  0.10, "tight": False,
     "note": "Evolutionary / genetics register (loose — wide traversal)"},
    {"name": "longitudinal", "delta":  0.14, "tight": True,
     "note": "Historical / comparative register"},
]

TIGHT_PARAMS = {"step_size": 0.03, "correction_steps": 5, "arc": "flat"}
LOOSE_PARAMS = {"step_size": 0.08, "correction_steps": 2, "arc": "bezier"}

# Synthesis address: Δ=+0.08 (peak naming from Phase 1 sweep)
SYNTHESIS_DELTA = 0.08
SYNTHESIS_STEP_SIZE = 0.05
SYNTHESIS_CORRECTION_STEPS = 3


# ── Ksi arithmetic ────────────────────────────────────────────────────────────

def measure_ksi(h, V_top):
    z = V_top.T @ h
    z_sq = z * z
    z_sum = float(z_sq.sum())
    if z_sum < 1e-12:
        return 0.5, z
    p = z_sq / z_sum
    p_safe = np.where(p > 1e-15, p, 1e-15)
    H = float(-np.sum(p * np.log(p_safe)))
    return float(np.clip(H / np.log(K_PROJ), 0.0, 1.0)), z


def ksi_gradient(z):
    z_sq = z * z
    S = float(z_sq.sum())
    if S < 1e-12:
        return np.zeros_like(z)
    p = z_sq / S
    p_safe = np.where(p > 1e-15, p, 1e-15)
    H = float(-np.sum(p * np.log(p_safe)))
    return -(2.0 * z) / (S * np.log(K_PROJ)) * (np.log(p_safe) + H)


def correct_ksi(h, V_top, target_ksi, current_ksi, step_size, n_steps):
    delta = target_ksi - current_ksi
    if abs(delta) < 5e-5:
        return h, current_ksi
    z = V_top.T @ h
    h_proj = V_top @ z
    h_res = h - h_proj
    proj_norm = float(np.linalg.norm(h_proj))
    sign = float(np.sign(delta))
    for _ in range(n_steps):
        grad = ksi_gradient(z)
        gn = float(np.linalg.norm(grad))
        if gn < 1e-12:
            break
        z = z + sign * step_size * grad / gn
    new_proj = V_top @ z
    nnorm = float(np.linalg.norm(new_proj))
    if nnorm > 1e-8 and proj_norm > 1e-8:
        z = z * (proj_norm / nnorm)
    h_out = (V_top @ z + h_res).astype(np.float32)
    ksi_out, _ = measure_ksi(h_out, V_top)
    return h_out, ksi_out


def build_schedule(baseline, delta_end, delta_start=-0.02, arc="flat", delta_peak=None):
    n = len(baseline)
    t = np.linspace(0.0, 1.0, n)
    base = np.array(baseline)
    if arc == "flat":
        deltas = np.full(n, delta_end)
    elif arc == "linear":
        deltas = delta_start + t * (delta_end - delta_start)
    elif arc == "bezier":
        dp = delta_peak if delta_peak is not None else (delta_start + delta_end) / 2.0
        deltas = (1-t)**2*delta_start + 2*(1-t)*t*dp + t**2*delta_end
    else:
        raise ValueError(f"Unknown arc: {arc}")
    return (base + deltas).tolist()


# ── Model helpers ─────────────────────────────────────────────────────────────

def extract_gate_svds(model, n_layers):
    print(f"  Extracting SVDs (K={K_PROJ}, {n_layers} layers)...", flush=True)
    t0 = time.time()
    svds = []
    for i in range(n_layers):
        w = model.transformer.h[i].mlp.c_fc.weight.data.float().numpy()
        U, _, _ = np.linalg.svd(w, full_matrices=False)
        svds.append(U[:, :K_PROJ].astype(np.float32))
    print(f"  SVD done in {time.time()-t0:.1f}s", flush=True)
    return svds


def generate(model, tokenizer, prompt, schedule, svds,
             step_size, correction_steps, max_tokens,
             temperature, top_k, apply_correction=True):
    """Single generation pass. Returns (text, mean_ksi_pre, mean_ksi_post, n_tokens)."""
    n_layers = len(schedule)
    ksi_pre_all, ksi_post_all = [], []

    def make_hook(li):
        def hook_fn(module, args, output):
            h_batch = output[0]
            h = h_batch[0, -1, :].float().cpu().numpy().astype(np.float32)
            ksi_pre, _ = measure_ksi(h, svds[li])
            ksi_pre_all.append(ksi_pre)
            if apply_correction:
                h_new, ksi_post = correct_ksi(
                    h, svds[li], schedule[li], ksi_pre, step_size, correction_steps)
                ksi_post_all.append(ksi_post)
                h_t = torch.tensor(h_new, dtype=h_batch.dtype, device=h_batch.device)
                h_batch = h_batch.clone()
                h_batch[0, -1, :] = h_t
                return (h_batch,) + output[1:] if isinstance(output, tuple) else h_batch
            else:
                ksi_post_all.append(ksi_pre)
        return hook_fn

    hooks = [model.transformer.h[i].register_forward_hook(make_hook(i))
             for i in range(n_layers)]
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    generated = input_ids.clone()
    n_tok = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            out = model(generated)
            logits = out.logits[0, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).unsqueeze(0)
            generated = torch.cat([generated, next_tok], dim=1)
            n_tok += 1
            if next_tok.item() == tokenizer.eos_token_id:
                break

    for h in hooks:
        h.remove()

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    mean_pre  = float(np.mean(ksi_pre_all))  if ksi_pre_all  else 0.0
    mean_post = float(np.mean(ksi_post_all)) if ksi_post_all else 0.0
    return text, mean_pre, mean_post, n_tok


# ── Synthesis prompt builder ──────────────────────────────────────────────────

def build_synthesis_prompt(question, expert_results):
    """
    Build the synthesis prompt from the expert corpus.
    The synthesizer receives all expert outputs tagged by their epistemic register.
    """
    lines = [
        f"Question: {question}",
        "",
        "The following perspectives were generated from different epistemic addresses "
        "in the conceptual manifold. Each represents a distinct explanatory tradition:",
        "",
    ]
    for i, r in enumerate(expert_results):
        lines.append(f"[Perspective {i+1} — {r['name']} register, Ksi lift={r['ksi_lift']:+.4f}]")
        # Trim to first 300 chars to fit context window cleanly
        snippet = r['text'].replace('\n', ' ').strip()[:300]
        lines.append(snippet)
        lines.append("")
    lines.append(
        "Synthesis: Integrating the above perspectives into a single coherent account "
        "that preserves their disagreements and identifies what they share:"
    )
    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_gem_pipeline(
    model, tokenizer, svds, n_layers,
    baseline, question, experts,
    synthesis_delta, synthesis_step_size, synthesis_correction_steps,
    max_tokens, temperature, top_k,
):
    expert_results = []

    print(f"\n{'='*66}")
    print(f"  PHASE 1: EXPERT CRAWL  ({len(experts)} experts)")
    print(f"{'='*66}")

    for i, expert in enumerate(experts):
        name  = expert.get("name", f"Dynamic_{i+1}")
        delta = expert["delta"]
        tight = expert.get("tight", True)
        note  = expert.get("note", "")
        params = TIGHT_PARAMS if tight else LOOSE_PARAMS
        arc    = params["arc"]
        ss     = params["step_size"]
        cs     = params["correction_steps"]

        print(f"\n  Expert {i+1}/{len(experts)}: [{name}] "
              f"Δ={delta:+.3f}  {'TIGHT' if tight else 'LOOSE'}  arc={arc}", flush=True)

        # Loose explorers: bezier arc from -0.02 to their target (traverse on the way)
        if arc == "flat":
            schedule = build_schedule(baseline, delta_end=delta, arc="flat")
        else:
            schedule = build_schedule(baseline, delta_end=delta, delta_start=-0.02,
                                      arc="bezier", delta_peak=(delta / 2.0))

        text, pre, post, ntok = generate(
            model, tokenizer, question, schedule, svds,
            step_size=ss, correction_steps=cs,
            max_tokens=max_tokens, temperature=temperature, top_k=top_k,
        )

        lift = post - pre
        result = {
            "name":     name,
            "delta":    delta,
            "tight":    tight,
            "arc":      arc,
            "note":     note,
            "step_size": ss,
            "ksi_pre":  round(pre,  4),
            "ksi_post": round(post, 4),
            "ksi_lift": round(lift, 4),
            "n_tokens": ntok,
            "text":     text,
        }
        expert_results.append(result)

        snippet = text.replace('\n', ' ').strip()[:120]
        print(f"  Ksi: pre={pre:.4f}  post={post:.4f}  lift={lift:+.4f}  tokens={ntok}")
        print(f"  → {snippet}")

    # ── SYNTHESIS ──────────────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"  PHASE 2: SYNTHESIS  (Δ={synthesis_delta:+.3f}, "
          f"peak naming address from Phase 1)")
    print(f"{'='*66}")

    synthesis_prompt = build_synthesis_prompt(question, expert_results)
    synthesis_schedule = build_schedule(
        baseline, delta_end=synthesis_delta, arc="flat")

        # Hard-truncate synthesis prompt to fit GPT-2's 1024-token context window
    _synth_ids = tokenizer(synthesis_prompt, return_tensors="pt")["input_ids"]
    _max_ctx   = getattr(model.config, "n_positions", 1024)
    _max_prompt = _max_ctx - min(max_tokens, 200) - 20
    if _synth_ids.shape[1] > _max_prompt:
        _synth_ids      = _synth_ids[:, :_max_prompt]
        synthesis_prompt = tokenizer.decode(_synth_ids[0], skip_special_tokens=True)
        print(f"  [Synthesis prompt truncated to {_synth_ids.shape[1]} tokens]", flush=True)
    synth_text, synth_pre, synth_post, synth_tok = generate(
        model, tokenizer, synthesis_prompt, synthesis_schedule, svds,
        step_size=synthesis_step_size,
        correction_steps=synthesis_correction_steps,
        max_tokens=max_tokens, temperature=temperature, top_k=top_k,
    )

    synth_lift = synth_post - synth_pre
    synthesis_result = {
        "delta":     synthesis_delta,
        "ksi_pre":   round(synth_pre,  4),
        "ksi_post":  round(synth_post, 4),
        "ksi_lift":  round(synth_lift, 4),
        "n_tokens":  synth_tok,
        "text":      synth_text,
        "prompt_used": synthesis_prompt,
    }

    print(f"\n  Synthesis Ksi: pre={synth_pre:.4f}  post={synth_post:.4f}  "
          f"lift={synth_lift:+.4f}  tokens={synth_tok}")

    # ── ENGRAM ─────────────────────────────────────────────────────────────────
    # The engram address: the full vector of expert deltas + synthesis delta.
    # This is the coordinate of the composite thought in the manifold.
    engram_address = [r["delta"] for r in expert_results] + [synthesis_delta]

    engram = {
        "experiment":       "ML-041 GeM Pipeline",
        "question":         question,
        "baseline_mean_ksi": float(np.mean(baseline)),
        "engram_address":   engram_address,
        "n_experts":        len(experts),
        "expert_results":   expert_results,
        "synthesis":        synthesis_result,
    }

    return engram


def print_engram(engram):
    print(f"\n{'='*66}")
    print(f"  GEOMETRIC RESEARCH ENGRAM")
    print(f"  ML-041 · Pirouette Framework Volume 8")
    print(f"{'='*66}")
    print(f"\n  Question: {engram['question']}")
    print(f"  Engram address: {engram['engram_address']}")
    print(f"  n_experts: {engram['n_experts']}")

    print(f"\n  EXPERT CORPUS:")
    print(f"  {'Expert':<14} {'Δ':>6}  {'Type':<6}  {'Lift':>7}  {'First 100 chars'}")
    print(f"  {'-'*62}")
    for r in engram["expert_results"]:
        snip = r["text"].replace("\n", " ").strip()[:80]
        t = "TIGHT" if r["tight"] else "LOOSE"
        print(f"  {r['name']:<14} {r['delta']:+6.3f}  {t:<6}  {r['ksi_lift']:+7.4f}  {snip}")

    print(f"\n  SYNTHESIS (Δ={engram['synthesis']['delta']:+.3f}, "
          f"lift={engram['synthesis']['ksi_lift']:+.4f}):")
    print(f"  {'-'*62}")
    synth = engram["synthesis"]["text"]
    # Strip the prompt prefix if present
    marker = "Synthesis:"
    if marker in synth:
        synth = synth[synth.rfind(marker) + len(marker):].strip()
    for line in textwrap.wrap(synth, width=62):
        print(f"  {line}")
    print(f"\n{'='*66}")


def main():
    p = argparse.ArgumentParser(description="ML-041 Geometric Mixture of Experts")
    p.add_argument("--model",           default=r"K:\models\gpt2-large")
    p.add_argument("--baseline_file",   default=None,
                   help="JSON from wandering_model_v2.py --mode calibrate")
    p.add_argument("--question",        default="The cause of altruistic behavior is")
    p.add_argument("--expert_config",   default=None,
                   help="JSON file with expert list (overrides DEFAULT_EXPERTS)")
    p.add_argument("--synthesis_delta", type=float, default=SYNTHESIS_DELTA)
    p.add_argument("--max_tokens",  type=int,   default=100)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k",       type=int,   default=50)
    p.add_argument("--save",        default=None, help="Save engram JSON")
    p.add_argument("--print_engram",default=None,
                   help="Print a previously saved engram JSON and exit")
    args = p.parse_args()

    # Print-only mode
    if args.print_engram:
        with open(args.print_engram) as f:
            engram = json.load(f)
        print_engram(engram)
        return

    if not args.baseline_file:
        print("ERROR: --baseline_file required (run wandering_model_v2.py --mode calibrate first)")
        sys.exit(1)

    print("=" * 66)
    print("  ML-041 Geometric Mixture of Experts")
    print("  Pirouette Framework Volume 8 · CORE-003")
    print("=" * 66)
    print(f"  Question: \"{args.question}\"")
    print(f"  Synthesis address: Δ={args.synthesis_delta:+.3f}")

    # Load baseline
    with open(args.baseline_file) as f:
        bl = json.load(f)
    baseline = bl["baseline"]
    print(f"  Baseline mean Ksi: {bl['mean_ksi']:.4f}")

    # Load expert config
    if args.expert_config:
        with open(args.expert_config) as f:
            loaded_config = json.load(f)
        
        # ADDED PIPING: Check if the JSON loaded as a dict and extract the list
        if isinstance(loaded_config, dict):
            # Assumes the list of experts is stored under a key like 'experts'
            experts = loaded_config.get("experts", loaded_config.get("expert_roster", []))
        else:
            # Assumes it loaded as a flat list
            experts = loaded_config
            
        print(f"  Expert config: {args.expert_config} ({len(experts)} experts)")
    else:
        experts = DEFAULT_EXPERTS
        print(f"  Expert config: DEFAULT ({len(experts)} experts)")

    print(f"\n  Expert roster:")
    for i, e in enumerate(experts):
        t = "TIGHT" if e.get("tight", True) else "LOOSE"
        name = e.get("name", f"Dynamic_{i+1}")
        print(f"    [{name:<14}] Δ={e['delta']:+.3f}  {t}  {e.get('note','')}")

    # Load model
    print(f"\nLoading GPT-2-Large from {args.model} ...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(
        args.model, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    n_layers = model.config.n_layer
    svds = extract_gate_svds(model, n_layers)

    # Run pipeline
    engram = run_gem_pipeline(
        model, tok, svds, n_layers,
        baseline=baseline,
        question=args.question,
        experts=experts,
        synthesis_delta=args.synthesis_delta,
        synthesis_step_size=SYNTHESIS_STEP_SIZE,
        synthesis_correction_steps=SYNTHESIS_CORRECTION_STEPS,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print_engram(engram)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(engram, f, indent=2)
        print(f"\n  Engram saved to: {args.save}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
