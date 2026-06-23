"""
wandering_sweep.py
ML-040 Phase 1c — Delta Sweep: Find the Planning Threshold
Pirouette Framework Volume 8 · CORE-003

Sweeps delta_end from a low value to a high value, holding delta_start fixed,
running MEASURE (natural) vs WANDER (corrected) at each point.

Goal: locate the Ksi delta at which:
  - Structured forward commitment (planning) emerges
  - Coherence degrades (too much correction)
  - Optimal generation quality is

Pre-registration:
  H-040-B: There exists a threshold delta_end* in (0, 0.10) above which
  WANDER output shows forward enumeration / structural commitment absent in
  MEASURE. Below this threshold, WANDER ≈ MEASURE. Above 2× threshold,
  WANDER degenerates.

  PRE-REGISTERED THEORETICAL CLAIM (Keaton Smith, June 2026):
  A model tuned to navigate the Ksi manifold and produce accurate outputs
  functions as a mathematical microscope for information space. The Ksi
  registry is a map; the Wandering Model is the explorer. Running the model
  at a controlled address IS a measurement of the structure at that address.
  Science becomes the thread connecting the fractal's discovery to itself —
  the closure of the fractal to itself in information space.
  Corollary: intelligence = the capacity to navigate the fractal such that
  the explorer can reconnect to its origin concept. The planning threshold
  delta_end* is the measurable signature of that capacity in this model.

Usage:
  python wandering_sweep.py --baseline_file baseline_math.json \\
    --prompt "The relationship between mathematical constants and information" \\
    --save sweep_results.json

  # Narrow sweep around the known-interesting region:
  python wandering_sweep.py --baseline_file baseline_math.json \\
    --delta_start_fixed -0.02 \\
    --delta_end_values -0.05,-0.02,0.00,0.02,0.04,0.06,0.08,0.10,0.14 \\
    --prompt "The relationship between mathematical constants and information" \\
    --save sweep_results.json

  # Different concept for replication:
  python wandering_sweep.py --baseline_file baseline_math.json \\
    --prompt "A neural network learns to" \\
    --save sweep_results_bio.json
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

K_PROJ = 16


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


def correct_ksi(h, V_top, target_ksi, current_ksi, step_size=0.05, n_steps=3):
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


def build_relative_bezier(baseline, delta_start, delta_end, delta_peak=None):
    n = len(baseline)
    t = np.linspace(0.0, 1.0, n)
    dp = delta_peak if delta_peak is not None else (delta_start + delta_end) / 2.0
    deltas = (1-t)**2 * delta_start + 2*(1-t)*t * dp + t**2 * delta_end
    return (np.array(baseline) + deltas).tolist()


# ── Model helpers ─────────────────────────────────────────────────────────────

def extract_gate_svds(model, n_layers):
    print(f"  Extracting gate SVDs (K={K_PROJ})...", flush=True)
    t0 = time.time()
    svds = []
    for i in range(n_layers):
        w = model.transformer.h[i].mlp.c_fc.weight.data.float().numpy()
        U, _, _ = np.linalg.svd(w, full_matrices=False)
        svds.append(U[:, :K_PROJ].astype(np.float32))
    print(f"  SVD done in {time.time()-t0:.1f}s", flush=True)
    return svds


def generate_one(
    model, tokenizer, prompt, schedule, svds,
    apply_correction, step_size, correction_steps,
    max_tokens, temperature, top_k,
):
    """Single generation pass. Returns (text, mean_ksi_pre, mean_ksi_post, n_tokens)."""
    n_layers = len(schedule)
    token_counter = [0]
    ksi_pre_all, ksi_post_all = [], []

    def make_hook(li):
        def hook_fn(module, args, output):
            h_batch = output[0]
            h = h_batch[0, -1, :].float().cpu().numpy().astype(np.float32)
            V_top = svds[li]
            tgt = schedule[li]
            ksi_pre, _ = measure_ksi(h, V_top)
            ksi_pre_all.append(ksi_pre)

            if apply_correction:
                h_new, ksi_post = correct_ksi(
                    h, V_top, tgt, ksi_pre, step_size, correction_steps)
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
    n_tokens = 0

    with torch.no_grad():
        for step in range(max_tokens):
            token_counter[0] = step
            out = model(generated)
            logits = out.logits[0, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).unsqueeze(0)
            generated = torch.cat([generated, next_tok], dim=1)
            n_tokens += 1
            if next_tok.item() == tokenizer.eos_token_id:
                break

    for h in hooks:
        h.remove()

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    mean_pre  = float(np.mean(ksi_pre_all))  if ksi_pre_all  else 0.0
    mean_post = float(np.mean(ksi_post_all)) if ksi_post_all else 0.0
    return text, mean_pre, mean_post, n_tokens


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(
    model, tokenizer, svds, n_layers,
    baseline, prompt,
    delta_start_fixed, delta_end_values,
    step_size, correction_steps,
    max_tokens, temperature, top_k,
):
    results = []

    # Run MEASURE once (delta=0 everywhere, natural baseline)
    print(f"\n{'='*66}")
    print(f"  MEASURE (natural baseline, no correction)")
    print(f"{'='*66}")
    flat_schedule = baseline[:]  # target = natural = no pressure
    text_m, pre_m, post_m, ntok_m = generate_one(
        model, tokenizer, prompt, flat_schedule, svds,
        apply_correction=False,
        step_size=step_size, correction_steps=correction_steps,
        max_tokens=max_tokens, temperature=temperature, top_k=top_k,
    )
    print(f"  Mean Ksi: {pre_m:.4f}  Tokens: {ntok_m}")
    print(f"\n  TEXT:\n  {text_m[:400].replace(chr(10), chr(10)+'  ')}")
    results.append({
        "mode": "MEASURE",
        "delta_start": 0.0,
        "delta_end": 0.0,
        "mean_ksi_pre": round(pre_m, 4),
        "mean_ksi_post": round(post_m, 4),
        "n_tokens": ntok_m,
        "text": text_m,
    })

    # Sweep WANDER across delta_end values
    for delta_end in delta_end_values:
        print(f"\n{'='*66}")
        print(f"  WANDER  delta_start={delta_start_fixed:+.3f}  delta_end={delta_end:+.3f}")
        print(f"{'='*66}")

        schedule = build_relative_bezier(
            baseline, delta_start_fixed, delta_end)

        text_w, pre_w, post_w, ntok_w = generate_one(
            model, tokenizer, prompt, schedule, svds,
            apply_correction=True,
            step_size=step_size, correction_steps=correction_steps,
            max_tokens=max_tokens, temperature=temperature, top_k=top_k,
        )

        ksi_lift = post_w - pre_m   # how much we lifted vs natural
        print(f"  Mean Ksi pre={pre_w:.4f}  post={post_w:.4f}  "
              f"lift={ksi_lift:+.4f}  Tokens={ntok_w}")
        print(f"\n  TEXT:\n  {text_w[:400].replace(chr(10), chr(10)+'  ')}")

        results.append({
            "mode": "WANDER",
            "delta_start": delta_start_fixed,
            "delta_end": delta_end,
            "mean_ksi_pre":  round(pre_w, 4),
            "mean_ksi_post": round(post_w, 4),
            "ksi_lift":      round(ksi_lift, 4),
            "n_tokens": ntok_w,
            "text": text_w,
        })

    return results


def print_summary_table(results):
    print(f"\n\n{'='*66}")
    print(f"  SWEEP SUMMARY TABLE")
    print(f"{'='*66}")
    print(f"  {'Mode':<10} {'Δend':>6}  {'Ksi_pre':>7}  {'Ksi_post':>8}  "
          f"{'Lift':>6}  {'Tokens':>6}")
    print(f"  {'-'*58}")
    for r in results:
        de    = r.get("delta_end", 0.0)
        lift  = r.get("ksi_lift", 0.0)
        print(f"  {r['mode']:<10} {de:+6.3f}  {r['mean_ksi_pre']:7.4f}  "
              f"{r['mean_ksi_post']:8.4f}  {lift:+6.4f}  {r['n_tokens']:6d}")
    print()
    print("  TEXT SNIPPETS (first 120 chars):")
    print(f"  {'-'*58}")
    for r in results:
        de   = r.get("delta_end", 0.0)
        tag  = f"{r['mode']} Δ={de:+.3f}" if r["mode"] == "WANDER" else "MEASURE      "
        snip = r["text"].replace("\n", " ")[:120]
        print(f"  [{tag}]")
        print(f"    {snip}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default=r"K:\models\gpt2-large")
    p.add_argument("--baseline_file",   required=True)
    p.add_argument("--prompt",          default="The relationship between mathematical constants and information")
    p.add_argument("--delta_start_fixed", type=float, default=-0.02)
    # nargs='+' + type=float: argparse distinguishes negative floats from flags
    # Pass space-separated on command line (no quotes needed):
    #   --delta_end_values -0.04 -0.02 0.00 0.02 0.04 0.06 0.08 0.10 0.14
    p.add_argument("--delta_end_values", nargs='+', type=float,
                   default=[-0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10],
                   help="Space-separated delta_end values (negatives OK without quotes)")
    p.add_argument("--step_size",        type=float, default=0.05)
    p.add_argument("--correction_steps", type=int,   default=3)
    p.add_argument("--max_tokens",  type=int,   default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k",       type=int,   default=50)
    p.add_argument("--save",        default=None, help="Save results JSON")
    args = p.parse_args()

    delta_end_values = args.delta_end_values  # already list[float] via nargs="+"

    print("=" * 66)
    print("  ML-040 Phase 1c — Delta Sweep: Planning Threshold")
    print("  Pirouette Framework Volume 8 · CORE-003")
    print("=" * 66)
    print(f"  Sweep range: delta_end ∈ {delta_end_values}")
    print(f"  delta_start fixed: {args.delta_start_fixed:+.3f}")
    print(f"  Prompt: \"{args.prompt}\"")

    # Load baseline
    with open(args.baseline_file) as f:
        bl_data = json.load(f)
    baseline = bl_data["baseline"]
    print(f"  Baseline mean Ksi: {bl_data['mean_ksi']:.4f}")

    # Load model
    print(f"\nLoading GPT-2-Large from {args.model} ...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(
        args.model, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    n_layers = model.config.n_layer
    svds = extract_gate_svds(model, n_layers)

    # Run sweep
    results = run_sweep(
        model, tok, svds, n_layers,
        baseline=baseline,
        prompt=args.prompt,
        delta_start_fixed=args.delta_start_fixed,
        delta_end_values=delta_end_values,
        step_size=args.step_size,
        correction_steps=args.correction_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print_summary_table(results)

    if args.save:
        out = {
            "experiment": "ML-040 Phase 1c Delta Sweep",
            "prompt": args.prompt,
            "baseline_file": args.baseline_file,
            "baseline_mean_ksi": bl_data["mean_ksi"],
            "delta_start_fixed": args.delta_start_fixed,
            "delta_end_values": delta_end_values,
            "step_size": args.step_size,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "results": results,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results saved to: {args.save}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
