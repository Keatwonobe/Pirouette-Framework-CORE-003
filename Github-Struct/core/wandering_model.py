"""
wandering_model.py
ML-040 Phase 1 — The Wandering Model (Inference Arc Probe)
Pirouette Framework Volume 8 · CORE-003

Tests inference-time Ksi navigation along certified registry arcs.
No training. Analytical gradient correction in projection space steers
each layer's hidden state toward a scheduled Ksi target.

The registry Ksi addresses are from ML-035 through ML-039 (GPT-2-Large).

Modes:
  --measure   : Log natural Ksi trajectory without correction
  --wander    : Apply correction, log trajectory + output
  --compare   : Run both modes on same prompt, side by side

Arc types:
  linear      : Straight line source → target
  bezier      : Quadratic Bezier source → peak → target (insight arc)
  flat        : Hold a single Ksi value across all layers (concept lock)
  custom      : Read schedule from --schedule_file (JSON list of floats, length=n_layers)

Usage examples:
  # Measure natural trajectory:
  python wandering_model.py --mode measure --prompt "Mathematical constants"

  # Linear arc math.pi → info.kolmogorov:
  python wandering_model.py --mode wander --arc linear \\
    --source math.pi --target info.kolmogorov \\
    --prompt "The relationship between mathematical constants and information"

  # Bezier arc through lang.metaphor (insight arc):
  python wandering_model.py --mode wander --arc bezier \\
    --source math.pi --target info.kolmogorov --peak lang.metaphor \\
    --prompt "Consider how precision and meaning connect"

  # Concept lock at bio.neural across all layers:
  python wandering_model.py --mode wander --arc flat --concept bio.neural \\
    --prompt "Neural computation relies on"

  # Full comparison (steered vs unsteered):
  python wandering_model.py --mode compare --arc bezier \\
    --source math.pi --target info.kolmogorov --peak lang.metaphor \\
    --prompt "Consider how precision and meaning connect"

  # Save trajectory to JSON:
  python wandering_model.py --mode wander --arc linear \\
    --source math.pi --target info.kolmogorov \\
    --prompt "Mathematical constants" --save_trajectory traj.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── GPT-2-Large Ksi Registry (ML-035 through ML-039, certified) ───────────────
# All values are GPT-2-Large projected Ksi (K=16, c_fc gate weights, U basis)
KSI_REGISTRY = {
    # Math constants — lowest Ksi band (universal floor, certified)
    "math.phi":         0.8045,
    "math.pi":          0.8054,
    "math.catalan":     0.8091,
    "math.ln2":         0.8124,
    "math.sqrt2":       0.8166,
    "math.e":           0.8202,
    # Physics — boundary constants
    "phys.alpha_fine":  0.8198,
    "phys.alpha_phys":  0.8280,
    "phys.hbar":        0.8281,
    "phys.kB":          0.8323,
    "phys.G":           0.8369,
    "phys.c":           0.8442,
    "phys.mu0":         0.8531,   # STABLE cross-model
    "phys.e_charge":    0.8537,   # STABLE cross-model
    # Biology
    "bio.neural":       0.8330,   # narrowest funnel in registry (0.054)
    "bio.dna":          0.8396,
    "bio.evolution":    0.8359,
    # Information
    "info.shannon":     0.8236,
    "info.carnot":      0.8232,
    "info.fisher":      0.8417,
    "info.kolmogorov":  0.8413,   # STABLE* cross-model
    # Dynamics / chaos
    "dyn.feigenbaum":   0.8217,
    "dyn.feigenbaum2":  0.8168,
    "dyn.wada_theta":   0.8383,
    "dyn.wada_scale":   0.8366,
    "dyn.ising_eta":    0.8340,
    "dyn.ising_nu":     0.8405,   # STABLE* cross-model
    "dyn.lyapunov_hh":  0.8407,   # widest funnel (0.1145)
    # Structure
    "struct.manifold":  0.8462,
    "struct.gauge":     0.8368,
    "struct.group":     0.8504,   # STABLE cross-model (GPT-2 v ML-033)
    "struct.hilbert":   0.8436,   # STABLE* cross-model
    # Language
    "lang.metaphor":    0.8271,   # natural transit hub
    "lang.story":       0.8227,
    "lang.grammar":     0.8504,   # STABLE* cross-model
}

K_PROJ = 16          # certified SVD projection rank
GPT2_LARGE_LAYERS = 36
GPT2_LARGE_D = 1280


# ── Ksi arithmetic ────────────────────────────────────────────────────────────

def measure_ksi(h: np.ndarray, V_top: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Measure projected Ksi from a single hidden state vector.

    Args:
        h     : hidden state [d], float32
        V_top : top-K left singular vectors of c_fc weight [d, K], float32

    Returns:
        ksi   : float in [0, 1]
        z     : projection coefficients [K]
    """
    z = V_top.T @ h          # [K]
    z_sq = z * z
    z_sum = float(z_sq.sum())
    if z_sum < 1e-12:
        return 0.5, z
    p = z_sq / z_sum
    # Shannon entropy, normalized by log(K)
    log_K = float(np.log(K_PROJ))
    p_safe = np.where(p > 1e-15, p, 1e-15)
    H = float(-np.sum(p * np.log(p_safe)))
    ksi = float(np.clip(H / log_K, 0.0, 1.0))
    return ksi, z


def ksi_gradient(z: np.ndarray) -> np.ndarray:
    """
    Analytical gradient of Ksi with respect to projection coefficients z.

    Derivation:
        p_i = z_i^2 / S,  S = sum z_j^2
        H   = -sum_i p_i log(p_i)   (raw Shannon entropy)
        Ksi = H / log(K)

    ∂Ksi/∂z_i = -(2*z_i) / (S * log(K)) * (log(p_i) + H)

    Gradient direction: increasing Ksi flattens spectrum (more uniform p);
    decreasing Ksi sharpens spectrum (more concentrated p).
    """
    z_sq = z * z
    S = float(z_sq.sum())
    if S < 1e-12:
        return np.zeros_like(z)
    p = z_sq / S
    log_K = float(np.log(K_PROJ))
    p_safe = np.where(p > 1e-15, p, 1e-15)
    H = float(-np.sum(p * np.log(p_safe)))
    log_p = np.log(p_safe)
    grad = -(2.0 * z) / (S * log_K) * (log_p + H)
    return grad


def correct_ksi(
    h: np.ndarray,
    V_top: np.ndarray,
    target_ksi: float,
    current_ksi: float,
    step_size: float = 0.15,
    n_steps: int = 3,
) -> tuple[np.ndarray, float]:
    """
    Nudge hidden state h toward target_ksi via iterative projection-space steps.

    The orthogonal residual (component of h not in span(V_top)) is preserved
    exactly — only the projected component is modified.

    Args:
        h           : hidden state [d], float32
        V_top       : gate SVD basis [d, K], float32
        target_ksi  : float
        current_ksi : float
        step_size   : gradient step magnitude
        n_steps     : correction iterations per layer

    Returns:
        h_corrected : [d], float32
        new_ksi     : float
    """
    delta = target_ksi - current_ksi
    if abs(delta) < 5e-5:
        return h, current_ksi

    # Decompose h into projection + orthogonal residual
    z = V_top.T @ h                   # [K]
    h_proj = V_top @ z                # projected component [d]
    h_res  = h - h_proj               # orthogonal residual [d] (preserved)
    proj_norm = float(np.linalg.norm(h_proj))

    sign = float(np.sign(delta))

    for _ in range(n_steps):
        grad = ksi_gradient(z)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-12:
            break
        z = z + sign * step_size * grad / grad_norm

    # Restore original projected-component magnitude to prevent scale drift
    new_proj = V_top @ z
    new_proj_norm = float(np.linalg.norm(new_proj))
    if new_proj_norm > 1e-8 and proj_norm > 1e-8:
        z = z * (proj_norm / new_proj_norm)

    h_corrected = V_top @ z + h_res
    new_ksi, _ = measure_ksi(h_corrected, V_top)
    return h_corrected.astype(np.float32), new_ksi


# ── Arc scheduler ─────────────────────────────────────────────────────────────

def build_arc(
    n_layers: int,
    arc_type: str,
    source_ksi: float = None,
    target_ksi: float = None,
    peak_ksi: float = None,
    concept_ksi: float = None,
) -> list[float]:
    """
    Build Ksi schedule [k_0, ..., k_{n_layers-1}].

    arc_type options:
      'linear'  — straight interpolation source → target
      'bezier'  — quadratic Bezier source → peak → target
      'flat'    — hold concept_ksi constant across all layers
    """
    t = np.linspace(0.0, 1.0, n_layers)

    if arc_type == "flat":
        assert concept_ksi is not None, "--concept required for flat arc"
        schedule = np.full(n_layers, concept_ksi)

    elif arc_type == "linear":
        assert source_ksi is not None and target_ksi is not None
        schedule = source_ksi + t * (target_ksi - source_ksi)

    elif arc_type == "bezier":
        assert source_ksi is not None and target_ksi is not None
        if peak_ksi is None:
            peak_ksi = (source_ksi + target_ksi) / 2.0
        # Quadratic Bezier: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
        schedule = ((1 - t)**2 * source_ksi
                    + 2 * (1 - t) * t * peak_ksi
                    + t**2 * target_ksi)
    else:
        raise ValueError(f"Unknown arc type: {arc_type}")

    return schedule.tolist()


# ── Model helpers ─────────────────────────────────────────────────────────────

def extract_gate_svds(model: GPT2LMHeadModel, n_layers: int) -> list[np.ndarray]:
    """
    Pre-compute top-K_PROJ left singular vectors of each layer's c_fc gate.

    GPT-2 Conv1D weight: [d_model, d_ffn] (input dim × output dim).
    SVD: W = U S V^T; U ∈ [d, d], S ∈ [d], V^T ∈ [d, d_ffn].
    We use V_top = U[:, :K] (certified: Conv1D → use U).

    Returns:
        list of [d, K] float32 arrays, one per layer
    """
    print(f"  Extracting gate SVDs (K={K_PROJ}) for {n_layers} layers...", flush=True)
    t0 = time.time()
    svds = []
    for i in range(n_layers):
        w = model.transformer.h[i].mlp.c_fc.weight.data.float().numpy()
        # w: [d_model, d_ffn]  Conv1D stores [in, out]
        U, _, _ = np.linalg.svd(w, full_matrices=False)
        # U: [d_model, min(d_model, d_ffn)] = [d_model, d_model] for gpt2-large
        V_top = U[:, :K_PROJ].astype(np.float32)  # [d, K]
        svds.append(V_top)
    print(f"  SVD done in {time.time()-t0:.1f}s", flush=True)
    return svds


def load_model_and_tokenizer(model_path: str):
    print(f"Loading GPT-2-Large from {model_path} ...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
    tok.pad_token = tok.eos_token
    n_layers = model.config.n_layer
    d_model  = model.config.n_embd
    print(f"  {n_layers} layers, d={d_model}", flush=True)
    return model, tok


# ── Generation with optional Ksi correction ───────────────────────────────────

def run_generation(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    schedule: list[float],
    svds: list[np.ndarray],
    max_new_tokens: int = 120,
    step_size: float = 0.15,
    correction_steps: int = 3,
    apply_correction: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = True,
) -> tuple[str, list[dict]]:
    """
    Generate text with optional per-layer Ksi correction.

    Returns:
        text        : full decoded text (prompt + generation)
        trajectory  : list of per-layer records for each generation step
    """
    n_layers = len(schedule)
    token_counter = [0]
    trajectory = []

    def make_hook(layer_idx: int):
        def hook_fn(module, args, output):
            # GPT2Block returns (hidden_states,) or (hidden_states, present, ...)
            h_batch = output[0]  # [batch, seq, d]

            # Last-token hidden state (generation is autoregressive)
            h_np = h_batch[0, -1, :].float().cpu().numpy().astype(np.float32)
            V_top = svds[layer_idx]
            tgt = schedule[layer_idx]

            ksi_pre, _ = measure_ksi(h_np, V_top)
            record = {
                "token":   token_counter[0],
                "layer":   layer_idx,
                "pre":     round(ksi_pre, 5),
                "target":  round(tgt, 5),
            }

            if apply_correction:
                h_new, ksi_post = correct_ksi(
                    h_np, V_top, tgt, ksi_pre,
                    step_size=step_size,
                    n_steps=correction_steps,
                )
                record["post"] = round(ksi_post, 5)
                record["moved"] = round(ksi_post - ksi_pre, 5)

                h_tensor = torch.tensor(
                    h_new, dtype=h_batch.dtype, device=h_batch.device
                )
                h_batch = h_batch.clone()
                h_batch[0, -1, :] = h_tensor

                if isinstance(output, tuple):
                    return (h_batch,) + output[1:]
                return h_batch
            else:
                record["post"] = record["pre"]
                record["moved"] = 0.0

            trajectory.append(record)
        return hook_fn

    hooks = [
        model.transformer.h[i].register_forward_hook(make_hook(i))
        for i in range(n_layers)
    ]

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    generated  = input_ids.clone()

    mode_label = "WANDER" if apply_correction else "MEASURE"
    print(f"\n  Generating ({mode_label}, max_tokens={max_new_tokens})...", flush=True)

    with torch.no_grad():
        for step in range(max_new_tokens):
            token_counter[0] = step
            out    = model(generated)
            logits = out.logits[0, -1, :]  # [vocab]

            if greedy:
                next_tok = logits.argmax().unsqueeze(0).unsqueeze(0)
            else:
                if temperature != 1.0:
                    logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[-1]] = float("-inf")
                probs   = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).unsqueeze(0)

            generated = torch.cat([generated, next_tok], dim=1)

            if next_tok.item() == tokenizer.eos_token_id:
                print(f"  [EOS at generation step {step}]", flush=True)
                break

    for h in hooks:
        h.remove()

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, trajectory


# ── Trajectory summary ────────────────────────────────────────────────────────

def summarize_trajectory(trajectory: list[dict], n_layers: int, label: str):
    """Print per-layer Ksi summary averaged across generation steps."""
    if not trajectory:
        print(f"  [{label}] No trajectory data.")
        return

    # Average per layer across all tokens
    from collections import defaultdict
    layer_data = defaultdict(list)
    for rec in trajectory:
        layer_data[rec["layer"]].append(rec)

    print(f"\n  [{label}] Per-layer Ksi averages (first 6 / last 6 of {n_layers}):")
    print(f"    {'Layer':>5}  {'Pre':>7}  {'Post':>7}  {'Target':>7}  {'Error':>7}")

    indices = list(range(min(6, n_layers))) + list(range(max(6, n_layers - 6), n_layers))
    for i in sorted(set(indices)):
        recs = layer_data.get(i, [])
        if not recs:
            continue
        pre    = np.mean([r["pre"]    for r in recs])
        post   = np.mean([r["post"]   for r in recs])
        target = np.mean([r["target"] for r in recs])
        err    = post - target
        print(f"    {i:5d}  {pre:7.4f}  {post:7.4f}  {target:7.4f}  {err:+7.4f}")

    # Global stats
    all_err = [abs(r["post"] - r["target"]) for r in trajectory]
    print(f"\n    Mean |error|: {np.mean(all_err):.5f}")
    print(f"    Max  |error|: {np.max(all_err):.5f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="ML-040 Wandering Model Phase 1")

    # Core arguments
    p.add_argument("--model",  default=r"K:/models/gpt2-large",
                   help="Path to GPT-2-Large (local)")
    p.add_argument("--mode",   choices=["measure", "wander", "compare"],
                   default="wander",
                   help="measure=no correction  wander=with correction  compare=both")
    p.add_argument("--prompt", default="The relationship between mathematical constants and information",
                   help="Generation prompt")

    # Arc definition
    p.add_argument("--arc",     choices=["linear", "bezier", "flat", "custom"],
                   default="bezier")
    p.add_argument("--source",  default="math.pi",  help="Source concept (registry key)")
    p.add_argument("--target",  default="info.kolmogorov", help="Target concept")
    p.add_argument("--peak",    default="lang.metaphor",   help="Bezier midpoint concept")
    p.add_argument("--concept", default=None, help="Concept for flat arc")
    p.add_argument("--schedule_file", default=None,
                   help="JSON file with list of floats (custom arc)")

    # Correction hyperparameters
    p.add_argument("--step_size",        type=float, default=0.15,
                   help="Gradient step size per correction iteration")
    p.add_argument("--correction_steps", type=int,   default=3,
                   help="Number of correction iterations per layer")

    # Generation settings
    p.add_argument("--max_tokens", type=int,   default=120)
    p.add_argument("--greedy",     action="store_true", default=True)
    p.add_argument("--temperature",type=float, default=1.0)
    p.add_argument("--top_k",      type=int,   default=0)

    # Output
    p.add_argument("--save_trajectory", default=None,
                   help="Save trajectory JSON to this path")

    args = p.parse_args()

    print("=" * 70)
    print("  ML-040 Wandering Model Phase 1 — Inference Arc Probe")
    print("  Pirouette Framework Volume 8 · CORE-003")
    print("=" * 70)

    # ── Validate registry keys ────────────────────────────────────────────────
    for key in [args.source, args.target, args.peak]:
        if key and key not in KSI_REGISTRY:
            print(f"ERROR: '{key}' not in registry.")
            print(f"Available: {sorted(KSI_REGISTRY.keys())}")
            sys.exit(1)
    if args.concept and args.concept not in KSI_REGISTRY:
        print(f"ERROR: --concept '{args.concept}' not in registry.")
        sys.exit(1)

    source_ksi  = KSI_REGISTRY.get(args.source)
    target_ksi  = KSI_REGISTRY.get(args.target)
    peak_ksi    = KSI_REGISTRY.get(args.peak)
    concept_ksi = KSI_REGISTRY.get(args.concept) if args.concept else None

    print(f"\n  Arc:    {args.arc.upper()}")
    if args.arc == "flat":
        print(f"  Concept: {args.concept} (Ksi={concept_ksi:.4f})")
    elif args.arc in ("linear", "bezier"):
        print(f"  Source: {args.source} (Ksi={source_ksi:.4f})")
        print(f"  Target: {args.target} (Ksi={target_ksi:.4f})")
        if args.arc == "bezier":
            print(f"  Peak:   {args.peak}  (Ksi={peak_ksi:.4f})")
    print(f"  Mode:   {args.mode.upper()}")
    print(f"  Prompt: \"{args.prompt}\"")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model)
    n_layers = model.config.n_layer

    # ── Build schedule ────────────────────────────────────────────────────────
    if args.arc == "custom":
        assert args.schedule_file, "--schedule_file required for custom arc"
        with open(args.schedule_file) as f:
            schedule = json.load(f)
        assert len(schedule) == n_layers, \
            f"Schedule length {len(schedule)} != n_layers {n_layers}"
    else:
        schedule = build_arc(
            n_layers,
            arc_type=args.arc,
            source_ksi=source_ksi,
            target_ksi=target_ksi,
            peak_ksi=peak_ksi,
            concept_ksi=concept_ksi,
        )

    # ── Extract SVDs ──────────────────────────────────────────────────────────
    svds = extract_gate_svds(model, n_layers)

    # ── Generation arguments shared between modes ─────────────────────────────
    gen_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        schedule=schedule,
        svds=svds,
        max_new_tokens=args.max_tokens,
        step_size=args.step_size,
        correction_steps=args.correction_steps,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    results = {}

    # ── MEASURE ───────────────────────────────────────────────────────────────
    if args.mode in ("measure", "compare"):
        text_m, traj_m = run_generation(**gen_kwargs, apply_correction=False)
        results["measure"] = {"text": text_m, "trajectory": traj_m}
        summarize_trajectory(traj_m, n_layers, "MEASURE")

    # ── WANDER ────────────────────────────────────────────────────────────────
    if args.mode in ("wander", "compare"):
        text_w, traj_w = run_generation(**gen_kwargs, apply_correction=True)
        results["wander"] = {"text": text_w, "trajectory": traj_w}
        summarize_trajectory(traj_w, n_layers, "WANDER")

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)

    if "measure" in results:
        print("\n  [MEASURE — natural arc, no correction]")
        print("  " + "-" * 66)
        print("  " + results["measure"]["text"].replace("\n", "\n  "))

    if "wander" in results:
        print("\n  [WANDER — Ksi-steered along arc]")
        print("  " + "-" * 66)
        print("  " + results["wander"]["text"].replace("\n", "\n  "))

    print("\n" + "=" * 70)

    # ── Save trajectory ───────────────────────────────────────────────────────
    if args.save_trajectory:
        out = {
            "experiment": "ML-040 Phase 1",
            "arc_type": args.arc,
            "source": args.source,
            "target": args.target,
            "peak": args.peak if args.arc == "bezier" else None,
            "source_ksi": source_ksi,
            "target_ksi": target_ksi,
            "peak_ksi": peak_ksi if args.arc == "bezier" else None,
            "schedule": schedule,
            "prompt": args.prompt,
            "step_size": args.step_size,
            "correction_steps": args.correction_steps,
        }
        for mode_key, mode_data in results.items():
            out[mode_key] = {
                "text": mode_data["text"],
                # Truncate trajectory to first 500 records to keep file size manageable
                "trajectory": mode_data["trajectory"][:500],
            }
        with open(args.save_trajectory, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Trajectory saved to: {args.save_trajectory}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
