"""
gem_pipeline_arc.py
Geometric Mixture of Experts — Arc-Augmented with System Prompt
Pirouette Framework Volume 8 · CORE-003 · ML-071

THE ARCHITECTURE
=================

LAYER 0 — SYSTEM PROMPT
  A single persistent context that orients every generation.
  Injected as a prefix before the user prompt.
  Keeps the manifold anchored near the task domain.
  Prevents undirected generation from finding random attractors.

LAYER 1 — EXPERT ENSEMBLE (N experts, each at distinct delta)
  Each expert runs wandering_model_arc generation:
    - Ksi correction via Bezier arc at its assigned delta
    - Arc augmentation from bigram_db + bivector_field
    - Different deltas → different manifold addresses → different epistemic registers
  Default deltas: [-0.04, 0.0, +0.04, +0.08, +0.12]
  Each expert sees: system_prompt + user_prompt
  Each expert produces: text + ksi_post + arc_confidence

LAYER 2 — SYNTHESIS AGENT (at delta_c = +0.16)
  Sees: system_prompt + user_prompt + ALL expert outputs
  Runs at the certified quality-maximum address (delta_c)
  Arc-augmented like the experts
  Goal: integrate, bridge, find what none of the experts said alone

WHAT THIS CLOSES
=================
Prior results showed:
  - Base wandering model: coherent but motivational/circular
  - Arc augmented: research register, counterfactuals, "a priori"
  - Wiki bivector field: academic citations, Theory of Mind questionnaire
  - Undirected queries: random attractors (EulerPad, distress voice, counting)

The system prompt grounds the query.
The expert ensemble covers the manifold from multiple addresses.
The synthesis agent extracts the dividend.

DIVIDEND HYPOTHESIS (from ML paper):
  synthesis_lift / mean_expert_lift = dividend
  At delta_c = +0.16: dividend ≈ 1.63x (3 experts) to 2.0x (9 experts)
  The synthesis produces bridge concepts not present in any expert individually.

PRE-REGISTERED:
  H-GEM-001: Arc-augmented synthesis dividend > non-augmented baseline
    PASS: dividend_arc > dividend_base * 1.1
  H-GEM-002: System prompt reduces undirected generation failures
    PASS: no EulerPad / distress-voice / counting runs in 10 attempts
  H-GEM-003: Expert diversity correlates with synthesis quality
    PASS: Jaccard similarity between expert outputs < 0.4

Usage:
  python gem_pipeline_arc.py run ^
    --model "...\models\gpt2-large-cycle3-cust-arc1" ^
    --baseline_file baseline_math.json ^
    --bigram_db bigram_db_combined.json ^
    --engram engram_curve.json ^
    --field bivector_field.npz ^
    --system_prompt system_prompt.txt ^
    --prompt "The cause of altruistic behavior is" ^
    --n_experts 3 --output gem_arc_result.json

  python gem_pipeline_arc.py sweep ^
    --model "...\models\gpt2-large-cycle3-cust-arc1" ^
    --baseline_file baseline_math.json ^
    --bigram_db bigram_db_combined.json ^
    --engram engram_curve.json ^
    --field bivector_field.npz ^
    --system_prompt system_prompt.txt ^
    --prompts prompts.txt ^
    --output gem_arc_sweep.json
"""

import argparse, json, time, sys
import numpy as np
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

K_PROJ   = 16
DELTA_C  = 0.16     # certified quality-maximum synthesis address

# ── Default system prompt ──────────────────────────────────────────────────────
# Provides semantic grounding and task orientation.
# Prevents undirected generation from finding random attractors.
# Can be overridden with --system_prompt file.

DEFAULT_SYSTEM_PROMPT = """You are a precise research assistant with expertise in 
empirical science, mathematics, and structured analytical reasoning. When asked 
a question, you provide well-organized, evidence-based responses that cite 
relevant research, acknowledge uncertainty, and build logical arguments 
step by step. You prefer concrete examples over abstractions, and you 
acknowledge the limits of current knowledge when relevant."""

# ── Ksi correction (unchanged from wandering_model.py) ────────────────────────

def measure_ksi(h, V_top):
    z = V_top.T @ h; z2=z*z; S=float(z2.sum())
    if S < 1e-12: return 0.5, z
    p=z2/S; ps=np.where(p>1e-15,p,1e-15)
    return float(np.clip(-np.sum(p*np.log(ps))/np.log(K_PROJ),0,1)), z

def ksi_grad(z):
    z2=z*z; S=float(z2.sum())
    if S<1e-12: return np.zeros_like(z)
    p=z2/S; ps=np.where(p>1e-15,p,1e-15); H=float(-np.sum(p*np.log(ps)))
    return -(2.*z)/(S*np.log(K_PROJ))*(np.log(ps)+H)

def correct_ksi(h, V, tgt, cur, ss=0.05, ns=3):
    if abs(tgt-cur)<5e-5: return h, cur
    z=V.T@h; hp=V@z; hr=h-hp; pn=float(np.linalg.norm(hp))
    sign=float(np.sign(tgt-cur))
    for _ in range(ns):
        g=ksi_grad(z); gn=float(np.linalg.norm(g))
        if gn<1e-12: break
        z=z+sign*ss*g/gn
    np_=V@z; nn=float(np.linalg.norm(np_))
    if nn>1e-8 and pn>1e-8: z=z*(pn/nn)
    ho=(V@z+hr).astype(np.float32)
    ksi_out, _ = measure_ksi(ho, V)
    return ho, ksi_out

def bezier(baseline, ds, de):
    n=len(baseline); t=np.linspace(0,1,n)
    dp=(ds+de)/2.
    d=(1-t)**2*ds + 2*(1-t)*t*dp + t**2*de
    return (np.array(baseline)+d).tolist()

def extract_svds(model, n_layers):
    svds=[]
    for i in range(n_layers):
        w=model.transformer.h[i].mlp.c_fc.weight.data.float().numpy()
        U,_,_=np.linalg.svd(w,full_matrices=False)
        svds.append(U[:,:K_PROJ].astype(np.float32))
    return svds

# ── Arc utilities (from wandering_model_arc.py) ────────────────────────────────

def load_bigram_db(path):
    if not path or not Path(path).exists(): return None
    with open(path) as f: return json.load(f)

def load_field(path):
    if not path or not Path(path).exists(): return None
    return np.load(path, allow_pickle=True)['field']

def get_arc(tok_id, db, top_k=10):
    if db is None: return 0., 0.
    key=str(tok_id)
    if key not in db["tokens"]: return 0., 0.
    deps=db["tokens"][key]["departures"]
    if not deps: return 0., 0.
    items=sorted(deps.items(),key=lambda x:-x[1]["count"])[:top_k]
    total=sum(v["count"] for _,v in items)
    dj1=sum(v["mean_dj1"]*v["count"] for _,v in items)/(total+1e-12)
    dksi=sum(v["mean_dksi"]*v["count"] for _,v in items)/(total+1e-12)
    return float(dj1), float(dksi)

def query_field(field, j1, ksi):
    if field is None: return 0., 0., 0.
    BINS_J1=36; BINS_KSI=20
    count=field[:,:,2]; safe=np.where(count>0,count,1.)
    mean_dj1=field[:,:,0]/safe; mean_dksi=field[:,:,1]/safe
    js=int(float(j1)%360/10)%BINS_J1; ks=min(int(float(ksi)/0.05),BINS_KSI-1)
    c=float(count[js,ks])
    if c<5: return 0.,0.,0.
    return float(mean_dj1[js,ks]), float(mean_dksi[js,ks]), float(np.tanh(c/100.))

def nudge(h, pc1, pc2, dj1, alpha):
    if abs(dj1)<0.5 or alpha<0.01: return h
    h_n=h/(np.linalg.norm(h)+1e-12)
    a=float(np.dot(h_n,pc1)); b=float(np.dot(h_n,pc2))
    j1=(np.degrees(np.arctan2(a,b)))%360
    j1t=(j1+np.clip(dj1,-40,40))%360
    rt=np.radians(j1t)
    an=(1-alpha)*a+alpha*np.cos(rt); bn=(1-alpha)*b+alpha*np.sin(rt)
    nm=np.sqrt(an**2+bn**2)
    if nm<1e-12: return h
    on=np.sqrt(a**2+b**2)
    hp=a*pc1+b*pc2; hr=h-hp
    return ((an/nm*on)*pc1+(bn/nm*on)*pc2+hr).astype(np.float32)

def h_j1_ksi(h, U_ref, pc1, pc2):
    h_n=h/(np.linalg.norm(h)+1e-12)
    j1=float(np.degrees(np.arctan2(np.dot(h_n,pc1),np.dot(h_n,pc2)))%360)
    proj=U_ref.T@h_n; p=proj**2/(proj**2).sum()+1e-12
    ksi=float(np.clip(-np.sum(p*np.log(p+1e-12))/np.log(K_PROJ),0,1))
    return j1, ksi

# ── Core generation ────────────────────────────────────────────────────────────

def generate_expert(model, tok, prompt, schedule, svds, pc1, pc2, U_ref,
                     db, field, arc_strength=0.10,
                     max_tokens=120, temperature=0.9, top_k=50,
                     verbose=False):
    """
    Single expert generation pass with Ksi correction + arc augmentation.
    Returns text, ksi_pre_mean, ksi_post_mean, arc_confidence_mean.
    """
    n_layers = len(schedule)
    current_tok = [None]
    ksi_pre_list = []; ksi_post_list = []; conf_list = []

    def make_hook(li):
        def fn(mod, args_, out):
            hb=out[0]; h=hb[0,-1,:].float().cpu().numpy().astype(np.float64)
            V=svds[li]; tgt=schedule[li]

            ksi_pre, _ = measure_ksi(h.astype(np.float32), V)
            ksi_pre_list.append(ksi_pre)

            lyr_alpha = arc_strength * max(0., 1. - li/(n_layers*1.5))

            # Combined arc: bigram_db (token-specific) + field (position-based)
            j1_now, ksi_now = h_j1_ksi(h.astype(np.float32), U_ref, pc1, pc2)
            pred_dj1_db, _ = get_arc(current_tok[0], db) if current_tok[0] else (0., 0.)
            pred_dj1_fld, _, conf = query_field(field, j1_now, ksi_now)
            conf_list.append(conf)

            # Blend: field dominates if high confidence, db otherwise
            if conf > 0.3:
                pred_dj1 = 0.6 * pred_dj1_fld + 0.4 * pred_dj1_db
            else:
                pred_dj1 = pred_dj1_db

            if abs(pred_dj1) > 0.5 and lyr_alpha > 0.01:
                h = nudge(h.astype(np.float32),
                          pc1.astype(np.float32), pc2.astype(np.float32),
                          pred_dj1, lyr_alpha).astype(np.float64)

            h_new, ksi_post = correct_ksi(h.astype(np.float32), V, tgt, ksi_pre)
            ksi_post_list.append(ksi_post)

            ht=torch.tensor(h_new, dtype=hb.dtype, device=hb.device)
            hb=hb.clone(); hb[0,-1,:]=ht
            return (hb,)+out[1:] if isinstance(out,tuple) else hb
        return fn

    hooks=[model.transformer.h[i].register_forward_hook(make_hook(i))
           for i in range(n_layers)]

    inp=tok(prompt, return_tensors="pt")["input_ids"]
    current_tok[0]=int(inp[0,-1])
    gen=inp.clone(); n_tok=0

    with torch.no_grad():
        for _ in range(max_tokens):
            out=model(gen)
            logits=out.logits[0,-1,:]/temperature
            if top_k>0:
                v,_=torch.topk(logits,top_k)
                logits[logits<v[-1]]=float("-inf")
            probs=torch.softmax(logits,dim=-1)
            nt=torch.multinomial(probs,1).unsqueeze(0)
            gen=torch.cat([gen,nt],dim=1)
            current_tok[0]=int(nt[0,0])
            n_tok+=1
            if current_tok[0]==tok.eos_token_id: break

    for h in hooks: h.remove()

    text=tok.decode(gen[0],skip_special_tokens=True)
    kpi=float(np.mean(ksi_pre_list)) if ksi_pre_list else 0.
    kpo=float(np.mean(ksi_post_list)) if ksi_post_list else 0.
    conf_mean=float(np.mean(conf_list)) if conf_list else 0.
    return text, kpi, kpo, kpo-kpi, conf_mean

# ── Build synthesis prompt ─────────────────────────────────────────────────────

def build_synthesis_prompt(system_prompt, user_prompt, expert_results):
    """
    Build the prompt for the synthesis agent.
    Includes all expert outputs and their Ksi lifts.
    """
    parts = [system_prompt.strip(), "\n\n"]
    parts.append(f"QUESTION: {user_prompt}\n\n")
    parts.append("EXPERT ANALYSES:\n")
    for i, (text, kpi, kpo, lift, conf) in enumerate(expert_results):
        # Strip system prompt prefix from expert output
        expert_text = text
        if user_prompt in expert_text:
            expert_text = expert_text[expert_text.find(user_prompt)+len(user_prompt):].strip()
        parts.append(f"\n[Expert {i+1} | Ksi lift={lift:+.3f} | conf={conf:.2f}]:\n")
        parts.append(expert_text[:400].strip())   # cap at 400 chars per expert
        parts.append("\n")
    parts.append(f"\nSYNTHESIS: Integrating the above analyses, ")
    return "".join(parts)

# ── Subcommand: run ────────────────────────────────────────────────────────────

def cmd_run(args):
    print("\n" + "="*70)
    print("GEM PIPELINE — Arc-Augmented with System Prompt")
    print("="*70)

    print(f"\nLoading model: {args.model}", flush=True)
    model=GPT2LMHeadModel.from_pretrained(args.model,local_files_only=True,
                                           low_cpu_mem_usage=True)
    model.eval()
    tok=GPT2Tokenizer.from_pretrained(args.model,local_files_only=True)
    tok.pad_token=tok.eos_token
    n_layers=model.config.n_layer
    print("  Extracting SVDs...", flush=True)
    svds=extract_svds(model, n_layers)

    with open(args.baseline_file) as f: bl=json.load(f)
    baseline=bl["baseline"]
    db=load_bigram_db(getattr(args,'bigram_db',None))
    field=load_field(getattr(args,'field',None))

    with open(args.engram) as f: e=json.load(f)
    pc1=np.array(e["pc1"],dtype=np.float32)
    pc2=np.array(e["pc2"],dtype=np.float32)
    U_ref=np.array(e["U_ref"],dtype=np.float32)

    # System prompt
    sys_prompt_path = getattr(args,'system_prompt',None)
    if sys_prompt_path and Path(sys_prompt_path).exists():
        sys_prompt = Path(sys_prompt_path).read_text(encoding='utf-8').strip()
        print(f"  System prompt: {sys_prompt_path} ({len(sys_prompt)} chars)")
    else:
        sys_prompt = DEFAULT_SYSTEM_PROMPT
        print(f"  System prompt: default ({len(sys_prompt)} chars)")

    n_experts = getattr(args,'n_experts',3)
    arc_strength = getattr(args,'arc_strength',0.10)
    max_tokens = getattr(args,'max_tokens',120)

    # Expert delta schedule
    if n_experts == 3:
        expert_deltas = [-0.04, 0.04, 0.10]
    elif n_experts == 5:
        expert_deltas = [-0.04, 0.0, 0.04, 0.08, 0.12]
    else:
        expert_deltas = [round(-0.04 + i * 0.16/(n_experts-1), 3)
                         for i in range(n_experts)]

    user_prompt = args.prompt
    full_prompt = f"{sys_prompt}\n\n{user_prompt}"

    print(f"\n  Prompt: '{user_prompt}'")
    print(f"  n_experts: {n_experts}, deltas: {expert_deltas}")
    print(f"  synthesis delta_c: {DELTA_C}")
    print(f"  arc_strength: {arc_strength}")
    print()

    # ── Run experts ──
    expert_results = []
    expert_texts   = []
    for i, delta in enumerate(expert_deltas):
        print(f"  [Expert {i+1}/{n_experts}] delta={delta:+.3f}", flush=True)
        schedule = bezier(baseline, -0.02, delta)
        text, kpi, kpo, lift, conf = generate_expert(
            model, tok, full_prompt, schedule, svds, pc1, pc2, U_ref,
            db, field, arc_strength=arc_strength,
            max_tokens=max_tokens, temperature=0.9)
        expert_results.append((text, kpi, kpo, lift, conf))
        # Strip prefix from display
        disp = text[text.find(user_prompt)+len(user_prompt):].strip()[:200] if user_prompt in text else text[:200]
        print(f"    Ksi: {kpi:.4f} → {kpo:.4f} (lift={lift:+.4f})  conf={conf:.3f}")
        print(f"    {disp[:160]}")
        print()
        expert_texts.append(disp)

    mean_expert_lift = float(np.mean([r[3] for r in expert_results]))

    # ── Jaccard similarity between experts (H-GEM-003) ──
    token_sets = [set(t.split()) for t in expert_texts]
    if len(token_sets) >= 2:
        jaccard_pairs = []
        for i in range(len(token_sets)):
            for j in range(i+1, len(token_sets)):
                inter = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                jaccard_pairs.append(inter/(union+1e-12))
        mean_jaccard = float(np.mean(jaccard_pairs))
        h003 = bool(mean_jaccard < 0.4)
    else:
        mean_jaccard = 0.0; h003 = False

    # ── Run synthesis ──
    print(f"  [Synthesis] delta_c={DELTA_C:+.3f}", flush=True)
    synth_prompt = build_synthesis_prompt(sys_prompt, user_prompt, expert_results)
    schedule_s = bezier(baseline, -0.02, DELTA_C)
    s_text, s_kpi, s_kpo, s_lift, s_conf = generate_expert(
        model, tok, synth_prompt, schedule_s, svds, pc1, pc2, U_ref,
        db, field, arc_strength=arc_strength,
        max_tokens=max_tokens, temperature=0.85)   # slightly lower temp for synthesis

    # Extract synthesis output (after the "SYNTHESIS:" marker)
    synth_output = s_text
    if "SYNTHESIS:" in s_text:
        synth_output = s_text[s_text.rfind("SYNTHESIS:"):]

    dividend = s_lift / (mean_expert_lift + 1e-12)
    h001 = bool(dividend > 1.1)

    print(f"    Ksi: {s_kpi:.4f} → {s_kpo:.4f} (lift={s_lift:+.4f})  conf={s_conf:.3f}")
    print(f"    Dividend: {dividend:.3f}x  H-GEM-001: {'PASS' if h001 else 'FAIL'}")
    print()
    print("  SYNTHESIS OUTPUT:")
    print(f"  {synth_output[:600]}")

    result = {
        "prompt": user_prompt,
        "system_prompt_chars": len(sys_prompt),
        "n_experts": n_experts,
        "expert_deltas": expert_deltas,
        "arc_strength": float(arc_strength),
        "experts": [
            {"delta": float(d), "ksi_pre": float(kpi), "ksi_post": float(kpo),
             "lift": float(lift), "arc_confidence": float(conf),
             "text_snippet": et[:300]}
            for (_, kpi, kpo, lift, conf), et, d
            in zip(expert_results, expert_texts, expert_deltas)
        ],
        "mean_expert_lift": float(mean_expert_lift),
        "synthesis": {
            "delta_c": float(DELTA_C),
            "ksi_pre": float(s_kpi), "ksi_post": float(s_kpo),
            "lift": float(s_lift), "arc_confidence": float(s_conf),
            "dividend": float(dividend),
            "text": synth_output,
        },
        "verdicts": {
            "H_GEM_001_dividend": h001,
            "H_GEM_003_diversity": h003,
            "mean_jaccard": float(mean_jaccard),
        }
    }

    with open(args.output,'w') as f:
        json.dump(result, f, indent=2, default=lambda o:
            float(o) if isinstance(o,(np.float64,np.float32)) else
            int(o) if isinstance(o,(np.int64,np.int32,np.intp)) else
            bool(o) if isinstance(o,(np.bool_,bool)) else None)
    print(f"\n  Saved → {args.output}")
    return result

# ── Subcommand: sweep ──────────────────────────────────────────────────────────

def cmd_sweep(args):
    """Run multiple prompts, build comparison table."""
    print("\n" + "="*70)
    print("GEM SWEEP — Multiple Prompts")
    print("="*70)

    prompts_path = getattr(args,'prompts',None)
    if prompts_path and Path(prompts_path).exists():
        prompts = [l.strip() for l in Path(prompts_path).read_text().split('\n')
                   if l.strip() and not l.startswith('#')]
    else:
        prompts = [
            "The cause of altruistic behavior is",
            "The relationship between mathematical constants and information",
            "Share a mathematical proof of incompleteness and how it builds complexity please.",
            "Share how the fractal calculations are producing words in your model weights.",
        ]

    all_results = []
    for prompt in prompts:
        print(f"\n{'─'*60}")
        print(f"PROMPT: {prompt}")
        args.prompt = prompt
        args.output = f"gem_sweep_{len(all_results)}.json"
        r = cmd_run(args)
        all_results.append(r)

    print("\n" + "="*70)
    print("SWEEP SUMMARY")
    print(f"  {'Prompt':<45} {'divid':>7} {'H001':>6} {'H003':>6}")
    print("  " + "-"*65)
    for r in all_results:
        prompt_short = r['prompt'][:43]
        print(f"  {prompt_short:<45} {r['synthesis']['dividend']:>7.3f} "
              f"{'PASS' if r['verdicts']['H_GEM_001_dividend'] else 'FAIL':>6} "
              f"{'PASS' if r['verdicts']['H_GEM_003_diversity'] else 'FAIL':>6}")

    with open(args.output,'w') as f:
        json.dump({"results": all_results}, f, indent=2, default=lambda o:
            float(o) if isinstance(o,(np.float64,np.float32)) else
            int(o) if isinstance(o,(np.int64,np.int32,np.intp)) else
            bool(o) if isinstance(o,(np.bool_,bool)) else None)
    print(f"\n  Sweep saved → {args.output}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GeM Pipeline — Arc-Augmented")
    ap.add_argument("cmd", choices=["run","sweep"])
    ap.add_argument("--model",          required=True)
    ap.add_argument("--baseline_file",  required=True)
    ap.add_argument("--engram",         required=True)
    ap.add_argument("--bigram_db",      default=None)
    ap.add_argument("--field",          default=None)
    ap.add_argument("--system_prompt",  default=None)
    ap.add_argument("--prompt",         default="The cause of altruistic behavior is")
    ap.add_argument("--prompts",        default=None)
    ap.add_argument("--n_experts",      type=int,   default=3)
    ap.add_argument("--arc_strength",   type=float, default=0.10)
    ap.add_argument("--max_tokens",     type=int,   default=120)
    ap.add_argument("--output",         default="gem_arc_result.json")

    args = ap.parse_args()
    {"run": cmd_run, "sweep": cmd_sweep}[args.cmd](args)

if __name__ == "__main__":
    main()
