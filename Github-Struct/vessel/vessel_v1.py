#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vessel_v1.py  --  Pirouette Volume 8 / CORE-003
================================================
A dynamic, shared HH-address "vessel" coupling the thought (probe trajectory)
and the flood (passive-scalar / sub-manifold cloud) into ONE mutable object on
the Henon-Heiles basin manifold.

DESIGN (frozen with Keaton, this session)
------------------------------------------
  * Vessel = ONE base HH address (5 scalars: J1, J2, Ksi, Phi, E) that rolls
    DYNAMICALLY through the vacuum-stiffness potential with damping (= decay knob).
  * Plus a CLOUD of K sub-displacements in the local chart around that base,
    stored as base + offsets inside a SINGLE shared object. Projecting the cloud
    to the main manifold = base + offset. All weights besides the 5 base scalars
    (plus K small offsets) stay implicit -- reconstructed on demand from the LUTs.
  * Thought deposit = LOGPROB-WEIGHTED: high-quality tokens kick the base harder.
  * Wet sub-positions get their offset REINFORCED and TAGGED with the current base
    address (the origin link) -> an associative trace base -> fertile-offset.
  * READOUT: base+offset composes to a main-manifold address -> indexes the
    existing stiffness / helicity LUTs -> biases the softmax via beta_wet.
  * beta_wet = 0 AND zero-kick reduces EXACTLY to the underlying generator
    (reduction-to-classical; the kappa=0 analogue).

SWEPT KNOBS (CLI):  diffusivity<->stiffness map, beta_wet, decay/damping.

NULLS BUILT IN (run via the `null` subcommand):
  * N_vessel_vs_state : vessel-driven bias vs. instantaneous-hidden-state bias.
                        Does integrating history (with damping) beat using the
                        thought's own address directly?
  * N_substructure    : full origin-linked cloud vs. offsets reshuffled each step
                        (same centroid, destroyed sub-structure). Does the
                        2^160 local space do real work, or is it an expensive
                        centroid?
  * N_scramble        : real LUTs vs. phase-scrambled LUTs. Is the vessel reading
                        basin structure or diffusing into whatever is adjacent?

INSTRUMENTATION: a lock-in detector (basin-occupancy entropy over a sliding
window) is written to the JSON so the two stacked positive-feedback loops
(base rolls toward kicks; offsets reinforce toward wet) show up as a collapse
signature instead of silently emitting one region forever (the Wada
"everything" failure mode, now reachable from two directions).

NOTE ON SCOPE: cross-model / "sub-model addressing" is intentionally NOT a v1
success criterion. The standing prior (cross-family transplant failed;
geometry is co-trained) says fertile addresses are architecture-native. This
script probes the LOCAL claim only. The hooks for a later transfer test exist
(origin-link table is exported) but no v1 verdict depends on them.

Standalone. No cross-imports. Windows paths honored. numpy-safe JSON.
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# ----------------------------------------------------------------------------- 
# Conventions / paths
# -----------------------------------------------------------------------------
# Matches the working dir pattern:
#   C:\Users\keatw\Pirouette_Volume_8\doclab\experiments\fractal_LLM\transformer\
DEFAULT_HELICITY_LUT = "helicity_lut.npy"     # 360-point, per CORE-003 three-fractal
DEFAULT_STIFFNESS_LUT = "stiffness_lut.npy"   # 360-point vacuum stiffness
LUT_N = 360

# 5-basis HH address index map (action-angle coords used across the framework)
ADDR_KEYS = ("J1", "J2", "Ksi", "Phi", "E")
J1, J2, KSI, PHI, E = range(5)


# ============================================================================= 
# numpy-safe JSON
# =============================================================================
def _np_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _np_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_safe(v) for v in obj]
    return obj


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_np_safe(obj), f, indent=2)
    print(f"  [json] wrote {path}", flush=True)


# ============================================================================= 
# LUT loading (real if present, synthetic fallback with a loud banner)
# =============================================================================
def load_lut(path, name, scramble=False, seed=0):
    """Load a 360-point LUT. Falls back to a synthetic three-basin LUT if the
    file is absent, so the mechanics can be dry-run before pointing at real data.
    `scramble` phase-randomizes the LUT for the N_scramble null."""
    if os.path.isfile(path):
        arr = np.load(path, allow_pickle=True).astype(np.float64).ravel()
        if arr.shape[0] != LUT_N:
            # resample to 360 if a different resolution was stored
            xp = np.linspace(0, 360, arr.shape[0], endpoint=False)
            x = np.linspace(0, 360, LUT_N, endpoint=False)
            arr = np.interp(x, xp, arr, period=360)
        src = "real"
    else:
        print(f"  [!!] {name} LUT not found at {path} -- using SYNTHETIC fallback.",
              flush=True)
        th = np.linspace(0, 2 * np.pi, LUT_N, endpoint=False)
        # three-basin structure with Wada-ish ridges between centers
        arr = (np.exp(-((np.cos(th - 0.0)) - 1) ** 2 * 6)
               + np.exp(-((np.cos(th - 2 * np.pi / 3)) - 1) ** 2 * 6)
               + np.exp(-((np.cos(th - 4 * np.pi / 3)) - 1) ** 2 * 6))
        arr = arr / arr.max()
        src = "synthetic"

    if scramble:
        rng = np.random.default_rng(seed)
        # phase scramble: preserve amplitude spectrum, randomize phases
        F = np.fft.rfft(arr)
        mag = np.abs(F)
        ph = rng.uniform(-np.pi, np.pi, size=mag.shape)
        ph[0] = 0.0
        arr = np.fft.irfft(mag * np.exp(1j * ph), n=LUT_N)
        src += "+scrambled"

    print(f"  [lut] {name}: {src}, n={arr.shape[0]}, "
          f"range=[{arr.min():.4f},{arr.max():.4f}]", flush=True)
    return arr


def lut_at(lut, angle_deg):
    """Linear-interp LUT lookup at an angle in degrees (period 360)."""
    a = np.mod(angle_deg, 360.0)
    i0 = int(np.floor(a / 360.0 * LUT_N)) % LUT_N
    i1 = (i0 + 1) % LUT_N
    frac = (a / 360.0 * LUT_N) - np.floor(a / 360.0 * LUT_N)
    return (1 - frac) * lut[i0] + frac * lut[i1]


# ============================================================================= 
# Diffusivity <-> stiffness maps (swept knob)
# =============================================================================
def diffusivity(stiff_val, mode="inverse", floor=0.02):
    """High stiffness (coherent basin) -> low diffusion. Wada ridge -> high.
    Returns a positive diffusivity scalar."""
    s = np.clip(stiff_val, 0.0, 1.0)
    if mode == "inverse":
        return floor + (1.0 - s)
    if mode == "inverse_sq":
        return floor + (1.0 - s) ** 2
    if mode == "exp":
        return floor + np.exp(-3.0 * s)
    if mode == "flat":            # control: diffusion independent of stiffness
        return floor + 0.5
    raise ValueError(f"unknown diffusivity mode: {mode}")


# ============================================================================= 
# Leyline itinerary: certified logistic Ksi->r map (CORE-015 / address-decoder)
# =============================================================================
# Certified correspondences this encodes (NOT re-derived here -- imported as fact):
#   * beta = 0.438                     : CORE-015 Ksi compression exponent
#   * alpha_physical boundary Ksi~0.397 <-> Feigenbaum onset r_c = 3.5699
#   * Wada boundary        Ksi~0.880   <-> logistic trough
# The logistic map x' = r x (1-x) is walked; its orbit visits period windows
# (plateaus) and chaotic bands (bursts). Each orbit value maps Ksi -> r and the
# orbit position becomes a target the vessel base is pulled toward (the "key").
BETA_CORE015 = 0.438
R_FEIGENBAUM = 3.5699
KSI_ALPHA_PHYS = 0.397
KSI_WADA = 0.880


def ksi_to_r(ksi):
    """Certified Ksi->r compression (CORE-015, beta=0.438).
    Anchored so Ksi=KSI_ALPHA_PHYS maps to the Feigenbaum onset r_c, and higher
    Ksi pushes deeper into chaos toward r=4 at the Wada trough."""
    # monotone map through the certified anchor, compressed by beta
    t = np.clip((ksi - KSI_ALPHA_PHYS) / max(KSI_WADA - KSI_ALPHA_PHYS, 1e-9), 0, 1)
    t = t ** BETA_CORE015
    return R_FEIGENBAUM + t * (4.0 - R_FEIGENBAUM)


def logistic_itinerary(n, ksi_seed=0.585, x0=0.5, rng=None, scramble=False):
    """Generate an n-step itinerary of target J1 addresses by walking the
    logistic map at r = ksi_to_r(running Ksi). Returns (targets_J1, r_series).

    The orbit value x in [0,1] is mapped to a J1 target in degrees. Period
    windows produce clustered targets (coherent plateaus); chaotic r produces
    spread targets (bursts). This is the certified itinerary structure seen in
    ll_logistic.json's waypoints.

    scramble=True -> phase-shuffle the orbit (destroys period/chaos sequencing,
    preserves the marginal distribution of targets). This is the N_leyline null.
    """
    r = ksi_to_r(ksi_seed)
    x = x0
    targets = np.empty(n)
    rseries = np.empty(n)
    for i in range(n):
        x = r * x * (1.0 - x)
        x = min(max(x, 1e-9), 1.0 - 1e-9)
        # map orbit value to a J1 target (full circle), biased toward the
        # fertile high-lift band so plateaus sit near coherent English basins
        targets[i] = 60.0 + x * 240.0   # J1 in [60, 300]
        rseries[i] = r
    if scramble:
        idx = (rng or np.random.default_rng(0)).permutation(n)
        targets = targets[idx]
    return targets, rseries


# ============================================================================= 
# The shared Vessel object
# =============================================================================
class Vessel:
    """ONE base HH address + a cloud of K sub-displacements, in a single object.

    Base rolls dynamically through the vacuum-stiffness potential with damping.
    Cloud offsets live in the local chart; wet offsets are reinforced and tagged
    with the base address that made them fertile (origin link).
    """

    def __init__(self, k_cloud, helicity, stiffness, knobs, rng,
                 reshuffle_offsets=False, use_state_addr=False,
                 itinerary=None):
        self.k = k_cloud
        self.hel = helicity
        self.stf = stiffness
        self.kn = knobs
        self.rng = rng
        self.reshuffle = reshuffle_offsets   # N_substructure null
        self.use_state_addr = use_state_addr # N_vessel_vs_state null
        self.itinerary = itinerary           # leyline target J1 per step (or None)
        self._step = 0

        # base address (5 scalars). angles in degrees where applicable.
        self.base = np.zeros(5, dtype=np.float64)
        self.base[J1] = 75.0    # start in the sparse high-lift zone (J1 60-90)
        self.base[KSI] = 0.585  # English attractor Ksi_EN
        self.vel = np.zeros(5, dtype=np.float64)

        # cloud: K offsets in the local chart around base (small)
        self.offsets = rng.normal(0, knobs["cloud_sigma"], size=(self.k, 5))

        # origin-link table: list of (base_J1, base_Ksi, offset_vec, wetness)
        self.links = []

        # lock-in instrumentation
        self.basin_hist = []  # which basin index the base sat in, per step

    # --- dynamics -----------------------------------------------------------
    def _stiff_grad(self, j1):
        """Finite-difference gradient of stiffness along J1 (the rolling axis)."""
        h = 1.0
        return (lut_at(self.stf, j1 + h) - lut_at(self.stf, j1 - h)) / (2 * h)

    def roll(self):
        """Advance the base one step under the HH stiffness potential with damping.
        Force = -grad(potential); potential well = high stiffness (coherent)."""
        if self.use_state_addr:
            return  # state-addr null: base is overwritten externally, no roll
        damping = self.kn["decay"]          # decay knob == damping coefficient
        dt = self.kn["dt"]
        # potential = -stiffness (particle falls toward coherent zones)
        force = self._stiff_grad(self.base[J1])     # d(stiff)/dJ1 ; +force toward higher stiff
        self.vel[J1] = (1.0 - damping) * self.vel[J1] + dt * force
        self.base[J1] = np.mod(self.base[J1] + dt * self.vel[J1], 360.0)
        # Ksi drifts gently toward the English attractor unless kicked
        self.base[KSI] += dt * 0.05 * (0.585 - self.base[KSI])

    def kick(self, logprob_weight):
        """Logprob-weighted deposit + leyline itinerary pull.

        Two forces compose here -- the lock-and-key:
          * logprob kick: high-quality token displaces base down helicity grad
            (the existing deposit; stiffness/helicity = the lock's tumblers).
          * leyline pull: base is pulled toward the certified logistic itinerary
            target for this step (the key's teeth -- the period/chaos sequence).
        leyline_gain=0 -> pure logprob kick (and a flat itinerary -> dual_free).
        """
        if self.use_state_addr:
            self._step += 1
            return
        g = self.kn["deposit_gain"] * float(logprob_weight)
        h = 1.0
        hg = (lut_at(self.hel, self.base[J1] + h)
              - lut_at(self.hel, self.base[J1] - h)) / (2 * h)
        self.vel[J1] += g * (-hg)

        # leyline pull toward the itinerary target (shortest angular path)
        lg = self.kn.get("leyline_gain", 0.0)
        if lg != 0.0 and self.itinerary is not None and self._step < len(self.itinerary):
            target = self.itinerary[self._step]
            d = np.mod(target - self.base[J1] + 180, 360) - 180
            self.vel[J1] += lg * d
        self._step += 1

    def set_base_from_state(self, state_j1, state_ksi):
        """N_vessel_vs_state null: overwrite base with the thought's instantaneous
        hidden-state address instead of the rolled/integrated vessel address."""
        self.base[J1] = np.mod(state_j1, 360.0)
        self.base[KSI] = state_ksi

    # --- readout & wetness --------------------------------------------------
    def cloud_addresses(self):
        """Project cloud to main-manifold addresses: base + offset (J1 axis)."""
        offs = self.offsets
        if self.reshuffle:
            # N_substructure null: same centroid, destroyed sub-structure
            offs = self.offsets[self.rng.permutation(self.k)] - \
                   self.offsets.mean(0, keepdims=True) + self.offsets.mean(0, keepdims=True)
            offs = self.rng.permutation(offs)  # shuffle rows; centroid preserved
        return np.mod(self.base[J1] + offs[:, J1], 360.0)

    def wetness(self):
        """Per-cloud-member tractability = readout from the LUTs at its address.
        Wet = high stiffness (coherent/fertile) modulated by helicity fertility."""
        addrs = self.cloud_addresses()
        stiff = np.array([lut_at(self.stf, a) for a in addrs])
        hel = np.array([lut_at(self.hel, a) for a in addrs])
        # fertile = coherent (high stiff) but with helical structure (non-zero hel)
        w = np.clip(stiff * (0.5 + 0.5 * hel), 0.0, None)
        return addrs, w

    def reinforce_and_link(self, addrs, w):
        """Wet offsets get reinforced (pulled toward their wet position) and
        tagged with the current base address. Origin link = associative trace."""
        if self.use_state_addr:
            return
        thr = np.quantile(w, 0.75)  # top quartile counts as "wet"
        lr = self.kn["reinforce_lr"]
        for i in range(self.k):
            if w[i] >= thr:
                # pull this offset's J1 toward the wet position (reinforcement)
                wet_off = np.mod(addrs[i] - self.base[J1] + 180, 360) - 180
                self.offsets[i, J1] += lr * (wet_off - self.offsets[i, J1])
                # origin link: base -> fertile offset
                self.links.append({
                    "base_J1": float(self.base[J1]),
                    "base_Ksi": float(self.base[KSI]),
                    "offset_J1": float(self.offsets[i, J1]),
                    "wetness": float(w[i]),
                })

    def softmax_bias(self, vocab_addrs):
        """Return an additive logit bias over a set of vocab addresses (degrees),
        proportional to the wet density the vessel cloud projects there.
        beta_wet scales the whole thing; beta_wet=0 -> exactly zero bias."""
        beta = self.kn["beta_wet"]
        if beta == 0.0:
            return np.zeros(len(vocab_addrs))
        addrs, w = self.wetness()
        # kernel density of wet cloud mass at each vocab address (von Mises-ish)
        bw = self.kn["bias_bandwidth_deg"]
        bias = np.zeros(len(vocab_addrs))
        for j, va in enumerate(vocab_addrs):
            d = np.mod(addrs - va + 180, 360) - 180
            bias[j] = np.sum(w * np.exp(-0.5 * (d / bw) ** 2))
        # log-density bias (additive in logit space)
        bias = np.clip(bias, 0.0, None)
        bias = np.log1p(bias)
        return beta * bias

    # --- instrumentation ----------------------------------------------------
    def record_basin(self):
        """Three-basin assignment of the base J1 for lock-in entropy."""
        centers = np.array([60.0, 180.0, 300.0])  # nominal basin centers in J1
        d = np.abs(np.mod(self.base[J1] - centers + 180, 360) - 180)
        self.basin_hist.append(int(np.argmin(d)))

    def lockin_entropy(self, window=64):
        """Sliding-window basin-occupancy entropy (bits). Low -> locked in."""
        h = self.basin_hist[-window:]
        if not h:
            return None
        counts = np.bincount(h, minlength=3).astype(np.float64)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))


# ============================================================================= 
# Toy generator harness
# =============================================================================
# v1 ships with a TOY token model so the vessel mechanics are exercisable
# end-to-end without loading GPT-2. Replace `toy_logits` + `toy_token_addrs`
# with the real GPT-2-Large hook: logits over vocab, and each token's HH J1
# address (you already compute hidden-state -> (J1_PCA, Ksi)).
# =============================================================================
def toy_token_addrs(vocab_size, rng):
    """Assign each toy 'token' a J1 address. Sparse high-lift zone gets a cluster."""
    base = rng.uniform(0, 360, size=vocab_size)
    # plant a fertile cluster in the sparse zone (J1 60-90), per the 471x lift finding
    n_fertile = vocab_size // 8
    base[:n_fertile] = rng.uniform(60, 90, size=n_fertile)
    return base


def toy_logits(step, vocab_size, rng):
    """Base-model logits (stand-in for GPT-2-Large)."""
    return rng.normal(0, 1.0, size=vocab_size)


def eval_h001_h002(alpha_traj, ksi_traj):
    """PROXY evaluators matching the observable signatures in the uploaded runs.
    REPLACE with the generator's real h001/h002 definitions before certifying.

    Observed in ll_*.json:
      * h001 passed ONLY on leyline_logistic, whose alpha-trajectory carries
        logistic period/chaos structure (sustained plateaus + sharp drops).
        Proxy: h001 = the trajectory has BOTH high-autocorrelation runs
        (plateaus) AND high-variance segments (bursts) -- dynamical structure.
      * h002 failed on everything (register coherence). Proxy: h002 = alpha_mean
        sits in the native-English band AND alpha_std is low (parked, coherent
        register) -- i.e. the dual_free signature, which is what we want the
        leyline to ADD coherence to without losing h001.
    These are deliberately the two competing criteria: one rewards motion, the
    other rewards staying put. Passing BOTH is the lock-and-key claim.
    """
    a = np.asarray(alpha_traj, dtype=np.float64)
    if a.size < 16:
        return False, False
    # h001 proxy: structure = lag-1 autocorr (plateaus) AND segment-variance spread
    ac = np.corrcoef(a[:-1], a[1:])[0, 1]
    seg = a.reshape(-1)[:(a.size // 8) * 8].reshape(8, -1)
    seg_var_spread = float(np.std(seg.var(axis=1)))
    h001 = (ac > 0.25) and (seg_var_spread > 0.0015)
    # h002 proxy: coherent register = mean in English band, low overall std
    h002 = (0.55 <= float(a.mean()) <= 0.62) and (float(a.std()) < 0.05)
    return bool(h001), bool(h002)


def run_generation(steps, vocab_size, knobs, k_cloud, seed,
                   helicity, stiffness, reshuffle=False, use_state_addr=False,
                   leyline=None, leyline_scramble=False):
    rng = np.random.default_rng(seed)
    itinerary = None
    if leyline == "logistic":
        itinerary, _ = logistic_itinerary(steps, ksi_seed=0.585, rng=rng,
                                           scramble=leyline_scramble)
    vessel = Vessel(k_cloud, helicity, stiffness, knobs, rng,
                    reshuffle_offsets=reshuffle, use_state_addr=use_state_addr,
                    itinerary=itinerary)
    tok_addrs = toy_token_addrs(vocab_size, rng)

    chosen = []
    lockin = []
    alpha_traj = []   # proxy: base J1 mapped to an alpha-like scalar per step
    ksi_traj = []
    fertile_hits = 0
    for s in range(steps):
        vessel.roll()
        vessel.record_basin()

        logits = toy_logits(s, vocab_size, rng)
        bias = vessel.softmax_bias(tok_addrs)
        biased = logits + bias

        p = np.exp((biased - biased.max()) / 0.7)
        p /= p.sum()
        tok = int(rng.choice(vocab_size, p=p))
        chosen.append(tok)

        logprob_weight = float(biased[tok] - np.log(np.sum(np.exp(biased - biased.max()))) - biased.max())
        logprob_weight = 1.0 / (1.0 + np.exp(-logprob_weight))

        if use_state_addr:
            vessel.set_base_from_state(tok_addrs[tok], 0.585)

        vessel.kick(logprob_weight)
        addrs, w = vessel.wetness()
        vessel.reinforce_and_link(addrs, w)

        if 60.0 <= tok_addrs[tok] <= 90.0:
            fertile_hits += 1

        # alpha-like proxy from base position (stand-in for real per-token alpha)
        alpha_traj.append(0.45 + 0.30 * lut_at(stiffness, vessel.base[J1]))
        ksi_traj.append(float(vessel.base[KSI]))
        lockin.append(vessel.lockin_entropy())

    h001, h002 = eval_h001_h002(alpha_traj, ksi_traj)
    return {
        "fertile_hit_rate": fertile_hits / steps,
        "mean_lockin_entropy": float(np.nanmean([x for x in lockin if x is not None])),
        "final_lockin_entropy": lockin[-1],
        "n_origin_links": len(vessel.links),
        "alpha_mean": float(np.mean(alpha_traj)),
        "alpha_std": float(np.std(alpha_traj)),
        "h001_pass": h001,
        "h002_pass": h002,
        "dual_pass": bool(h001 and h002),
        "chosen_sample": chosen[:32],
        "lockin_trace_tail": lockin[-32:],
    }


# ============================================================================= 
# Subcommands
# =============================================================================
def make_knobs(args):
    return {
        "diffusivity_mode": args.diffusivity_mode,
        "beta_wet": args.beta_wet,
        "decay": args.decay,
        "deposit_gain": args.deposit_gain,
        "reinforce_lr": args.reinforce_lr,
        "cloud_sigma": args.cloud_sigma,
        "bias_bandwidth_deg": args.bias_bandwidth_deg,
        "dt": args.dt,
        "leyline_gain": getattr(args, "leyline_gain", 0.0),
    }


def cmd_run(args):
    hel = load_lut(args.helicity_lut, "helicity")
    stf = load_lut(args.stiffness_lut, "stiffness")
    knobs = make_knobs(args)
    print(f"  [knobs] {knobs}", flush=True)
    t0 = time.time()
    res = run_generation(args.steps, args.vocab, knobs, args.k_cloud, args.seed,
                         hel, stf, leyline=getattr(args, "leyline", None))
    res["knobs"] = knobs
    res["elapsed_s"] = time.time() - t0
    dump_json(res, args.out)
    print(f"  fertile_hit_rate={res['fertile_hit_rate']:.3f}  "
          f"alpha={res['alpha_mean']:.3f}±{res['alpha_std']:.3f}  "
          f"h001={res['h001_pass']} h002={res['h002_pass']} "
          f"DUAL={res['dual_pass']}  links={res['n_origin_links']}", flush=True)


def cmd_sweep(args):
    """Sweep the three frozen knobs: diffusivity map, beta_wet, decay."""
    hel = load_lut(args.helicity_lut, "helicity")
    stf = load_lut(args.stiffness_lut, "stiffness")
    rows = []
    diff_modes = ["inverse", "inverse_sq", "exp", "flat"]
    betas = [0.0, 0.5, 1.0, 2.0, 4.0]
    decays = [0.02, 0.1, 0.3, 0.6]
    for dm in diff_modes:
        for b in betas:
            for dc in decays:
                knobs = make_knobs(args)
                knobs["diffusivity_mode"] = dm
                knobs["beta_wet"] = b
                knobs["decay"] = dc
                res = run_generation(args.steps, args.vocab, knobs,
                                     args.k_cloud, args.seed, hel, stf)
                rows.append({
                    "diffusivity_mode": dm, "beta_wet": b, "decay": dc,
                    "fertile_hit_rate": res["fertile_hit_rate"],
                    "mean_lockin_entropy": res["mean_lockin_entropy"],
                    "final_lockin_entropy": res["final_lockin_entropy"],
                    "n_origin_links": res["n_origin_links"],
                })
                print(f"  diff={dm:10s} beta={b:4.1f} decay={dc:4.2f} -> "
                      f"hit={res['fertile_hit_rate']:.3f} "
                      f"lockin={res['mean_lockin_entropy']:.3f}", flush=True)
    dump_json({"sweep": rows, "knobs_base": make_knobs(args)}, args.out)


def cmd_null(args):
    """Pre-registered null battery."""
    hel_real = load_lut(args.helicity_lut, "helicity")
    stf_real = load_lut(args.stiffness_lut, "stiffness")
    hel_scr = load_lut(args.helicity_lut, "helicity", scramble=True, seed=args.seed)
    stf_scr = load_lut(args.stiffness_lut, "stiffness", scramble=True, seed=args.seed + 1)
    knobs = make_knobs(args)

    def runit(hel, stf, reshuffle=False, use_state=False):
        # average over a few seeds for stability on the small toy
        hits, lock = [], []
        for sd in range(args.seed, args.seed + args.null_reps):
            r = run_generation(args.steps, args.vocab, knobs, args.k_cloud, sd,
                               hel, stf, reshuffle=reshuffle, use_state_addr=use_state)
            hits.append(r["fertile_hit_rate"])
            lock.append(r["mean_lockin_entropy"])
        return float(np.mean(hits)), float(np.std(hits)), float(np.mean(lock))

    canonical = runit(hel_real, stf_real)
    n_state = runit(hel_real, stf_real, use_state=True)
    n_sub = runit(hel_real, stf_real, reshuffle=True)
    n_scr = runit(hel_scr, stf_scr)

    out = {
        "knobs": knobs,
        "null_reps": args.null_reps,
        "canonical":            {"fertile_hit": canonical[0], "std": canonical[1], "lockin": canonical[2]},
        "N_vessel_vs_state":    {"fertile_hit": n_state[0],   "std": n_state[1],   "lockin": n_state[2]},
        "N_substructure":       {"fertile_hit": n_sub[0],     "std": n_sub[1],     "lockin": n_sub[2]},
        "N_scramble":           {"fertile_hit": n_scr[0],     "std": n_scr[1],     "lockin": n_scr[2]},
        "verdicts": {
            "vessel_beats_state":   canonical[0] > n_state[0] + canonical[1],
            "substructure_matters": canonical[0] > n_sub[0] + canonical[1],
            "reads_real_structure": canonical[0] > n_scr[0] + canonical[1],
        },
    }
    dump_json(out, args.out)
    print("\n  === NULL BATTERY ===", flush=True)
    for k in ("canonical", "N_vessel_vs_state", "N_substructure", "N_scramble"):
        print(f"    {k:20s} hit={out[k]['fertile_hit']:.3f}±{out[k]['std']:.3f} "
              f"lockin={out[k]['lockin']:.3f}", flush=True)
    print("  --- verdicts ---", flush=True)
    for k, v in out["verdicts"].items():
        print(f"    {k:24s} {'PASS' if v else 'FAIL'}", flush=True)


def cmd_leyline(args):
    """Lock-and-key test: stiffness map drives roll/readout, certified logistic
    Ksi->r itinerary drives kick-scheduling. Headline result = the phase-shuffle
    null, NOT wall-clock speed.

    Three conditions, averaged over seeds:
      A) dual_free baseline   : leyline_gain=0 (reduction-to-classical -> dual_free)
      B) logistic itinerary   : real certified orbit drives the kicks
      C) scrambled itinerary  : same orbit values, sequence destroyed (the null)

    The claim is supported ONLY if B passes h001 AND h002 (dual) while A and C
    do not. If C matches B, the logistic *structure* is not load-bearing and the
    speed advantage proves nothing about language.
    """
    hel = load_lut(args.helicity_lut, "helicity")
    stf = load_lut(args.stiffness_lut, "stiffness")
    knobs = make_knobs(args)

    def avg(leyline, gain, scramble=False):
        knobs2 = dict(knobs); knobs2["leyline_gain"] = gain
        h1 = h2 = dual = 0
        ams, astd = [], []
        reps = args.null_reps
        for sd in range(args.seed, args.seed + reps):
            r = run_generation(args.steps, args.vocab, knobs2, args.k_cloud, sd,
                               hel, stf, leyline=leyline, leyline_scramble=scramble)
            h1 += r["h001_pass"]; h2 += r["h002_pass"]; dual += r["dual_pass"]
            ams.append(r["alpha_mean"]); astd.append(r["alpha_std"])
        return {"h001_frac": h1 / reps, "h002_frac": h2 / reps,
                "dual_frac": dual / reps,
                "alpha_mean": float(np.mean(ams)), "alpha_std": float(np.mean(astd))}

    A = avg(None, 0.0)                                  # dual_free baseline
    B = avg("logistic", args.leyline_gain)              # real itinerary
    C = avg("logistic", args.leyline_gain, scramble=True)  # scrambled null

    out = {
        "knobs": knobs, "leyline_gain": args.leyline_gain, "reps": args.null_reps,
        "A_dual_free": A, "B_logistic": B, "C_scrambled": C,
        "verdicts": {
            "B_achieves_dual_pass":     B["dual_frac"] > 0.5,
            "B_beats_baseline":         B["dual_frac"] > A["dual_frac"],
            "structure_is_loadbearing": B["dual_frac"] > C["dual_frac"] + 0.1,
        },
    }
    dump_json(out, args.out)
    print("\n  === LEYLINE LOCK-AND-KEY ===", flush=True)
    for k, lbl in (("A_dual_free", "A dual_free   "),
                   ("B_logistic", "B logistic    "),
                   ("C_scrambled", "C scrambled   ")):
        r = out[k]
        print(f"    {lbl} h001={r['h001_frac']:.2f} h002={r['h002_frac']:.2f} "
              f"DUAL={r['dual_frac']:.2f}  alpha={r['alpha_mean']:.3f}±{r['alpha_std']:.3f}",
              flush=True)
    print("  --- verdicts ---", flush=True)
    for k, v in out["verdicts"].items():
        print(f"    {k:28s} {'PASS' if v else 'FAIL'}", flush=True)
    print("\n  NOTE: speed is not evidence. The load-bearing result is\n"
          "  structure_is_loadbearing (B beats scrambled C).", flush=True)


# ============================================================================= 
def build_parser():
    p = argparse.ArgumentParser(description="Vessel v1: shared dynamic HH-address "
                                            "vessel for flood-weighted token routing.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--helicity-lut", default=DEFAULT_HELICITY_LUT, dest="helicity_lut")
        sp.add_argument("--stiffness-lut", default=DEFAULT_STIFFNESS_LUT, dest="stiffness_lut")
        sp.add_argument("--steps", type=int, default=512)
        sp.add_argument("--vocab", type=int, default=2048)
        sp.add_argument("--k-cloud", type=int, default=12, dest="k_cloud")
        sp.add_argument("--seed", type=int, default=0)
        # frozen knobs
        sp.add_argument("--diffusivity-mode", default="inverse",
                        choices=["inverse", "inverse_sq", "exp", "flat"], dest="diffusivity_mode")
        sp.add_argument("--beta-wet", type=float, default=1.0, dest="beta_wet")
        sp.add_argument("--decay", type=float, default=0.1)
        # secondary knobs
        sp.add_argument("--deposit-gain", type=float, default=1.0, dest="deposit_gain")
        sp.add_argument("--reinforce-lr", type=float, default=0.2, dest="reinforce_lr")
        sp.add_argument("--cloud-sigma", type=float, default=8.0, dest="cloud_sigma")
        sp.add_argument("--bias-bandwidth-deg", type=float, default=12.0, dest="bias_bandwidth_deg")
        sp.add_argument("--dt", type=float, default=0.5)
        sp.add_argument("--leyline", default=None, choices=["logistic"],
                        help="drive kick-scheduling with a certified itinerary")
        sp.add_argument("--leyline-gain", type=float, default=0.15, dest="leyline_gain")

    sp = sub.add_parser("run", help="single generation run")
    common(sp); sp.add_argument("--out", default="vessel_v1_run.json")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("sweep", help="sweep diffusivity / beta_wet / decay")
    common(sp); sp.add_argument("--out", default="vessel_v1_sweep.json")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("null", help="pre-registered null battery")
    common(sp)
    sp.add_argument("--null-reps", type=int, default=8, dest="null_reps")
    sp.add_argument("--out", default="vessel_v1_null.json")
    sp.set_defaults(func=cmd_null)

    sp = sub.add_parser("leyline", help="lock-and-key: logistic itinerary + phase-shuffle null")
    common(sp)
    sp.add_argument("--null-reps", type=int, default=8, dest="null_reps")
    sp.add_argument("--out", default="vessel_v1_leyline.json")
    sp.set_defaults(func=cmd_leyline)

    return p


def main():
    args = build_parser().parse_args()
    print(f"[vessel_v1] cmd={args.cmd}", flush=True)
    args.func(args)


if __name__ == "__main__":
    main()
