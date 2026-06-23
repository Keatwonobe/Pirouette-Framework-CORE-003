"""
arc_vocabulary.py
Arc Vocabulary Builder — Manifold Grammar from Corpus Walk
Pirouette Framework Volume 8 · CORE-003 · ML-066

THE CORE IDEA
=============
The existing vocabulary maps each token to a POINT on the HH manifold:
  W[token] → (Ksi, J1)                         ← what we have now

This script builds an ARC vocabulary by walking an English corpus and
recording TRANSITIONS between token addresses:
  W[t1 → t2] → (ΔKsi, ΔJ1, arc_type, frequency)  ← what this builds

An arc is a directed edge on the manifold. It encodes not just WHERE a
token sits but HOW tokens move through semantic space.

WHY ARCS ARE DIFFERENT FROM POINTS
====================================
"the" sits at (Ksi≈0.25, J1≈0°). That's a point — it says nothing about
what comes next.

The arc "the → cat" might have ΔJ1≈+30°, ΔKsi≈+0.15 (determiner → noun,
slight diffusion toward content).
The arc "the → government" might have ΔJ1≈+80°, ΔKsi≈+0.25 (determiner
→ abstract institution noun, larger diffusion, longer arc).

Both arcs START at "the" but end in different semantic regions. The arc
vocabulary captures this: not just "the is at J1=0°" but "from 'the' you
can reach abstract nouns via a +80° arc."

WHAT THE VESSEL GAINS
======================
Currently the vessel uses stiffness and helicity as its only manifold
signals. After arc vocabulary is built:

  vessel.next_arc_target → predicted (ΔJ1, ΔKsi) for this step
    based on current token + arc grammar statistics
  vessel.softmax_bias += arc_compatibility(token, arc_target)

The vessel knows WHERE it is (current address) AND WHERE IT'S GOING
(arc target). Generation becomes directed, not just filtered.

THREE SUBCOMMANDS
==================
  walk      — walk a corpus, accumulate arc statistics
  analyze   — extract arc families, build arc grammar
  integrate — write arc priors into vessel_v1_gpt2 format

PRE-REGISTERED HYPOTHESES
==========================
H-AV-001: ARC FAMILIES EXIST
  Arc (ΔJ1, ΔKsi) vectors cluster into K < 20 statistically distinct
  families when K-means is applied. These families correspond to
  syntactic transition types (DET→NOUN, NOUN→VERB, etc.).
  PASS: silhouette score > 0.3 for K ∈ [4, 8]

H-AV-002: GRAMMAR ZONES ARE ARC-SEPARABLE
  Mean arc ΔJ1 differs significantly between Zone A (80-140°) departures
  and Zone B (280-340°) departures. Zone A departure arcs are shorter
  (more focused transitions) than Zone B departure arcs.
  PASS: |mean_ΔJ1(ZA) - mean_ΔJ1(ZB)| > 30°, p < 0.05

H-AV-003: ARC PRIOR IMPROVES VESSEL COHERENCE
  Adding arc_prior bias to vessel_v1_gpt2 softmax_bias increases
  mean_word_length above the stiffness-only baseline (6.02).
  PASS: mean_word_length > 6.5 with arc_prior enabled.

Usage:
  # Walk a corpus and build arc statistics:
  python arc_vocabulary.py walk ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --corpus corpus.txt ^
    --output arc_stats.json ^
    --max_sentences 5000

  # Analyze arc families:
  python arc_vocabulary.py analyze ^
    --arc_stats arc_stats.json ^
    --output arc_grammar.json ^
    --n_clusters 8

  # Integrate with vessel:
  python arc_vocabulary.py integrate ^
    --arc_grammar arc_grammar.json ^
    --engram engram_curve.json ^
    --output arc_prior.npy
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

K_PROJ = 16

# Arc type classification
ZONE_A = (80, 140)    # sentence starters, coherence peak
ZONE_B = (280, 340)   # flow words
DEAD   = (140, 265)   # garbage zone

def in_zone(j1, zone):
    return zone[0] <= (j1 % 360) <= zone[1]

def arc_type(j1_start, j1_end, dksi):
    """Classify a manifold arc into one of 7 grammar-meaningful types."""
    dj1 = float(((j1_end - j1_start + 180) % 360) - 180)  # signed shortest
    s = in_zone(j1_start, ZONE_A)
    e = in_zone(j1_end, ZONE_A)
    s_b = in_zone(j1_start, ZONE_B)
    e_b = in_zone(j1_end, ZONE_B)
    s_d = in_zone(j1_start, DEAD)
    e_d = in_zone(j1_end, DEAD)

    if s_d or e_d:                         return "DEAD_ZONE"
    if abs(dj1) < 20 and abs(dksi) < 0.08: return "HOVER"      # near-zero arc
    if s and e:                             return "ZONE_A_INT"  # within A
    if s_b and e_b:                         return "ZONE_B_INT"  # within B
    if s and e_b:                           return "A_TO_B"      # A → B hop
    if s_b and e:                           return "B_TO_A"      # B → A hop
    if abs(dj1) > 120:                      return "LONG_PIVOT"  # topic shift
    return "CROSS"                                                # other crossing

# ── Utilities ──────────────────────────────────────────────────────────────────

def load_model_engram(model_path, engram_path):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print(f"  Loading model: {model_path}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    tok   = GPT2Tokenizer.from_pretrained(model_path)

    with open(engram_path) as f: e = json.load(f)
    ksi_v = np.array(e["ksi_vals"], dtype=np.float32)
    j1_v  = np.array(e["j1_pca"], dtype=np.float32) % 360.0
    pc1   = np.array(e["pc1"], dtype=np.float32)
    pc2   = np.array(e["pc2"], dtype=np.float32)
    U_ref = np.array(e["U_ref"], dtype=np.float32)

    print(f"  Engram: {len(ksi_v)} tokens", flush=True)
    return tok, ksi_v, j1_v, pc1, pc2, U_ref

def angular_distance_signed(a, b):
    """Signed shortest angular distance a→b in degrees."""
    return float(((b - a + 180) % 360) - 180)

def read_corpus(path, max_sentences=10000):
    """Read corpus file, return list of sentences."""
    lines = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= max_sentences:
                break
    return lines

# ── Subcommand: walk ──────────────────────────────────────────────────────────

def cmd_walk(args):
    """
    Walk a corpus and accumulate arc statistics for each bigram.
    This is the core data collection pass.
    """
    print("\n" + "="*70)
    print("ARC WALK — Mapping Token Transitions to Manifold Arcs")
    print("="*70)

    tok, ksi_v, j1_v, pc1, pc2, U_ref = load_model_engram(args.model, args.engram)
    max_sent = getattr(args, 'max_sentences', 5000)

    # Check if corpus is a file or use a built-in mini-corpus
    corpus_path = getattr(args, 'corpus', None)
    if corpus_path and Path(corpus_path).exists():
        sentences = read_corpus(corpus_path, max_sent)
        print(f"  Corpus: {len(sentences)} sentences from {corpus_path}")
    else:
        print("  No corpus file found — using built-in English seed sentences")
        print("  (For full arc vocabulary, provide --corpus with a text file)")
        sentences = SEED_CORPUS
        print(f"  Seed corpus: {len(sentences)} sentences")

    # Arc accumulators
    # arc_sums[t] = [sum_dj1, sum_dksi, count] — destination token statistics
    # arc_type_counts[arc_type] = count
    # departure_stats[t] = [sum_dj1_departing, count_departing]
    arc_sums = defaultdict(lambda: np.zeros(3))   # [dj1, dksi, count]
    type_counts = defaultdict(int)
    departure_sums = defaultdict(lambda: np.zeros(3))  # per source token
    bigram_arcs = defaultdict(list)  # (t1_id, t2_id) → list of arc vectors

    n_arcs = 0; n_dead = 0; n_hover = 0
    print(f"\n  Walking {len(sentences)} sentences...")

    for sent_idx, sentence in enumerate(sentences):
        ids = tok.encode(sentence, add_special_tokens=False)
        if len(ids) < 2:
            continue

        for i in range(len(ids) - 1):
            t1 = ids[i]; t2 = ids[i + 1]
            if t1 >= len(ksi_v) or t2 >= len(ksi_v):
                continue

            ksi1 = float(ksi_v[t1]); j1_1 = float(j1_v[t1])
            ksi2 = float(ksi_v[t2]); j1_2 = float(j1_v[t2])

            dj1  = angular_distance_signed(j1_1, j1_2)
            dksi = ksi2 - ksi1
            atype = arc_type(j1_1, j1_2, dksi)

            arc_sums[t2][0]       += dj1
            arc_sums[t2][1]       += dksi
            arc_sums[t2][2]       += 1
            departure_sums[t1][0] += dj1
            departure_sums[t1][1] += dksi
            departure_sums[t1][2] += 1
            type_counts[atype]    += 1

            key = (t1, t2)
            bigram_arcs[key].append((float(j1_1), float(ksi1),
                                     float(j1_2), float(ksi2),
                                     float(dj1), float(dksi), atype))
            n_arcs += 1
            if atype == "DEAD_ZONE": n_dead += 1
            if atype == "HOVER":     n_hover += 1

        if sent_idx % 500 == 0:
            print(f"    sentence {sent_idx}/{len(sentences)}  "
                  f"arcs={n_arcs}", flush=True)

    print(f"\n  Total arcs: {n_arcs}")
    print(f"  Dead zone arcs: {n_dead} ({100*n_dead/(n_arcs+1):.1f}%)")
    print(f"  Hover arcs:     {n_hover} ({100*n_hover/(n_arcs+1):.1f}%)")
    print(f"\n  Arc type distribution:")
    for atype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {atype:<15} {cnt:>8}  ({100*cnt/n_arcs:.1f}%)")

    # Build per-token arrival arc statistics
    print(f"\n  Building arc prior vectors...")
    token_arc_arrivals  = {}   # what arc most commonly arrives at this token
    token_arc_departures = {}  # what arc most commonly departs from this token

    for t2_id, sums in arc_sums.items():
        if sums[2] < 1: continue
        count = float(sums[2])
        token_arc_arrivals[int(t2_id)] = {
            "mean_dj1":  float(sums[0] / count),
            "mean_dksi": float(sums[1] / count),
            "count":     int(count),
            "ksi":       float(ksi_v[t2_id]),
            "j1":        float(j1_v[t2_id]),
        }

    for t1_id, sums in departure_sums.items():
        if sums[2] < 1: continue
        count = float(sums[2])
        token_arc_departures[int(t1_id)] = {
            "mean_dj1":  float(sums[0] / count),
            "mean_dksi": float(sums[1] / count),
            "count":     int(count),
            "ksi":       float(ksi_v[t1_id]),
            "j1":        float(j1_v[t1_id]),
        }

    # Top 50 most common bigrams with their arcs
    bigram_summary = []
    for (t1, t2), arc_list in bigram_arcs.items():
        if len(arc_list) < 3: continue
        dj1s  = [a[4] for a in arc_list]
        dksis = [a[5] for a in arc_list]
        atypes = [a[6] for a in arc_list]
        from collections import Counter
        dominant_type = Counter(atypes).most_common(1)[0][0]
        t1s = tok.decode([t1], skip_special_tokens=False)
        t2s = tok.decode([t2], skip_special_tokens=False)
        bigram_summary.append({
            "t1_id": int(t1), "t1_str": t1s,
            "t2_id": int(t2), "t2_str": t2s,
            "count":     len(arc_list),
            "mean_dj1":  float(np.mean(dj1s)),
            "mean_dksi": float(np.mean(dksis)),
            "std_dj1":   float(np.std(dj1s)),
            "dominant_arc_type": dominant_type,
            "j1_t1": float(j1_v[t1]), "ksi_t1": float(ksi_v[t1]),
            "j1_t2": float(j1_v[t2]), "ksi_t2": float(ksi_v[t2]),
        })
    bigram_summary.sort(key=lambda x: -x["count"])

    print(f"\n  Top 20 bigrams by frequency:")
    print(f"  {'Bigram':<20} {'count':>6} {'ΔJ1':>8} {'ΔKsi':>7} {'arc_type'}")
    for bg in bigram_summary[:20]:
        pair = f"'{bg['t1_str']}' → '{bg['t2_str']}'"
        print(f"  {pair:<20} {bg['count']:>6} "
              f"{bg['mean_dj1']:>8.1f}° {bg['mean_dksi']:>7.3f}  {bg['dominant_arc_type']}")

    output = {
        "n_sentences":    len(sentences),
        "n_arcs":         n_arcs,
        "arc_type_distribution": dict(type_counts),
        "token_arc_arrivals":   token_arc_arrivals,
        "token_arc_departures": token_arc_departures,
        "top_bigrams":    bigram_summary[:200],
    }
    _save(output, args.output)
    print(f"\n  Saved → {args.output}")

# ── Subcommand: analyze ───────────────────────────────────────────────────────

def cmd_analyze(args):
    """
    Cluster arc vectors into families and extract the manifold grammar.
    H-AV-001: do arc families exist?
    H-AV-002: are grammar zones arc-separable?
    """
    print("\n" + "="*70)
    print("ARC ANALYZE — Extract Arc Families and Manifold Grammar")
    print("="*70)

    with open(args.arc_stats) as f:
        stats = json.load(f)

    n_clusters = getattr(args, 'n_clusters', 8)
    arrivals   = stats["token_arc_arrivals"]
    bigrams    = stats["top_bigrams"]

    if not bigrams:
        print("  No bigram data found. Run 'walk' first.")
        sys.exit(1)

    # Build feature matrix for clustering
    features = []
    labels   = []
    for bg in bigrams:
        if bg["count"] < 2: continue
        features.append([bg["mean_dj1"], bg["mean_dksi"] * 180.0,
                          bg["j1_t1"], bg["j1_t2"]])
        labels.append(bg)
    X = np.array(features, dtype=np.float32)
    print(f"  Clustering {len(X)} bigrams into {n_clusters} families...")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    best_sil = -1; best_k = n_clusters
    for k in range(4, min(n_clusters + 1, len(X) // 5 + 1)):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbs = km.fit_predict(X_s)
        if len(set(lbs)) < 2: continue
        sil = silhouette_score(X_s, lbs)
        print(f"    k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil; best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_s)
    print(f"\n  Best k={best_k}  silhouette={best_sil:.4f}")
    h001 = "PASS" if best_sil > 0.3 else "FAIL"
    print(f"  H-AV-001: {h001} (threshold=0.3)")

    # Characterize each cluster
    families = []
    print(f"\n  Arc families:")
    for c in range(best_k):
        mask = cluster_labels == c
        cluster_bgs = [labels[i] for i in range(len(labels)) if mask[i]]
        if not cluster_bgs: continue
        dj1s  = [b["mean_dj1"]  for b in cluster_bgs]
        dksis = [b["mean_dksi"] for b in cluster_bgs]
        types = [b["dominant_arc_type"] for b in cluster_bgs]
        from collections import Counter
        dominant_type = Counter(types).most_common(1)[0][0]
        total_count   = sum(b["count"] for b in cluster_bgs)
        example_bgs   = sorted(cluster_bgs, key=lambda b: -b["count"])[:5]

        family = {
            "family_id":      int(c),
            "n_bigrams":      int(sum(mask)),
            "total_count":    int(total_count),
            "mean_dj1":       float(np.mean(dj1s)),
            "std_dj1":        float(np.std(dj1s)),
            "mean_dksi":      float(np.mean(dksis)),
            "dominant_type":  dominant_type,
            "examples": [(b["t1_str"], b["t2_str"], b["count"])
                          for b in example_bgs],
        }
        families.append(family)
        print(f"\n  Family {c}: {dominant_type}  n={sum(mask)}  "
              f"ΔJ1={np.mean(dj1s):.1f}°±{np.std(dj1s):.1f}°  "
              f"ΔKsi={np.mean(dksis):+.3f}")
        ex_str = ', '.join(f"{b['t1_str']}→{b['t2_str']}({b['count']})" for b in example_bgs[:3])
        print(f"    Examples: {ex_str}")

    # H-AV-002: Zone A vs Zone B departure arc lengths
    za_dj1s = [b["mean_dj1"] for b in bigrams
               if ZONE_A[0] <= b["j1_t1"] <= ZONE_A[1]]
    zb_dj1s = [b["mean_dj1"] for b in bigrams
               if ZONE_B[0] <= b["j1_t1"] <= ZONE_B[1]]
    h002 = "NOT_TESTABLE"
    if za_dj1s and zb_dj1s:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(za_dj1s, zb_dj1s)
        diff = abs(np.mean(za_dj1s) - np.mean(zb_dj1s))
        h002 = "PASS" if (diff > 30 and p_val < 0.05) else "FAIL"
        print(f"\n  H-AV-002: mean_ΔJ1(ZoneA)={np.mean(za_dj1s):.1f}°  "
              f"mean_ΔJ1(ZoneB)={np.mean(zb_dj1s):.1f}°  "
              f"diff={diff:.1f}°  p={p_val:.3e}  → {h002}")

    output = {
        "n_clusters_best":   int(best_k),
        "silhouette_score":  float(best_sil),
        "h001_verdict":      h001,
        "h002_verdict":      h002,
        "arc_families":      families,
        "zone_a_mean_dj1":   float(np.mean(za_dj1s)) if za_dj1s else None,
        "zone_b_mean_dj1":   float(np.mean(zb_dj1s)) if zb_dj1s else None,
    }
    _save(output, args.output)
    print(f"\n  Saved → {args.output}")

# ── Subcommand: integrate ─────────────────────────────────────────────────────

def cmd_integrate(args):
    """
    Build the arc_prior vector: for each token, what is the expected
    arc it typically arrives via? Saves as arc_prior.npy for vessel use.

    Format: arc_prior[vocab_size × 2]
      arc_prior[t, 0] = mean ΔJ1 when arriving at token t (degrees)
      arc_prior[t, 1] = mean ΔKsi when arriving at token t

    At generation time in vessel_v1_gpt2:
      arc_target = vessel.next_target_arc()  → (target_dj1, target_dksi)
      arc_compat  = exp(-((arc_prior[:,0] - target_dj1)/sigma_j1)²
                      - ((arc_prior[:,1] - target_dksi)/sigma_ksi)²)
      biased_logits += beta_arc * log1p(arc_compat)
    """
    print("\n" + "="*70)
    print("ARC INTEGRATE — Build Arc Prior for Vessel")
    print("="*70)

    with open(args.arc_grammar) as f:
        grammar = json.load(f)
    with open(args.engram) as f:
        e = json.load(f)

    ksi_v = np.array(e["ksi_vals"], dtype=np.float32)
    j1_v  = np.array(e["j1_pca"], dtype=np.float32) % 360.0
    V     = len(ksi_v)

    # Initialize arc_prior with NaN (= no data)
    arc_prior = np.full((V, 2), np.nan, dtype=np.float32)

    # Fill from arc_stats if available
    arc_stats_path = getattr(args, 'arc_stats', None)
    if arc_stats_path and Path(arc_stats_path).exists():
        with open(arc_stats_path) as f:
            stats = json.load(f)
        arrivals = stats.get("token_arc_arrivals", {})
        filled = 0
        for t_str, arr in arrivals.items():
            t = int(t_str)
            if t < V:
                arc_prior[t, 0] = float(arr["mean_dj1"])
                arc_prior[t, 1] = float(arr["mean_dksi"])
                filled += 1
        print(f"  Filled {filled}/{V} tokens from arc walk statistics")
    else:
        print("  No arc_stats provided — building prior from engram geometry only")

    # Fill missing tokens with geometry-based prior
    # Default: token at (Ksi, J1) expects short arc from within same zone
    missing = np.isnan(arc_prior[:, 0])
    n_missing = missing.sum()
    if n_missing > 0:
        print(f"  Computing geometry prior for {n_missing} tokens...")
        for t in np.where(missing)[0]:
            j1 = float(j1_v[t])
            ksi = float(ksi_v[t])
            # Geometry prior: tokens in Zone A tend to arrive from Zone B (short B→A)
            if ZONE_A[0] <= j1 <= ZONE_A[1]:
                arc_prior[t, 0] = -90.0   # arrive from Zone B (clockwise)
                arc_prior[t, 1] = -0.10   # slight focus
            elif ZONE_B[0] <= j1 <= ZONE_B[1]:
                arc_prior[t, 0] = +90.0   # arrive from Zone A (counter-clockwise)
                arc_prior[t, 1] = +0.05   # slight diffusion
            elif DEAD[0] <= j1 <= DEAD[1]:
                arc_prior[t, 0] = 0.0     # neutral (dead zone)
                arc_prior[t, 1] = 0.0
            else:
                arc_prior[t, 0] = 0.0     # neutral
                arc_prior[t, 1] = 0.0

    print(f"\n  Arc prior statistics:")
    valid = ~np.isnan(arc_prior[:, 0])
    print(f"    dj1:  mean={arc_prior[valid,0].mean():.1f}°  "
          f"std={arc_prior[valid,0].std():.1f}°")
    print(f"    dksi: mean={arc_prior[valid,1].mean():.4f}  "
          f"std={arc_prior[valid,1].std():.4f}")

    np.save(args.output, arc_prior)
    print(f"\n  Saved arc_prior.npy ({V} × 2 float32) → {args.output}")
    print(f"  Load in vessel: arc_prior = np.load('{args.output}')")
    print(f"  Usage: arc_compat[t] = exp(-((arc_prior[t,0]-target_dj1)/20)²)")

# ── Built-in seed corpus ───────────────────────────────────────────────────────
# Used when no corpus file is provided. 200 sentences covering diverse English.

SEED_CORPUS = [
    "The cat sat on the mat and watched the birds outside.",
    "In the beginning, the universe was created from nothing.",
    "She walked slowly through the garden, thinking about what had happened.",
    "The government announced new policies to address the economic crisis.",
    "Scientists discovered a new species of fish in the deep ocean.",
    "He opened the book and began to read the first chapter carefully.",
    "The sun rises in the east and sets in the west every day.",
    "Children learn language naturally through exposure and interaction.",
    "The ancient ruins were discovered by archaeologists in the desert.",
    "Music has the power to evoke deep emotions in the listener.",
    "The stock market fell sharply following the unexpected announcement.",
    "She studied mathematics for years before becoming a professor.",
    "The river flows down from the mountains to the sea.",
    "Democracy depends on informed citizens who participate in elections.",
    "The doctor examined the patient and prescribed medication.",
    "Beautiful flowers bloomed in the garden after the spring rain.",
    "Technology has transformed how people communicate and work.",
    "The chef prepared a delicious meal using fresh local ingredients.",
    "History repeats itself when people fail to learn from the past.",
    "The astronauts returned safely after six months on the space station.",
    "Language is the most powerful tool humans have ever developed.",
    "The mountains were covered in snow during the long winter months.",
    "She built a successful business through hard work and determination.",
    "The treaty brought peace to the region after decades of conflict.",
    "Music therapy helps patients recover from neurological conditions.",
    "The library contained thousands of books on every subject imaginable.",
    "He solved the complex equation by breaking it into smaller parts.",
    "The economy grew steadily as unemployment rates continued to fall.",
    "Birds migrate south in autumn to escape the cold northern winters.",
    "The professor explained the theory using simple everyday examples.",
    "Innovation requires both creativity and the willingness to fail.",
    "The city was built on a network of canals and bridges.",
    "She painted landscapes inspired by the countryside where she grew up.",
    "The experiment confirmed the hypothesis that had been proposed.",
    "Forests absorb carbon dioxide and produce oxygen for the planet.",
    "The athlete trained for years to compete in the Olympics.",
    "Philosophy asks questions that science alone cannot answer.",
    "The engine hummed quietly as the train moved through the night.",
    "Newspapers reported on the election results throughout the country.",
    "The software was updated to fix security vulnerabilities.",
    "Water covers approximately seventy percent of the Earth's surface.",
    "The committee reviewed the proposal and requested further information.",
    "She taught the children to read using colorful illustrated books.",
    "The bridge was constructed over fifty years ago and still stands.",
    "Mathematics is the language in which the laws of nature are written.",
    "The farmer harvested the wheat before the rains arrived.",
    "Space exploration has revealed the vastness of the universe.",
    "The hospital treated hundreds of patients during the epidemic.",
    "Art reflects the values and concerns of the society that creates it.",
    "The debate continued for hours without reaching a conclusion.",
    "Neurons communicate through electrical and chemical signals.",
    "The composer wrote the symphony over the course of three years.",
    "Global temperatures have risen significantly over the past century.",
    "The engineer designed the bridge to withstand extreme weather.",
    "Meditation reduces stress and improves mental clarity.",
    "The parliament passed legislation to protect environmental resources.",
    "Ancient civilizations built remarkable structures without modern tools.",
    "The journalist investigated the corruption and published the findings.",
    "Light travels faster than any other known physical phenomenon.",
    "The author wrote novels that explored the nature of human identity.",
    "Trade routes connected distant cultures and facilitated exchange.",
    "The researcher analyzed data from thousands of survey responses.",
    "Education is the foundation on which prosperous societies are built.",
    "The volcano erupted without warning and covered the valley in ash.",
    "She learned to play the piano by practicing every single day.",
    "Genetics determines many aspects of physical appearance and health.",
    "The negotiators worked through the night to reach an agreement.",
    "Forests provide habitat for millions of species of animals.",
    "The invention of the printing press changed the world forever.",
    "She argued that the policy would harm rather than help communities.",
    "The satellite transmitted images back to the research station.",
    "Philosophy of mind investigates the relationship between brain and consciousness.",
    "The team collaborated effectively despite working in different countries.",
    "Natural disasters can reshape entire landscapes in moments.",
    "The museum displayed artifacts from ancient civilizations.",
    "Computer algorithms can detect patterns invisible to the human eye.",
    "The river overflowed after days of heavy continuous rainfall.",
    "She balanced work and family with remarkable efficiency.",
    "The discovery changed what scientists believed about human origins.",
    "The economy depends on trust between buyers and sellers.",
    "Children develop empathy through stories and play.",
    "The telescope revealed galaxies billions of light years away.",
    "Engineers solved the problem using principles from fluid dynamics.",
    "The forest was silent except for the sound of falling leaves.",
    "Migration has shaped cultures and economies throughout history.",
    "The surgeon performed the operation with precision and care.",
    "Mathematical proofs require both creativity and rigorous logic.",
    "The debate about free will has occupied philosophers for centuries.",
    "She used satellite data to track changes in the polar ice.",
    "The novel explored themes of loss identity and redemption.",
    "Biodiversity ensures that ecosystems remain resilient and productive.",
    "The building was designed to minimize energy consumption.",
    "Language acquisition follows predictable developmental stages.",
    "The election results surprised analysts who predicted a different outcome.",
    "She developed a new method for treating antibiotic resistant infections.",
    "The mountains were formed by the collision of tectonic plates.",
    "Artificial intelligence is transforming industries from medicine to finance.",
    "The dancers moved in perfect synchrony to the complex rhythm.",
    "Archaeology reveals how ancient peoples lived thought and created.",
    "The ocean depths remain largely unexplored and mysterious.",
    "She translated the ancient text to reveal its hidden meanings.",
    "The market responded positively to the announcement of new growth.",
    "Architecture shapes how people experience the spaces they inhabit.",
    "The algorithm processes millions of data points in seconds.",
    "The river carved the canyon over millions of years.",
    "She overcame numerous obstacles to achieve her goals.",
    "The mission gathered data that will take decades to fully analyze.",
    "Language evolves continuously as cultures change and interact.",
    "The laboratory developed a vaccine in record time.",
    "Philosophy shapes how we understand justice freedom and truth.",
    "The weather system brought storms across the entire northern region.",
    "She built coalitions across different communities to achieve change.",
    "The signal was detected by instruments sensitive enough to measure it.",
    "Ancient trade networks spread ideas as well as goods.",
    "The composer blended traditional and contemporary musical influences.",
    "Soil health determines the productivity of agricultural land.",
    "The treaty established boundaries that have lasted for generations.",
    "She analyzed the problem from multiple theoretical perspectives.",
    "The machine learned to recognize patterns through repeated exposure.",
    "Tides are caused by the gravitational pull of the moon and sun.",
    "The debate revealed deep disagreements about fundamental values.",
    "She documented the endangered language before its last speakers died.",
    "The infrastructure required decades of investment and planning.",
    "Chemical reactions release or absorb energy in measurable quantities.",
    "The council voted unanimously in favor of the proposal.",
    "She described the galaxy as a slowly rotating pinwheel of stars.",
    "Logic provides tools for evaluating the validity of arguments.",
    "The expedition mapped previously uncharted regions of the continent.",
    "Medicine advances through careful observation and controlled experiments.",
    "The composer heard music in the patterns of everyday life.",
    "Inequality undermines the social cohesion that democracy requires.",
    "The organism adapted to its changing environment over generations.",
    "She wrote her thesis on the intersection of language and power.",
    "The policy reduced emissions while maintaining economic growth.",
    "Crystals form when atoms arrange themselves in regular geometric patterns.",
    "The architect designed the building to draw in natural light.",
    "She examined the relationship between sleep and memory consolidation.",
    "The signal was faint but detectable with sensitive instruments.",
    "Competition and cooperation are both essential to evolutionary success.",
    "The researcher mapped the neural pathways involved in decision making.",
    "Cities grow along rivers because water supports agriculture and trade.",
    "She played the violin with technical precision and emotional depth.",
    "The system failed because of a single point of vulnerability.",
    "Language shapes perception in ways that speakers rarely notice.",
    "The probe transmitted data from the surface of the distant planet.",
    "She identified the bacterium responsible for the epidemic.",
    "The novel described the city as a character with its own desires.",
    "Mathematics describes both the structure and the limits of knowledge.",
    "The experiment required conditions free from any contamination.",
    "She demonstrated that the effect persisted across different contexts.",
    "The river delta is one of the most biodiverse regions on Earth.",
    "He argued that the meaning of words shifts over historical time.",
    "The data supported the hypothesis beyond reasonable doubt.",
    "She synthesized findings from hundreds of studies on the topic.",
    "The glacier moved imperceptibly but continuously toward the sea.",
    "Reason and intuition both play roles in mathematical discovery.",
    "The organization distributed resources to communities in need.",
    "She designed experiments that could distinguish competing theories.",
    "The forest regenerated naturally after the fire cleared the land.",
    "The evidence pointed clearly toward a single explanation.",
    "Communication between cells coordinates the development of the organism.",
    "She balanced competing demands with grace and attention to detail.",
    "The satellite detected a change in atmospheric composition.",
    "Art and science both seek truth through different methods.",
    "The architect balanced aesthetics with structural requirements.",
    "She questioned assumptions that others had accepted without examination.",
    "The data revealed patterns that had been invisible before analysis.",
    "The city evolved over centuries from a small trading post.",
    "She integrated insights from multiple disciplines into a unified framework.",
    "The river shaped the culture of every civilization along its banks.",
    "He solved problems by first understanding them from multiple angles.",
    "The experiment was replicated in laboratories across the world.",
    "Language connects us to those who came before and those who follow.",
    "She published results that challenged long-held assumptions in the field.",
    "The structure of DNA encodes the instructions for all living things.",
    "The desert blooms briefly but brilliantly after rare rainfall.",
    "She built a model that predicted outcomes with high accuracy.",
    "The debate continues because the question admits no simple answer.",
    "Light bends when it passes through a medium of different density.",
    "She negotiated an agreement that satisfied all parties involved.",
    "The telescope changed humanity's understanding of its place in the universe.",
    "Memory is reconstructive rather than reproductive in nature.",
    "The expedition returned with specimens that transformed the field.",
    "She connected disparate findings into a coherent theoretical framework.",
    "The system generates outputs that no single component could produce alone.",
    "He found beauty in the precise formal structures of mathematics.",
    "The organism responded to stress by activating protective mechanisms.",
    "She traced the history of the idea across cultures and centuries.",
    "The mountain ecosystem is fragile and responds quickly to disturbance.",
    "He transformed abstract mathematical objects into intuitive visual forms.",
    "The measurement revealed a discrepancy that required a new explanation.",
    "She created work that spoke simultaneously to individuals and to history.",
    "The boundary between the disciplines grew more permeable over time.",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(obj, path):
    def _cvt(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.int32, np.int64, np.intp)): return int(o)
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.bool_, bool)): return bool(o)
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_cvt)

def main():
    ap = argparse.ArgumentParser(description="Arc Vocabulary Builder")
    sub = ap.add_subparsers(dest="cmd")

    pw = sub.add_parser("walk")
    pw.add_argument("--model",         required=True)
    pw.add_argument("--engram",        required=True)
    pw.add_argument("--corpus",        default=None)
    pw.add_argument("--output",        default="arc_stats.json")
    pw.add_argument("--max_sentences", type=int, default=5000)

    pa = sub.add_parser("analyze")
    pa.add_argument("--arc_stats",  required=True)
    pa.add_argument("--output",     default="arc_grammar.json")
    pa.add_argument("--n_clusters", type=int, default=8)

    pi = sub.add_parser("integrate")
    pi.add_argument("--arc_grammar", required=True)
    pi.add_argument("--engram",      required=True)
    pi.add_argument("--arc_stats",   default=None)
    pi.add_argument("--output",      default="arc_prior.npy")

    args = ap.parse_args()
    {"walk": cmd_walk, "analyze": cmd_analyze, "integrate": cmd_integrate
     }.get(args.cmd, lambda _: (ap.print_help(), sys.exit(1)))(args)

if __name__ == "__main__":
    main()
