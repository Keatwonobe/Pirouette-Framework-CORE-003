"""
wiki_bivector_field.py
Wikipedia Bivector Field Builder
Pirouette Framework Volume 8 · CORE-003 · ML-070

THE IDEA
=========
The bigram database covers 42% of the vocabulary — it misses 58% of tokens
because they haven't appeared in our small corpus. The bivector field solves
this by mapping positions on the manifold (Ksi, J1) → expected next arc,
independent of which specific token is at that position.

From Wikipedia's multistream XML (enwiki-YYYYMMDD-pages-articles-multistream.xml.bz2):
  - Stream articles directly (no full decompress needed for subsets)
  - Clean wikitext markup (templates, refs, links, tables)
  - Tokenize with GPT-2 tokenizer
  - For each bigram (t1, t2): record arc (ΔJ1, ΔKsi) → accumulate into field

The FIELD: a 36×20 grid over (J1, Ksi) space.
  J1: 36 sectors of 10° each (0-360°)
  Ksi: 20 bands of 0.05 each (0-1.0)

At each cell (j1_sector, ksi_sector):
  field[j1, ksi] = [sum_ΔJ1, sum_ΔKsi, count, sum_ΔJ1², sum_ΔKsi²]

This gives: mean arc, variance, and confidence at each manifold position.
Storage: ~720 cells × 5 floats = trivially small numpy array (17KB).

PUNCTUATION FILTERING
======================
Wikipedia wikitext contains structured noise that produces meaningless arcs:
  - [[link|display]] tokens map to fragmented addresses
  - {{template}} produces systematic artifacts
  - HTML entities, table markup, category tags

Filter strategy:
  1. Strip all wikitext markup BEFORE tokenizing
  2. Quality filter: sentence must be >70% alphabetic characters
  3. Token-level: skip arcs involving tokens with J1 in dead zone AND
     no alphabetic characters (pure punctuation in garbage zone)
  4. Arc-level: skip DEAD arcs (J1_start or J1_end in [140°, 265°])
     These are systematically noise from tokenizer artifacts

WHAT THE RSI LOOP LOOKS LIKE
=============================
The fractal telescope metaphor implies this:

  orbit 0: model generates text from manifold position
  orbit 1: generated text re-enters as corpus → updates bivector field
  orbit 2: updated field gives better arc predictions → better generation
  orbit 3: better generation → higher quality text → better bivectors
  ...

The bivector field is the model's "proprioception" — its sense of where it
is on the manifold and where it naturally goes from there. As it reads more
of its own high-quality output, the field self-refines.

The RSI gate: generated text quality can be measured (Ksi trajectory
coherence, A_INT fraction, absence of DEAD arcs). Only high-quality
output (coherence > threshold) feeds back into the field update.
This prevents degenerate fixed points.

THREE SUBCOMMANDS
=================
  build     — read Wikipedia XML, build bivector field
  augment   — add additional text files to existing field
  generate  — use field for arc-steered generation (replaces bigram db)

Usage:
  # Build from Wikipedia multistream (streams N articles, no full decompress):
  python wiki_bivector_field.py build ^
    --wiki enwiki-20250201-pages-articles-multistream.xml.bz2 ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --output bivector_field.npz ^
    --n_articles 100000

  # Augment with existing corpus files:
  python wiki_bivector_field.py augment ^
    --field bivector_field.npz ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --texts bigram_db_combined.json ^
    --output bivector_field_aug.npz

  # Generate using field (drop-in for wandering_model_arc.py):
  python wiki_bivector_field.py generate ^
    --model models\\gpt2-large-cycle3-cust-arc1 ^
    --engram engram_curve.json ^
    --baseline_file baseline_math.json ^
    --field bivector_field.npz ^
    --prompt "The cause of altruistic behavior is" ^
    --delta_end 0.08 --arc_strength 0.10 ^
    --output wiki_gen.json
"""

import argparse, bz2, json, re, sys, time
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

K_PROJ = 16
FIELD_J1_BINS  = 36   # 10° per bin
FIELD_KSI_BINS = 20   # 0.05 per bin

# ── Wikipedia text cleaning ────────────────────────────────────────────────────

# Punctuation strings to de-emphasize (filter ARC if either endpoint is one of these)
PUNCT_ONLY_RE = re.compile(r'^[^a-zA-Z0-9]+$')
# Wiki hyperlink artifacts that produce garbage addresses
WIKI_ARTIFACT_RE = re.compile(r'^(disambiguation|redirect|stub|template|see also)$',
                               re.IGNORECASE)

RE_REFS = re.compile(r'<ref[^>]*/>|<ref[^>]*>.*?</ref>', re.DOTALL)
RE_HTML_COMMENTS = re.compile(r'', re.DOTALL)
RE_HTML_TAGS = re.compile(r'<[^>]+>')
RE_CATEGORIES = re.compile(r'\[\[(Category|File|Image|Media|Talk):[^\]]*\]\]', re.IGNORECASE)
RE_PIPE_LINKS = re.compile(r'\[\[([^|\]]+)\|([^\]]+)\]\]')
RE_LINKS = re.compile(r'\[\[([^\]]+)\]\]')
RE_EXT_LINKS_TEXT = re.compile(r'\[https?://\S+\s+([^\]]+)\]')
RE_EXT_LINKS = re.compile(r'\[https?://\S+\]')
RE_BRACKETS = re.compile(r'[\[\]]')
RE_TABLE_ROWS = re.compile(r'^\s*[\|!].*$', re.MULTILINE)
RE_TABLES = re.compile(r'\{\|.*?\|\}', re.DOTALL)
RE_BOLD_ITALIC = re.compile(r"'{2,}")
RE_HEADINGS = re.compile(r'={2,}[^=]*={2,}')
RE_URLS = re.compile(r'https?://\S+')
RE_SPACES = re.compile(r'[ \t]+')
RE_NEWLINES = re.compile(r'\n\s*\n+')
RE_SENTENCES = re.compile(r'(?<=[.!?])\s+')
RE_TEMPLATES = re.compile(r'\{\{[^{}]*\}\}')

def clean_wikitext(text: str) -> str:
    """
    Convert raw wikitext to clean prose sentences.
    Strips: templates, refs, HTML, wikilinks, tables, markup
    Preserves: words, standard sentence punctuation
    """
    # FAST TEMPLATE STRIPPING: 
    # 4 passes of regex to strip nested {{templates}} instantly in C.
    # This prevents the script from getting trapped by an unclosed bracket if sliced.
    for _ in range(4):
        text = RE_TEMPLATES.sub(' ', text)

    # Slice to 5000 characters AFTER templates are stripped to save tokenizer time
    text = text[:5000]

    # Remove <ref> blocks
    text = RE_REFS.sub(' ', text)

    # Remove HTML tags and comments
    text = RE_HTML_COMMENTS.sub(' ', text)
    text = RE_HTML_TAGS.sub(' ', text)

    # Remove category/file/image links entirely
    text = RE_CATEGORIES.sub(' ', text)

    # [[link|display]] → display
    text = RE_PIPE_LINKS.sub(r'\2', text)
    # [[link]] → link
    text = RE_LINKS.sub(r'\1', text)

    # External links [url text] → text
    text = RE_EXT_LINKS_TEXT.sub(r'\1', text)
    text = RE_EXT_LINKS.sub(' ', text)

    # Remaining brackets
    text = RE_BRACKETS.sub(' ', text)

    # Tables (|...) — remove table syntax
    text = RE_TABLE_ROWS.sub(' ', text)
    text = RE_TABLES.sub(' ', text)

    # Bold/italic markers
    text = RE_BOLD_ITALIC.sub('', text)

    # Section headings == Heading ==
    text = RE_HEADINGS.sub(' ', text)

    # HTML entities
    for ent, repl in [('&amp;','&'),('&lt;','<'),('&gt;','>'),
                      ('&nbsp;',' '),('&ndash;','–'),('&mdash;','—'),
                      ('&quot;','"'),('&#91;','['),('&#93;',']')]:
        text = text.replace(ent, repl)

    # URLs (bare)
    text = RE_URLS.sub(' ', text)

    # Collapse whitespace
    text = RE_SPACES.sub(' ', text)
    text = RE_NEWLINES.sub('\n', text)

    # Quality filter: lowered strictness for Wikipedia text
    sentences = RE_SENTENCES.split(text)
    good = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20: continue # Lowered length threshold to catch shorter facts
        alpha = sum(c.isalpha() for c in s)
        
        # Encyclopedic text has tons of numbers, dates, and punctuation.
        # Lowering the threshold to 40% ensures we don't drop valid data.
        if alpha / (len(s) + 1e-12) > 0.40:
            good.append(s)

    return ' '.join(good)

def iter_wiki_articles(wiki_path: str, n_articles: int = 100000):
    """
    Stream articles from a Wikipedia XML BZ2 file.
    Yields (title, text) pairs.
    """
    if wiki_path.endswith('.bz2'):
        import bz2
        opener = bz2.open
    else:
        opener = open

    count = 0
    elements_scanned = 0
    
    with opener(wiki_path, 'rb') as f:
        # Track both start and end to allow root clearing
        context = ET.iterparse(f, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        for event, elem in context:
            if event == 'end':
                elements_scanned += 1
                
                # Raw scan tracker (runs on every element)
                if elements_scanned % 500000 == 0:
                    print(f"  [Scan] Parsed {elements_scanned:,} raw XML elements...", flush=True)

                tag = elem.tag.split('}')[-1]
                
                # PROCESSING BLOCK: Strictly isolated to <page> tags
                if tag == 'page':
                        title = ""
                        raw = ""
                        
                        for child in elem.iter():
                            ctag = child.tag.split('}')[-1]
                            if ctag == 'title' and child.text:
                                title = child.text
                            elif ctag == 'text' and child.text:
                                raw = child.text 

                        if title and raw and len(raw) >= 200:
                            if ':' not in title or title.startswith('Wikipedia:'):
                                if not title.endswith('(disambiguation)'):
                                    
                                    # Safe slice
                                    cleaned = clean_wikitext(raw[:25000])
                                        
                                    if len(cleaned) > 100:
                                        yield title, cleaned
                                        count += 1

                        # MEMORY WIPE
                        elem.clear()
                        root.clear()

                        if count >= n_articles:
                            break

# ── Engram and address utilities ───────────────────────────────────────────────

def load_engram(path):
    with open(path) as f: e = json.load(f)
    return {
        "ksi_vals": np.array(e["ksi_vals"], dtype=np.float32),
        "j1_360":   np.array(e["j1_pca"], dtype=np.float32) % 360.0,
        "pc1":      np.array(e["pc1"], dtype=np.float32),
        "pc2":      np.array(e["pc2"], dtype=np.float32),
    }

def angular_signed(a, b):
    return float(((b - a + 180) % 360) - 180)

def is_dead_zone(j1):
    return 140 <= (float(j1) % 360) <= 265

def j1_to_sector(j1):
    return int(float(j1) % 360 / 10) % FIELD_J1_BINS

def ksi_to_sector(ksi):
    return min(int(float(ksi) / 0.05), FIELD_KSI_BINS - 1)

# ── Bivector field ─────────────────────────────────────────────────────────────

def new_field():
    """
    field[j1_sector, ksi_sector] = [sum_ΔJ1, sum_ΔKsi, count, sum_ΔJ1², sum_ΔKsi²]
    Packed as float64 array [36, 20, 5].
    """
    return np.zeros((FIELD_J1_BINS, FIELD_KSI_BINS, 5), dtype=np.float64)

def field_stats(field):
    """Compute per-cell mean and std from accumulated sums."""
    count = field[:, :, 2]
    safe_count = np.where(count > 0, count, 1.0)
    mean_dj1  = field[:, :, 0] / safe_count
    mean_dksi = field[:, :, 1] / safe_count
    # Population std (Welford: var = E[x²] - E[x]²)
    var_dj1  = np.maximum(field[:, :, 3]/safe_count - mean_dj1**2, 0)
    var_dksi = np.maximum(field[:, :, 4]/safe_count - mean_dksi**2, 0)
    return mean_dj1, mean_dksi, np.sqrt(var_dj1), np.sqrt(var_dksi), count

def update_field(field, j1_start, ksi_start, dj1, dksi):
    """Accumulate one arc observation into the field."""
    js = j1_to_sector(j1_start)
    ks = ksi_to_sector(ksi_start)
    field[js, ks, 0] += dj1
    field[js, ks, 1] += dksi
    field[js, ks, 2] += 1
    field[js, ks, 3] += dj1**2
    field[js, ks, 4] += dksi**2

def query_field(field, j1, ksi):
    """Get predicted arc at (J1, Ksi) with confidence weight."""
    mean_dj1, mean_dksi, std_dj1, std_dksi, count = field_stats(field)
    js = j1_to_sector(j1); ks = ksi_to_sector(ksi)
    c = float(count[js, ks])
    if c < 5:  # not enough data — return zero
        return 0.0, 0.0, 0.0
    # Confidence = tanh(count/100) → [0,1], saturates at ~100 obs
    confidence = float(np.tanh(c / 100.0))
    return float(mean_dj1[js, ks]), float(mean_dksi[js, ks]), confidence

def save_field(field, path, metadata=None):
    meta = metadata or {}
    np.savez_compressed(path,
                         field=field,
                         j1_bins=np.array(FIELD_J1_BINS),
                         ksi_bins=np.array(FIELD_KSI_BINS),
                         **{f"meta_{k}": np.array([str(v)]) for k, v in meta.items()})
    print(f"  [field] saved → {path}")

def load_field(path):
    d = np.load(path, allow_pickle=True)
    return d['field']

# ── Subcommand: build ─────────────────────────────────────────────────────────

def cmd_build(args):
    print("\n" + "="*70)
    print("WIKI BIVECTOR FIELD — Building from Wikipedia")
    print("="*70)

    from transformers import GPT2Tokenizer
    print(f"\n  Loading tokenizer...", flush=True)
    tok = GPT2Tokenizer.from_pretrained(args.model)
    engram = load_engram(args.engram)
    ksi_v = engram["ksi_vals"]; j1_v = engram["j1_360"]

    field = new_field()
    n_arts = getattr(args, 'n_articles', 100000)
    skip_dead = not getattr(args, 'keep_dead', False)

    print(f"  Streaming {n_arts} articles from {args.wiki}")
    print(f"  Skip dead-zone arcs: {skip_dead}")

    print("  Pre-computing dead punctuation tokens...", flush=True)
    dead_punct_tokens = set()
    for tid in range(len(ksi_v)):
        if is_dead_zone(j1_v[tid]):
            s = tok.decode([tid]).strip()
            if PUNCT_ONLY_RE.match(s):
                dead_punct_tokens.add(tid)
    print(f"  [Debug] Done pre-computing. Found {len(dead_punct_tokens)} punctuation tokens.", flush=True)

    t0 = time.time()
    total_arcs = 0; skipped = 0
    arc_type_counts = {}

    for title_idx, (title, text) in enumerate(iter_wiki_articles(args.wiki, n_arts)):
        # Tokenize the cleaned text
        try:
            ids = tok.encode(text, add_special_tokens=False)[:512]  # cap per article
        except Exception:
            continue

        for i in range(len(ids) - 1):
            t1, t2 = ids[i], ids[i+1]
            if t1 in dead_punct_tokens or t2 in dead_punct_tokens:
                skipped += 1; continue

            # Token-level filter: skip pure-punctuation tokens in dead zone
            try:
                s1 = tok.decode([t1]); s2 = tok.decode([t2])
            except Exception:
                continue
            if PUNCT_ONLY_RE.match(s1.strip()) and is_dead_zone(j1_v[t1]):
                skipped += 1; continue
            if PUNCT_ONLY_RE.match(s2.strip()) and is_dead_zone(j1_v[t2]):
                skipped += 1; continue

            j1_1 = float(j1_v[t1]); ksi_1 = float(ksi_v[t1])
            j1_2 = float(j1_v[t2]); ksi_2 = float(ksi_v[t2])
            dj1  = angular_signed(j1_1, j1_2)
            dksi = ksi_2 - ksi_1

            # Arc-level filter: skip dead-zone arcs
            if skip_dead and (is_dead_zone(j1_1) or is_dead_zone(j1_2)):
                skipped += 1; continue

            # Classify arc type for statistics
            def in_z(j, lo, hi): return lo <= (j%360) <= hi
            if abs(dj1) < 20 and abs(dksi) < 0.08: atype = "HOVER"
            elif in_z(j1_1,80,140) and in_z(j1_2,80,140): atype = "A_INT"
            elif in_z(j1_1,280,340) and in_z(j1_2,280,340): atype = "B_INT"
            elif in_z(j1_1,80,140) and in_z(j1_2,280,340): atype = "A2B"
            elif in_z(j1_1,280,340) and in_z(j1_2,80,140): atype = "B2A"
            elif abs(dj1) > 120: atype = "PIVOT"
            else: atype = "CROSS"
            arc_type_counts[atype] = arc_type_counts.get(atype, 0) + 1

            update_field(field, j1_1, ksi_1, dj1, dksi)
            total_arcs += 1

        # --- NEW CONSOLE TRACKING & CHECKPOINTING ---
        if title_idx % 100 == 0 and title_idx > 0:
            elapsed = time.time() - t0
            rate = title_idx / elapsed
            print(f"  [Article {title_idx}/{n_arts}] | Arcs: {total_arcs:,} | "
                  f"{rate:.1f} art/s | Skipped: {skipped:,}", flush=True)

        if title_idx % 5000 == 0 and title_idx > 0:
            save_field(field, args.output, {
                "n_articles": title_idx,
                "total_arcs": total_arcs,
                "skipped_arcs": skipped
            })
            print(f"  [Checkpoint] Bivector field securely saved.", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {total_arcs:,} arcs in {elapsed:.1f}s")
    print(f"  Skipped: {skipped:,} dead-zone/punct arcs")
    print(f"\n  Arc type distribution:")
    total = sum(arc_type_counts.values())
    for atype, cnt in sorted(arc_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {atype:<10} {cnt:>8,}  ({100*cnt/total:.1f}%)")

    # Field statistics
    mean_dj1, mean_dksi, std_dj1, std_dksi, count = field_stats(field)
    populated = (count > 5).sum()
    print(f"\n  Populated cells: {populated}/{FIELD_J1_BINS*FIELD_KSI_BINS} "
          f"({100*populated/(FIELD_J1_BINS*FIELD_KSI_BINS):.0f}%)")
    print(f"  Mean arc (all cells): ΔJ1={mean_dj1[count>5].mean():.2f}°  "
          f"ΔKsi={mean_dksi[count>5].mean():.4f}")

    # Show the field (ASCII heatmap of count)
    print(f"\n  Count heatmap (J1 sectors × Ksi bands, log scale):")
    log_count = np.log1p(count)
    mx = log_count.max()
    chars = ' ░▒▓█'
    print("  J1\\Ksi " + " ".join(f"{i*0.05:.2f}" for i in range(0, 20, 2)))
    for j in range(0, 36, 2):
        row = ""
        for k in range(0, 20, 1):
            idx = int(log_count[j, k] / (mx + 1e-12) * (len(chars)-1))
            row += chars[idx]
        j1_label = f"{j*10:3d}°"
        print(f"  {j1_label}   {row}")

    save_field(field, args.output, {
        "n_articles": total_arcs // 100,
        "total_arcs": total_arcs,
        "skipped_arcs": skipped,
    })

# ── Subcommand: augment ───────────────────────────────────────────────────────

def cmd_augment(args):
    """Add plain text files to an existing field."""
    print("\n" + "="*70)
    print("WIKI BIVECTOR FIELD — Augmenting from Text Files")
    print("="*70)

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(args.model)
    engram = load_engram(args.engram)
    ksi_v = engram["ksi_vals"]; j1_v = engram["j1_360"]

    field_path = getattr(args, 'field', None)
    field = load_field(field_path) if field_path and Path(field_path).exists() else new_field()
    pre_count = int(field[:,:,2].sum())
    print(f"  Field loaded: {pre_count:,} existing arcs")

    for text_path in args.texts:
        if not Path(text_path).exists():
            print(f"  [skip] {text_path}"); continue
        print(f"  Adding: {text_path}")
        with open(text_path, encoding='utf-8', errors='replace') as f:
            lines = [l.strip() for l in f if len(l.strip()) > 30]
        n_added = 0
        for line in lines[:getattr(args,'max_lines',50000)]:
            ids = tok.encode(line, add_special_tokens=False)
            for i in range(len(ids)-1):
                t1, t2 = ids[i], ids[i+1]
                if t1 >= len(ksi_v) or t2 >= len(ksi_v): continue
                j1_1=float(j1_v[t1]); ksi_1=float(ksi_v[t1])
                j1_2=float(j1_v[t2]); ksi_2=float(ksi_v[t2])
                if is_dead_zone(j1_1) or is_dead_zone(j1_2): continue
                update_field(field, j1_1, ksi_1,
                             angular_signed(j1_1, j1_2), ksi_2-ksi_1)
                n_added += 1
        print(f"    Added {n_added:,} arcs")

    post_count = int(field[:,:,2].sum())
    print(f"  Total arcs: {pre_count:,} → {post_count:,} (+{post_count-pre_count:,})")
    save_field(field, args.output)

# ── Subcommand: generate ──────────────────────────────────────────────────────

def cmd_generate(args):
    """
    Generation using the bivector field for arc prediction.
    Replaces bigram_db lookup with field query — position-complete.
    """
    print("\n" + "="*70)
    print("WIKI BIVECTOR FIELD — Generation")
    print("="*70)

    # Import wandering model infrastructure
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"  Loading model: {args.model}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model, local_files_only=True,
                                             low_cpu_mem_usage=True)
    model.eval()
    tok = GPT2Tokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    n_layers = model.config.n_layer

    with open(args.engram) as f: e = json.load(f)
    pc1 = np.array(e["pc1"], dtype=np.float32)
    pc2 = np.array(e["pc2"], dtype=np.float32)
    U_ref = np.array(e["U_ref"], dtype=np.float32)

    with open(args.baseline_file) as f: bl = json.load(f)
    baseline = bl["baseline"]
    delta_end = getattr(args, 'delta_end', 0.08)
    arc_strength = getattr(args, 'arc_strength', 0.10)

    field = load_field(args.field)
    total_arcs = int(field[:,:,2].sum())
    populated  = int((field[:,:,2] > 5).sum())
    print(f"  Field: {total_arcs:,} arcs, {populated}/720 cells populated")

    # Extract SVDs (from wandering_model.py)
    print("  Extracting SVDs...", flush=True)
    svds = []
    for i in range(n_layers):
        w = model.transformer.h[i].mlp.c_fc.weight.data.float().numpy()
        U, _, _ = np.linalg.svd(w, full_matrices=False)
        svds.append(U[:, :K_PROJ].astype(np.float32))

    # Build schedule
    t = np.linspace(0,1,n_layers)
    dp = (-0.02 + delta_end) / 2.
    deltas = (1-t)**2*(-0.02) + 2*(1-t)*t*dp + t**2*delta_end
    schedule = (np.array(baseline) + deltas).tolist()

    def h_ksi(h, V_top):
        z = V_top.T @ h; z2=z*z; S=float(z2.sum())
        if S < 1e-12: return 0.5
        p=z2/S; ps=np.where(p>1e-15,p,1e-15)
        return float(np.clip(-np.sum(p*np.log(ps))/np.log(K_PROJ),0,1))

    def correct_ksi_fast(h, V_top, tgt, cur, ss=0.05, ns=3):
        d=tgt-cur
        if abs(d)<5e-5: return h
        z=V_top.T@h; hp=V_top@z; hr=h-hp; pn=float(np.linalg.norm(hp))
        sign=float(np.sign(d))
        for _ in range(ns):
            z2=z*z; S=float(z2.sum())
            if S<1e-12: break
            p=z2/S; ps=np.where(p>1e-15,p,1e-15); H=float(-np.sum(p*np.log(ps)))
            g=-(2.*z)/(S*np.log(K_PROJ))*(np.log(ps)+H); gn=float(np.linalg.norm(g))
            if gn<1e-12: break
            z=z+sign*ss*g/gn
        np_=V_top@z; nn=float(np.linalg.norm(np_))
        if nn>1e-8 and pn>1e-8: z=z*(pn/nn)
        return (V_top@z+hr).astype(np.float32)

    def h_j1(h):
        h_n=h/(np.linalg.norm(h)+1e-12)
        return float(np.degrees(np.arctan2(np.dot(h_n,pc1), np.dot(h_n,pc2)))%360)

    def nudge(h, dj1, alpha):
        if abs(dj1)<0.5 or alpha<0.01: return h
        h_n=h/(np.linalg.norm(h)+1e-12)
        a=float(np.dot(h_n,pc1)); b=float(np.dot(h_n,pc2))
        j1=(np.degrees(np.arctan2(a,b)))%360
        j1t=(j1+np.clip(dj1,-40,40))%360    # cap at ±40° per step
        rt=np.radians(j1t)
        an=(1-alpha)*a+alpha*np.cos(rt); bn=(1-alpha)*b+alpha*np.sin(rt)
        nm=np.sqrt(an**2+bn**2)
        if nm<1e-12: return h
        on=np.sqrt(a**2+b**2)
        hp=a*pc1+b*pc2; hr=h-hp
        return ((an/nm*on)*pc1+(bn/nm*on)*pc2+hr).astype(np.float32)

    current_tok = [None]
    ksi_traj = []; arc_traj = []

    def make_hook(li):
        def fn(mod, args_, out):
            hb = out[0]; h = hb[0,-1,:].float().cpu().numpy().astype(np.float32)
            V = svds[li]; tgt = schedule[li]
            ksi_pre = h_ksi(h, V)

            # Field-based arc nudge
            j1_now = h_j1(h)
            ksi_now = ksi_pre
            pred_dj1, pred_dksi, conf = query_field(field, j1_now, ksi_now)

            lyr_alpha = arc_strength * max(0., 1. - li/(n_layers*1.5))
            weighted_alpha = lyr_alpha * conf
            if weighted_alpha > 0.01:
                h = nudge(h, pred_dj1, weighted_alpha)

            h = correct_ksi_fast(h, V, tgt, ksi_pre)
            ksi_traj.append(h_ksi(h, V))
            arc_traj.append((float(pred_dj1), float(conf), float(j1_now)))

            ht = torch.tensor(h, dtype=hb.dtype, device=hb.device)
            hb = hb.clone(); hb[0,-1,:] = ht
            return (hb,)+out[1:] if isinstance(out,tuple) else hb
        return fn

    hooks = [model.transformer.h[i].register_forward_hook(make_hook(i))
             for i in range(n_layers)]

    input_ids = tok(args.prompt, return_tensors="pt")["input_ids"]
    current_tok[0] = int(input_ids[0,-1])
    generated = input_ids.clone()
    n_tok = 0; max_tok = getattr(args,'max_tokens',120)
    temperature = getattr(args,'temperature',0.9)

    print(f"\n  Prompt: '{args.prompt}'")
    print(f"  delta_end={delta_end}  arc_strength={arc_strength}")
    print(f"\n  ", end="", flush=True)

    with torch.no_grad():
        for _ in range(max_tok):
            out = model(generated)
            logits = out.logits[0,-1,:] / temperature
            v, _ = torch.topk(logits, 50)
            logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nt = torch.multinomial(probs, 1).unsqueeze(0)
            generated = torch.cat([generated, nt], dim=1)
            current_tok[0] = int(nt[0,0])
            print(tok.decode([int(nt[0,0])], skip_special_tokens=False),
                  end="", flush=True)
            n_tok += 1
            if int(nt[0,0]) == tok.eos_token_id: break

    for h in hooks: h.remove()
    print("\n")

    text = tok.decode(generated[0], skip_special_tokens=True)
    mean_ksi = float(np.mean(ksi_traj)) if ksi_traj else 0.
    mean_conf = float(np.mean([a[1] for a in arc_traj])) if arc_traj else 0.
    print(f"  Ksi mean: {mean_ksi:.4f}  arc_confidence: {mean_conf:.3f}")

    result = {"generated_text": text, "n_tokens": n_tok,
              "mean_ksi": mean_ksi, "arc_confidence": mean_conf,
              "delta_end": float(delta_end), "arc_strength": float(arc_strength)}
    with open(args.output,'w') as f:
        json.dump(result, f, indent=2, default=lambda o:
            float(o) if isinstance(o,(np.float64,np.float32)) else
            int(o) if isinstance(o,(np.int64,np.int32,np.intp)) else None)
    print(f"  Saved → {args.output}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Wikipedia Bivector Field Builder")
    sub = ap.add_subparsers(dest="cmd")

    pb = sub.add_parser("build")
    pb.add_argument("--wiki",        required=True)
    pb.add_argument("--model",       required=True)
    pb.add_argument("--engram",      required=True)
    pb.add_argument("--output",      default="bivector_field.npz")
    pb.add_argument("--n_articles",  type=int, default=100000)
    pb.add_argument("--keep_dead",   action="store_true")

    pa = sub.add_parser("augment")
    pa.add_argument("--field",       required=True)
    pa.add_argument("--model",       required=True)
    pa.add_argument("--engram",      required=True)
    pa.add_argument("--texts",       nargs="+", required=True)
    pa.add_argument("--output",      default="bivector_field_aug.npz")
    pa.add_argument("--max_lines",   type=int, default=50000)

    pg = sub.add_parser("generate")
    pg.add_argument("--model",         required=True)
    pg.add_argument("--engram",        required=True)
    pg.add_argument("--baseline_file", required=True)
    pg.add_argument("--field",         required=True)
    pg.add_argument("--prompt",        default="The cause of altruistic behavior is")
    pg.add_argument("--delta_end",     type=float, default=0.08)
    pg.add_argument("--arc_strength",  type=float, default=0.10)
    pg.add_argument("--max_tokens",    type=int,   default=120)
    pg.add_argument("--temperature",   type=float, default=0.9)
    pg.add_argument("--output",        default="wiki_gen.json")

    args = ap.parse_args()
    {"build": cmd_build, "augment": cmd_augment, "generate": cmd_generate
     }.get(args.cmd, lambda _: (ap.print_help(), sys.exit(1)))(args)

if __name__ == "__main__":
    main()
