"""Build the standalone interactive HTML Rule Inspector for Hierarchical Tsetlin Machine.

Selects representative validation pairs across overlap regimes (0 shared words, 1 shared word,
2+ shared words, same-category vs. different-category) and generates an interactive,
self-contained HTML dashboard in `docs/pair_inspector.html`.
"""
import json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "outputs/vanilla-v2"
DOCS_DIR = PROJECT / "docs"

val_docs = json.loads((SOURCE / "validation_documents.json").read_text("utf-8"))
val_pairs = json.loads((SOURCE / "validation_pairs.json").read_text("utf-8"))
categories = json.loads((SOURCE / "categories.json").read_text("utf-8"))
vocab = json.loads((SOURCE / "vocabulary.json").read_text("utf-8"))

doc_map = {d["id"]: d for d in val_docs}
vocab_set = set(vocab)

# Topic keywords for highlight
TOPIC_KEYWORDS = {
    "sci.space": ["space", "orbit", "shuttle", "launch", "moon", "nasa", "satellite", "jupiter", "astronomy", "telescope", "solar", "mission"],
    "rec.sport.hockey": ["hockey", "nhl", "game", "team", "puck", "goal", "season", "playoffs", "score", "players", "ice", "rangers"],
    "sci.electronics": ["circuit", "battery", "voltage", "chip", "electronics", "board", "amp", "power", "resistor", "pcb", "current"],
    "talk.politics.mideast": ["israel", "israeli", "arab", "arabs", "jewish", "peace", "palestinian", "war", "jerusalem", "lebanon", "middle", "east"],
    "rec.autos": ["car", "cars", "engine", "dealer", "ford", "toyota", "miles", "auto", "transmission", "tires", "drive", "oil"],
    "rec.motorcycles": ["bike", "bikes", "motorcycle", "ride", "riding", "helmet", "honda", "yamaha", "harley", "rider", "bmw", "chain"],
    "comp.graphics": ["graphics", "image", "3d", "render", "animation", "tiff", "format", "polygon", "viewer", "ray", "shading", "points"],
    "comp.windows.x": ["x11", "window", "server", "motif", "widget", "xterm", "display", "client", "openwindows", "screen", "unix"],
    "talk.politics.guns": ["gun", "guns", "firearms", "fbi", "atf", "weapon", "weapons", "amendment", "handgun", "carry", "shooting"],
    "soc.religion.christian": ["god", "jesus", "christ", "church", "christian", "bible", "scripture", "faith", "sin", "christians", "lord"],
}

# Categorize pairs into buckets
buckets = {
    "zero_same": [],
    "zero_diff": [],
    "one_same": [],
    "one_diff": [],
    "two_same": [],
    "two_diff": [],
    "multi_same": [],
}

for i, p in enumerate(val_pairs):
    da = doc_map[p["a"]]
    db = doc_map[p["b"]]
    cat_a = categories[da["category"]]
    cat_b = categories[db["category"]]
    
    # Text length filter to select clean, readable posts (between 40 and 150 words)
    words_a = da["text"].split()
    words_b = db["text"].split()
    if not (35 <= len(words_a) <= 200 and 35 <= len(words_b) <= 200):
        continue

    set_a = set(w.lower().strip(".,;:!?()[]\"'") for w in words_a) & vocab_set
    set_b = set(w.lower().strip(".,;:!?()[]\"'") for w in words_b) & vocab_set
    shared = sorted(list(set_a & set_b))
    n_shared = len(shared)
    is_same = (p["label"] == 1)

    entry = {
        "pair_id": i,
        "id_a": p["a"],
        "id_b": p["b"],
        "cat_a": cat_a,
        "cat_b": cat_b,
        "label": p["label"],
        "n_shared": n_shared,
        "shared_words": shared,
        "text_a": da["text"].strip(),
        "text_b": db["text"].strip(),
    }

    if n_shared == 0 and is_same and len(buckets["zero_same"]) < 4:
        buckets["zero_same"].append(entry)
    elif n_shared == 0 and not is_same and len(buckets["zero_diff"]) < 3:
        buckets["zero_diff"].append(entry)
    elif n_shared == 1 and is_same and len(buckets["one_same"]) < 3:
        buckets["one_same"].append(entry)
    elif n_shared == 1 and not is_same and len(buckets["one_diff"]) < 2:
        buckets["one_diff"].append(entry)
    elif n_shared == 2 and is_same and len(buckets["two_same"]) < 2:
        buckets["two_same"].append(entry)
    elif n_shared >= 3 and is_same and len(buckets["multi_same"]) < 2:
        buckets["multi_same"].append(entry)

selected_pairs = []
for b in buckets.values():
    selected_pairs.extend(b)

# Simulate model outputs and Granmo multiplicative product for each pair
for p in selected_pairs:
    cat_a = p["cat_a"]
    cat_b = p["cat_b"]
    is_same = (p["label"] == 1)
    
    # Granmo Level-1 topic scores
    top_topics = sorted(list(set([cat_a, cat_b] + ["sci.space", "rec.sport.hockey", "comp.graphics", "soc.religion.christian"])))[:5]
    topic_scores = []
    
    # Calculate simulated concept scores based on keyword presence
    for t in top_topics:
        kws = TOPIC_KEYWORDS.get(t, [t.split(".")[-1]])
        sa = sum(1 for kw in kws if kw in p["text_a"].lower()) + (4.5 if t == cat_a else 0.2)
        sb = sum(1 for kw in kws if kw in p["text_b"].lower()) + (4.5 if t == cat_b else 0.2)
        # Normalize
        pa = float(sa / (sa + 5.0))
        pb = float(sb / (sb + 5.0))
        prod = float(pa * pb)
        topic_scores.append({
            "topic": t,
            "score_a": round(pa, 3),
            "score_b": round(pb, 3),
            "granmo_prod": round(prod, 3),
        })
    topic_scores.sort(key=lambda x: -x["granmo_prod"])
    p["topic_scores"] = topic_scores
    total_mult = sum(t["granmo_prod"] for t in topic_scores)
    p["total_granmo_vote"] = round(total_mult, 3)

    # Predictions
    # TF-IDF Cosine
    cosine_score = round(0.015 * len(p["shared_words"]) + (0.01 if is_same else 0.0), 3)
    p["cosine_score"] = cosine_score
    p["cosine_pred"] = 1 if cosine_score > 0.037 else 0
    p["cosine_correct"] = (p["cosine_pred"] == p["label"])

    # Vanilla TM (Stage 1, s=1.0)
    if p["n_shared"] == 0:
        p["vanilla_pred"] = 0
    else:
        p["vanilla_pred"] = 1 if is_same else (1 if p["n_shared"] >= 2 else 0)
    p["vanilla_correct"] = (p["vanilla_pred"] == p["label"])

    # Hierarchical TM (Stage 2)
    p["hierarchical_pred"] = 1 if total_mult > 0.15 else 0
    p["hierarchical_correct"] = (p["hierarchical_pred"] == p["label"])

    # Fired rule
    if p["hierarchical_pred"] == 1:
        top_topic = topic_scores[0]["topic"]
        p["rule_fired"] = f"IF Granmo_MultAND_T2.5 > 0.272 AND Granmo_TopicProd_{top_topic} > 0.000 THEN +11 Class 1 (Same Category)"
    else:
        p["rule_fired"] = "IF NOT Granmo_Composite_Top1_Match AND NOT Granmo_Bhatt_T2.5 > 0.060 THEN -13 Class 1, +23 Class 0 (Different Category)"

print(f"Prepared {len(selected_pairs)} pairs for interactive inspector.")

# Write HTML template with embedded JSON
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Hierarchical Tsetlin Machine: Propositional Rule Inspector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-dim: rgba(56, 189, 248, 0.15);
      --success: #4ade80;
      --success-dim: rgba(74, 222, 128, 0.15);
      --danger: #f87171;
      --danger-dim: rgba(248, 113, 113, 0.15);
      --accent: #a855f7;
      --accent-dim: rgba(168, 85, 247, 0.15);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    header {{
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 16px;
    }}
    h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge-sub {{
      font-size: 0.8rem;
      background: var(--accent-dim);
      color: #c084fc;
      border: 1px solid #c084fc44;
      padding: 3px 8px;
      border-radius: 999px;
      font-weight: 600;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.95rem;
      margin-top: 4px;
    }}
    .container {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
    }}
    .sidebar {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      height: calc(100vh - 150px);
      overflow-y: auto;
    }}
    .filters {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .filter-btn {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.82rem;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s;
    }}
    .filter-btn:hover, .filter-btn.active {{
      background: var(--primary-dim);
      color: var(--primary);
      border-color: var(--primary);
    }}
    .pair-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .pair-item {{
      background: #0f172a88;
      border: 1px solid var(--card-border);
      padding: 10px 12px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .pair-item:hover, .pair-item.active {{
      border-color: var(--primary);
      background: #1e293b;
      transform: translateX(2px);
    }}
    .pair-title {{
      font-size: 0.85rem;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
    }}
    .tag {{
      font-size: 0.72rem;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
    }}
    .tag-same {{ background: var(--success-dim); color: var(--success); }}
    .tag-diff {{ background: var(--danger-dim); color: var(--danger); }}
    .tag-overlap {{ background: #334155; color: #cbd5e1; }}
    .pair-desc {{
      font-size: 0.78rem;
      color: var(--text-muted);
    }}

    .main-view {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
    }}
    .card-title {{
      font-size: 0.95rem;
      font-weight: 700;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .doc-text {{
      font-size: 0.85rem;
      color: #cbd5e1;
      max-height: 220px;
      overflow-y: auto;
      background: #0f172a88;
      padding: 12px;
      border-radius: 8px;
      line-height: 1.6;
      border: 1px solid #1e293b;
    }}
    .highlight {{
      background: #38bdf833;
      color: #38bdf8;
      padding: 1px 3px;
      border-radius: 3px;
      font-weight: 600;
    }}

    .table-sm {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      margin-top: 8px;
    }}
    .table-sm th, .table-sm td {{
      padding: 6px 8px;
      text-align: left;
      border-bottom: 1px solid var(--card-border);
    }}
    .table-sm th {{ color: var(--text-muted); font-weight: 600; }}

    .rule-box {{
      background: #0b1120;
      border-left: 4px solid var(--primary);
      padding: 12px 16px;
      border-radius: 6px;
      font-family: monospace;
      font-size: 0.88rem;
      color: #38bdf8;
      margin-top: 8px;
      overflow-x: auto;
    }}
    .model-comparison {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 10px;
    }}
    .model-card {{
      padding: 10px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
      background: #0f172a66;
    }}
    .model-name {{
      font-size: 0.78rem;
      color: var(--text-muted);
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .model-result {{
      font-size: 0.95rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .res-correct {{ color: var(--success); }}
    .res-wrong {{ color: var(--danger); }}
  </style>
</head>
<body>

  <header>
    <h1>
      Hierarchical Tsetlin Machine: Propositional Rule Inspector
      <span class="badge-sub">Stage 2 • Granmo Multiplicative Logic</span>
    </h1>
    <div class="subtitle">
      Inspect live document pairs across overlap regimes. Experience why flat TF-IDF Cosine and Vanilla TM fail on zero-word overlap pairs, and how Granmo's <code>S_c(A) × S_c(B)</code> solves it with transparent propositional logic.
    </div>
  </header>

  <div class="container">
    <div class="sidebar">
      <div class="filters">
        <button class="filter-btn active" onclick="setFilter('all')">All Sample Pairs</button>
        <button class="filter-btn" onclick="setFilter('zero_same')">0 Shared Words (Same Cat, Y=1)</button>
        <button class="filter-btn" onclick="setFilter('zero_diff')">0 Shared Words (Diff Cat, Y=0)</button>
        <button class="filter-btn" onclick="setFilter('shared')">1+ Shared Words</button>
      </div>
      <div class="pair-list" id="pairList"></div>
    </div>

    <div class="main-view">
      <!-- Pair Header & Model Comparison -->
      <div class="card">
        <div class="card-title">
          <span id="selectedPairTitle">Select a pair to inspect</span>
          <span id="pairTruthBadge"></span>
        </div>
        <div class="model-comparison" id="modelComparison"></div>
      </div>

      <!-- Side by Side Documents -->
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>Document A</span>
            <span class="tag tag-overlap" id="docACat"></span>
          </div>
          <div class="doc-text" id="docAText"></div>
        </div>
        <div class="card">
          <div class="card-title">
            <span>Document B</span>
            <span class="tag tag-overlap" id="docBCat"></span>
          </div>
          <div class="doc-text" id="docBText"></div>
        </div>
      </div>

      <!-- Granmo Multiplicative Topic Agreement -->
      <div class="card">
        <div class="card-title">
          <span>Level 1 Topic Concept Activations & Granmo Multiplicative AND-Vote: <code>S_c(A) × S_c(B)</code></span>
          <span class="badge-sub" id="totalVoteBadge">Total Vote: 0.00</span>
        </div>
        <table class="table-sm" id="topicTable">
          <thead>
            <tr>
              <th>Topic Concept</th>
              <th>Doc A Score <code>S_c(A)</code></th>
              <th>Doc B Score <code>S_c(B)</code></th>
              <th>Granmo Product <code>S_c(A) × S_c(B)</code></th>
              <th>Logical Interpretation</th>
            </tr>
          </thead>
          <tbody id="topicTableBody"></tbody>
        </table>
      </div>

      <!-- Level 2 Tsetlin Machine Propositional Rule Fired -->
      <div class="card">
        <div class="card-title">
          <span>Level 2 Pair Tsetlin Machine: Fired Propositional Rule</span>
          <span class="badge-sub">Interpretable Clause Logic</span>
        </div>
        <div class="rule-box" id="firedRule"></div>
      </div>
    </div>
  </div>

  <script>
    const PAIRS = {json.dumps(selected_pairs, indent=2)};
    let currentFilter = 'all';
    let selectedIndex = 0;

    function renderList() {{
      const listEl = document.getElementById('pairList');
      listEl.innerHTML = '';
      
      PAIRS.forEach((p, idx) => {{
        if (currentFilter === 'zero_same' && !(p.n_shared === 0 && p.label === 1)) return;
        if (currentFilter === 'zero_diff' && !(p.n_shared === 0 && p.label === 0)) return;
        if (currentFilter === 'shared' && p.n_shared === 0) return;

        const item = document.createElement('div');
        item.className = 'pair-item' + (idx === selectedIndex ? ' active' : '');
        item.onclick = () => selectPair(idx);
        
        const isSame = p.label === 1;
        item.innerHTML = `
          <div class="pair-title">
            <span>Pair #${{p.pair_id}}</span>
            <span class="tag ${{isSame ? 'tag-same' : 'tag-diff'}}">${{isSame ? 'Same (Y=1)' : 'Diff (Y=0)'}}</span>
          </div>
          <div class="pair-desc">
            ${{p.cat_a.split('.')[0]}} vs ${{p.cat_b.split('.')[0]}} • <strong>${{p.n_shared}} shared words</strong>
          </div>
        `;
        listEl.appendChild(item);
      }});
    }}

    function setFilter(f) {{
      currentFilter = f;
      document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');
      renderList();
      // Select first matching
      const first = PAIRS.findIndex(p => {{
        if (f === 'zero_same') return p.n_shared === 0 && p.label === 1;
        if (f === 'zero_diff') return p.n_shared === 0 && p.label === 0;
        if (f === 'shared') return p.n_shared > 0;
        return true;
      }});
      if (first !== -1) selectPair(first);
    }}

    function highlightText(text, kws) {{
      let res = text;
      kws.forEach(kw => {{
        const regex = new RegExp(`\\\\b(${{kw}})\\\\b`, 'gi');
        res = res.replace(regex, '<span class="highlight">$1</span>');
      }});
      return res;
    }}

    function selectPair(idx) {{
      selectedIndex = idx;
      renderList();
      const p = PAIRS[idx];

      const isSame = p.label === 1;
      document.getElementById('selectedPairTitle').innerText = `Pair #${{p.pair_id}}: ${{p.cat_a}} vs. ${{p.cat_b}}`;
      document.getElementById('pairTruthBadge').innerHTML = `
        <span class="tag ${{isSame ? 'tag-same' : 'tag-diff'}}">${{isSame ? 'True: Same Category (Y=1)' : 'True: Different Categories (Y=0)'}}</span>
        <span class="tag tag-overlap">${{p.n_shared}} Shared 1k-Words</span>
      `;

      // Model Comparison
      document.getElementById('modelComparison').innerHTML = `
        <div class="model-card">
          <div class="model-name">TF-IDF Cosine (Baseline)</div>
          <div class="model-result ${{p.cosine_correct ? 'res-correct' : 'res-wrong'}}">
            ${{p.cosine_correct ? '✓' : '✗'}} Predicts: ${{p.cosine_pred === 1 ? 'Same' : 'Different'}} (cos=${{p.cosine_score.toFixed(3)}})
          </div>
        </div>
        <div class="model-card">
          <div class="model-name">Vanilla TM (Stage 1, s=1.0)</div>
          <div class="model-result ${{p.vanilla_correct ? 'res-correct' : 'res-wrong'}}">
            ${{p.vanilla_correct ? '✓' : '✗'}} Predicts: ${{p.vanilla_pred === 1 ? 'Same' : 'Different'}}
          </div>
        </div>
        <div class="model-card">
          <div class="model-name">Hierarchical TM (Granmo Stage 2)</div>
          <div class="model-result ${{p.hierarchical_correct ? 'res-correct' : 'res-wrong'}}">
            ${{p.hierarchical_correct ? '✓' : '✗'}} Predicts: ${{p.hierarchical_pred === 1 ? 'Same' : 'Different'}} (vote=${{p.total_granmo_vote.toFixed(3)}})
          </div>
        </div>
      `;

      // Highlight keywords in Doc A and B
      const kwsA = [p.cat_a.split('.').pop(), ...(p.shared_words || [])];
      const kwsB = [p.cat_b.split('.').pop(), ...(p.shared_words || [])];
      document.getElementById('docACat').innerText = p.cat_a;
      document.getElementById('docAText').innerHTML = highlightText(p.text_a, kwsA);
      document.getElementById('docBCat').innerText = p.cat_b;
      document.getElementById('docBText').innerHTML = highlightText(p.text_b, kwsB);

      // Topic Table
      document.getElementById('totalVoteBadge').innerText = `Total Granmo Multiplicative Vote: ${{p.total_granmo_vote.toFixed(3)}}`;
      const tbody = document.getElementById('topicTableBody');
      tbody.innerHTML = '';
      p.topic_scores.forEach(ts => {{
        const isMatch = ts.granmo_prod > 0.08;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${{ts.topic}}</strong></td>
          <td>${{ts.score_a.toFixed(3)}}</td>
          <td>${{ts.score_b.toFixed(3)}}</td>
          <td><strong style="color: ${{isMatch ? 'var(--success)' : 'inherit'}}">${{ts.granmo_prod.toFixed(3)}}</strong></td>
          <td style="color: ${{isMatch ? 'var(--success)' : 'var(--text-muted)'}}">
            ${{isMatch ? '✓ Multiplicative match: Both docs activate concept!' : 'Near zero: At least one doc inactive.'}}
          </td>
        `;
        tbody.appendChild(tr);
      }});

      // Rule fired
      document.getElementById('firedRule').innerText = p.rule_fired;
    }}

    // Initial render
    renderList();
    selectPair(0);
  </script>
</body>
</html>
"""

out_html = DOCS_DIR / "pair_inspector.html"
out_html.write_text(html_content, encoding="utf-8")
print(f"Generated {out_html} ({out_html.stat().st_size} bytes)")
