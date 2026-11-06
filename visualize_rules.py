"""Summarise and plot the learned clauses of a vanilla TM run.

  python visualize_rules.py outputs/vanilla-v2/validation_rules_all.txt docs/figures

Reads the full rules file written by run.py (one clause per line, `IF a AND b THEN +1 0`).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse(path):
    clauses = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        m = re.match(r'IF (.*) THEN (.*)$', line)
        if not m:
            continue
        lits = [] if m.group(1) == '(always)' else m.group(1).split(' AND ')
        votes = {}
        for part in m.group(2).split(', '):
            w, cls = part.split()
            votes[int(cls)] = int(w)
        # The export omits zero weights. A -1 vote for class 0 favours class 1.
        margin = votes.get(1, 0) - votes.get(0, 0)
        supports = None if margin == 0 else int(margin > 0)
        clauses.append({'literals': lits, 'supports': supports,
                        'negated': [l[4:] for l in lits if l.startswith('NOT ')],
                        'positive': [l for l in lits if not l.startswith('NOT ')]})
    return clauses


def summarise(clauses):
    lengths = [len(clause['literals']) for clause in clauses]
    neg_features = Counter(literal.partition('.')[0] for clause in clauses for literal in clause['negated'])
    out = {'clauses': len(clauses),
           'grouping': 'direction of the fired vote: weight[1] minus weight[0]',
           'neutral_clauses': sum(clause['supports'] is None for clause in clauses),
           'median_literals': float(np.median(lengths)) if lengths else 0,
           'mean_literals': float(np.mean(lengths)) if lengths else 0,
           'negated_by_feature_group': dict(sorted(neg_features.items()))}
    for cls, label in ((1, 'same-category'), (0, 'different-category')):
        cs = [c for c in clauses if c['supports'] == cls]
        sizes = [len(c['literals']) for c in cs]
        neg = sum(len(c['negated']) for c in cs); pos = sum(len(c['positive']) for c in cs)
        out[label] = {'clauses': len(cs), 'empty': sum(s == 0 for s in sizes),
                      'median_literals': float(np.median(sizes)) if sizes else 0,
                      'positive_literals': pos, 'negated_literals': neg,
                      'negated_share': neg / max(1, neg + pos)}
    return out


def plot(clauses, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    colors = {1: '#16a34a', 0: '#64748b'}
    names = {1: 'Firing favours same category', 0: 'Firing favours different categories'}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(len(c['literals']) for c in clauses) + 1, 25)
    for cls in (0, 1):
        ax.hist([len(c['literals']) for c in clauses if c['supports'] == cls], bins=bins,
                alpha=.7, color=colors[cls], label=names[cls])
    ax.set_xlabel('Literals (conditions) in one clause'); ax.set_ylabel('Number of clauses')
    ax.set_title('How long are the learned clauses?'); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / 'rules_length.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos = sum(len(c['positive']) for c in clauses); neg = sum(len(c['negated']) for c in clauses)
    bars = ax.bar(['Positive feature\nconditions', 'Negated feature\nconditions'], [pos, neg],
                  color=['#16a34a', '#dc2626'])
    for b, v in zip(bars, (pos, neg)):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom')
    ax.set_ylabel('Count across all clauses')
    ax.set_title('Literal signs in the learned clauses')
    fig.tight_layout(); fig.savefig(outdir / 'rules_not_vs_present.png', dpi=170); plt.close(fig)

    # Words the clauses that support "same category" ask for (positive literals only).
    cnt = Counter(l for c in clauses if c['supports'] == 1 for l in c['positive'])
    top = cnt.most_common(15)
    if top:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([t[0] for t in reversed(top)], [t[1] for t in reversed(top)], color='#16a34a')
        ax.set_xlabel('Clauses that require it'); ax.set_title('Most-required "present" conditions (same-category clauses)')
        fig.tight_layout(); fig.savefig(outdir / 'rules_top_present.png', dpi=170); plt.close(fig)
    return top


if __name__ == '__main__':
    clauses = parse(sys.argv[1])
    outdir = sys.argv[2] if len(sys.argv) > 2 else 'docs/figures'
    top = plot(clauses, outdir)
    summary = summarise(clauses); summary['top_present'] = top
    Path(outdir, '..', 'results').resolve().mkdir(parents=True, exist_ok=True)
    Path(outdir, '..', 'results', 'rules_summary.json').resolve().write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
