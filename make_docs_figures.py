"""Draw the comparison charts used in docs/ from measured results.

  python make_docs_figures.py outputs/vanilla-v2/validation_metrics.json

Needs docs/results/ablation.json and docs/results/sweep.json (from sweep.py).
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIG = Path('docs/figures'); FIG.mkdir(parents=True, exist_ok=True)
RES = Path('docs/results')


def barh_with_error(rows, title, fname, cosine_acc, highlight=None):
    labels = [r['name'] for r in rows][::-1]
    means = [r['acc_mean'] for r in rows][::-1]
    errs = [r['acc_std'] for r in rows][::-1]
    colors = ['#16a34a' if (highlight and l.startswith(highlight)) else '#38bdf8' for l in labels]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 2))
    ax.barh(labels, means, xerr=errs, color=colors, capsize=3)
    ax.axvline(0.5, color='#64748b', ls='--', label='chance / majority 0.50')
    ax.axvline(cosine_acc, color='#dc2626', ls='-', label=f'TF-IDF cosine {cosine_acc:.3f}')
    ax.set_xlim(0.45, max(0.65, max(means) + 0.03))
    ax.set_xlabel('Validation accuracy (mean of 3 seeds, bars = standard deviation)')
    ax.set_title(title); ax.legend(loc='lower right')
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(m + e + 0.004, i, f'{m:.3f}', va='center', fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=170); plt.close(fig)


def t_curve(cosine_acc):
    """Accuracy against T, joining sweep2 (T=5..25) with sweep (T=50, 100)."""
    s1 = json.loads((RES / 'sweep.json').read_text(encoding='utf-8'))
    s2 = json.loads((RES / 'sweep2.json').read_text(encoding='utf-8'))
    pts = {}
    for r in s2:
        if r['clauses'] == 100:
            pts[r['threshold']] = r
    for r in s1:
        if r['clauses'] == 100 and r['vocabulary'] == 1000 and r['threshold'] in (50, 100):
            pts[r['threshold']] = r
    ts = sorted(pts)
    m = np.array([pts[t]['acc_mean'] for t in ts]); e = np.array([pts[t]['acc_std'] for t in ts])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(ts, m, yerr=e, marker='o', color='#16a34a', capsize=4, label='Vanilla TM (mean of 3 seeds)')
    ax.axhline(cosine_acc, color='#dc2626', label=f'TF-IDF cosine {cosine_acc:.3f}')
    ax.axhline(0.5, color='#64748b', ls='--', label='chance 0.50')
    best = ts[int(np.argmax(m))]
    ax.annotate(f'best T={best}: {m.max():.3f}', (best, m.max()), (best + 8, m.max() + 0.012),
                arrowprops=dict(arrowstyle='->'))
    ax.set_xlabel('Threshold T'); ax.set_ylabel('Validation accuracy'); ax.set_ylim(0.48, 0.64)
    ax.set_title('Effect of the threshold T (100 clauses per class)'); ax.legend(loc='lower right')
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / 't_curve.png', dpi=170); plt.close(fig)


def feature_diagram():
    """Worked example of how a pair becomes TM input."""
    words = ['space', 'orbit', 'hockey', 'goal']
    a = np.array([1, 1, 0, 0]); b = np.array([1, 0, 1, 0])
    rows = [('Post A', a), ('Post B', b), ('BOTH (A and B)', a & b), ('ONE (A xor B)', a ^ b)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    grid = np.array([r[1] for r in rows])
    ax.imshow(grid, cmap='Greens', vmin=0, vmax=1.4)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, str(grid[i, j]), ha='center', va='center', fontsize=15)
    ax.set_xticks(range(len(words)), words); ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.xaxis.tick_top(); ax.axhline(1.5, color='k', lw=2)
    ax.set_title('Pair features: the TM sees only the two bottom rows', pad=28)
    fig.tight_layout(); fig.savefig(FIG / 'feature_example.png', dpi=170); plt.close(fig)


def main(metrics_path):
    ablation = json.loads((RES / 'ablation.json').read_text(encoding='utf-8'))
    barh_with_error(ablation, 'What helped? One change at a time', 'ablation.png',
                    ablation[-1]['cosine_acc'], highlight='A3')
    sweep_path = RES / 'sweep.json'
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding='utf-8'))
        barh_with_error(sweep, 'Settings sweep around the new default', 'sweep.png',
                        sweep[0]['cosine_acc'], highlight='base')
    if (RES / 'sweep2.json').exists() and sweep_path.exists():
        t_curve(ablation[-1]['cosine_acc'])
    feature_diagram()
    r = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ['Majority', 'TF-IDF cosine', 'Vanilla TM\n(new default)']
    vals = [r['majority']['accuracy'], r['cosine']['accuracy'], r['tm']['accuracy']]
    bars = ax.bar(labels, vals, color=['#94a3b8', '#38bdf8', '#4ade80'])
    ax.set_ylim(0, 1); ax.set_ylabel('Accuracy'); ax.set_title('Validation accuracy, 4,000 pairs')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .02, f'{v:.3f}', ha='center')
    fig.tight_layout(); fig.savefig(FIG / 'headline.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = np.array(r['tm']['confusion_matrix'])
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16)
    ax.set_xticks([0, 1], ['Different', 'Same']); ax.set_yticks([0, 1], ['Different', 'Same'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Vanilla TM confusion matrix')
    fig.tight_layout(); fig.savefig(FIG / 'confusion_matrix.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])
