"""Figure for docs/03 section 3.11, from docs/results/bittm_tools.json."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('docs/results/bittm_tools.json', encoding='utf-8'))
rows = d['simplify']
labels = ['original'] + ['tol ' + str(r['tolerance']) for r in rows[1:]]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(labels, [r['median_literals'] for r in rows], color='#3446d8')
ax[0].set_title('Median conditions per rule'); ax[0].tick_params(axis='x', rotation=20)
for i, r in enumerate(rows):
    ax[0].text(i, r['median_literals'] + 10, f"{r['median_literals']:.0f}", ha='center')
ax[1].plot(labels, [r['accuracy'] for r in rows], 'o-', color='#0f8b8d', label='accuracy')
ax[1].plot(labels, [r['auc'] for r in rows], 's-', color='#a8409b', label='AUC')
ax[1].axhline(d['cosine_auc']['mean'], color='#c26a00', ls='--', label='cosine AUC')
ax[1].set_title('Validation score after shortening'); ax[1].set_ylim(0.5, 0.65); ax[1].legend(); ax[1].tick_params(axis='x', rotation=20)
fig.tight_layout(); fig.savefig('docs/figures/simplify.png', dpi=150)
