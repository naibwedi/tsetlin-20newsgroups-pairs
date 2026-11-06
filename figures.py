"""Render charts from measured validation/test JSON output."""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('metrics', type=Path)
a=p.parse_args()
r=json.loads(a.metrics.read_text(encoding='utf-8'))
out=a.metrics.parent
fig,ax=plt.subplots(figsize=(8,4.5))
labels=['Majority','TF-IDF cosine','Vanilla BitTM']
vals=[r['majority']['accuracy'],r['cosine']['accuracy'],r['tm']['accuracy']]
bars=ax.bar(labels,vals,color=['#94a3b8','#38bdf8','#4ade80'])
ax.set_ylim(0,1);ax.set_ylabel('Accuracy');ax.set_title('Document-pair accuracy on '+r['split'])
for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v+.02,f'{v:.3f}',ha='center')
fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(5,4.5))
cm=np.array(r['tm']['confusion_matrix'])
ax.imshow(cm,cmap='Blues')
for i in range(2):
 for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16)
ax.set_xticks([0,1],['Different','Same']);ax.set_yticks([0,1],['Different','Same'])
ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title('Vanilla BitTM confusion matrix')
fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=170);plt.close(fig)
