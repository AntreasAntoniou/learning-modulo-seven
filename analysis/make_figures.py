"""Rebuild report figures and numeric tables from the published measurements."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report/figures"
TABLE = ROOT / "report/tables"
COLORS = ["#1849a9", "#c26916", "#25836b"]
SEEDS = [101, 202, 303]
NAMES = {
    "mlp3":"MLP 3", "mlp20":"MLP 20", "mlp50":"MLP 50",
    "resmlp20":"Residual MLP 20", "resmlp50":"Residual MLP 50",
    **{f"transformer{d}":f"Transformer {d}" for d in [2,8,12]},
    **{f"relational{d}":f"Relational {d}" for d in [2,8,12]},
    **{n:n.replace("hyper_", "Hyper ").replace("_", "/") for n in ["hyper_g2_f3","hyper_g8_f3","hyper_g12_f3","hyper_g8_f1","hyper_g8_f5"]},
}

def csv_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))

def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight", metadata={"Creator":"Matplotlib", "CreationDate":None})
    fig.savefig(FIG / (name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

def style():
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Serif", "font.size":9,
        "axes.titlesize":10, "axes.labelsize":9, "legend.fontsize":8,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.edgecolor":"#818895", "axes.linewidth":.6,
        "grid.color":"#dfe3e8", "grid.linewidth":.5, "pdf.fonttype":42})

def primary():
    selected = csv_rows(ROOT / "results/screen/selected-fits.csv")
    summary = csv_rows(ROOT / "results/screen/summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.8,4.0), sharex=True, sharey=True)
    for j,name in enumerate(["hyper_g2_f3", "hyper_g8_f3"]):
        for k,split in enumerate(["test", "ood"]):
            ax = axes[k,j]
            for seed,color in zip(SEEDS,COLORS):
                values = sorted([r for r in selected if r['task']=='modulo_bits' and r['architecture']==name and int(r['seed'])==seed], key=lambda r:int(r['training_samples']))
                ax.plot(range(3),[100*float(r[split+'_accuracy']) for r in values], '-o',
                        color=color, lw=1, ms=4, label=f"Seed {seed}")
            ax.axhline(100/7, color='#777', ls=':', lw=.8)
            ax.set_ylim(0,105); ax.set_yticks([0,25,50,75,100]); ax.grid(axis='y')
            ax.set_xticks(range(3),['256','4,096','65,536'])
            if k==0: ax.set_title(NAMES[name], fontweight='bold')
            if j==0: ax.set_ylabel(('Within-range' if k==0 else 'Outside-range')+' test (%)')
            if k==1: ax.set_xlabel('Unique training examples')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,'data-and-seeds')

    fig, axes=plt.subplots(1,2,figsize=(6.8,4.7),sharey=True)
    models=list(NAMES)
    for ax,split,title in zip(axes,['test','ood'],['Within-range test','Outside-range test']):
        grid=np.array([[100*float(next(r for r in summary if r['task']=='modulo_bits' and r['architecture']==m and int(r['training_samples'])==n)[split+'_accuracy_mean']) for n in [256,4096,65536]] for m in models])
        im=ax.imshow(grid,aspect='auto',cmap='Blues',norm=Normalize(0,100))
        for (i,j),value in np.ndenumerate(grid):
            ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=8,color='white' if value>60 else '#152438')
        ax.set_xticks(range(3),['256','4,096','65,536']);ax.set_title(title,fontweight='bold')
        ax.set_yticks(range(len(models)),[NAMES[m] for m in models]);ax.tick_params(length=0)
    fig.subplots_adjust(left=.27,right=.99,bottom=.07,top=.93,wspace=.10)
    save(fig,'architecture-grid')

    rows=[]
    for m in models:
        def val(task,split):
            r=next(r for r in summary if r['task']==task and r['architecture']==m and int(r['training_samples'])==65536)
            return f"{float(r[split+'_accuracy_mean'])*100:.2f} $\\pm$ {float(r[split+'_accuracy_std'])*100:.2f}"
        rows.append(NAMES[m]+" & "+" & ".join(val(t,s) for t,s in [('modulo_scalar','test'),('modulo_bits','test'),('modulo_bits','ood')])+r" \\")
    (TABLE/'screen.tex').write_text('\n'.join(rows)+'\n\\bottomrule\n')

def live():
    snapshot=json.loads((ROOT/'results/million/validation-snapshot.json').read_text())
    groups=defaultdict(list)
    with gzip.open(ROOT/'results/million/validation-curves.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);groups[row['case_id']].append(row)
    fig, axes=plt.subplots(2,3,figsize=(7.0,4.4),sharex=True)
    for i in range(2):
        for seed,color in zip(SEEDS,COLORS):
            key=f'bits-hyper_g2_f3-wd0-lr{i}-s{seed}'
            values=sorted(groups[key],key=lambda r:r['samples'])
            x=np.array([r['samples']/1e6 for r in values])
            for j,extract in enumerate([lambda r:r['train_loss'],lambda r:r['val_within']['loss'],lambda r:100*r['val_outside']['accuracy']]):
                axes[i,j].plot(x,[extract(r) for r in values],color=color,lw=.7,alpha=.85,label=f'Seed {seed}')
        for j in [0,1]:
            axes[i,j].set_yscale('log');axes[i,j].set_ylim(3e-4,4)
            axes[i,j].axhline(np.log(7),color='#777',ls=':',lw=.7)
        axes[i,2].set_ylim(0,105);axes[i,2].axhline(100/7,color='#777',ls=':',lw=.7)
        axes[i,0].set_ylabel('LR '+['0.0003','0.003'][i])
        for j in range(3):
            axes[i,j].grid(axis='y');axes[i,j].set_xticks([0,5,10])
            if i==1: axes[i,j].set_xlabel('Examples processed (M)')
    for ax,title in zip(axes[0],['Training loss','Within-validation loss','Outside-validation accuracy (%)']):
        ax.set_title(title,fontsize=8.5,fontweight='bold')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'million-learning-curves')
    lines=[]
    for r in sorted(snapshot['results'],key=lambda r:(r['case']['weight_decay'],r['case']['lr'],r['case']['seed'])):
        c=r['case']
        lines.append(f"{c['weight_decay']:g} & {c['lr']:g} & {c['seed']} & {r['best_samples']/1e6:.3f} & {r['best']['loss']:.4f} & {100*r['best']['accuracy']:.2f} & {100*r['outside_at_selected_checkpoint']['accuracy']:.2f}"+r" \\")
    (TABLE/'million.tex').write_text('\n'.join(lines)+'\n\\bottomrule\n')

if __name__=='__main__':
    style();primary();live()
    from million_draft import build
    build()
