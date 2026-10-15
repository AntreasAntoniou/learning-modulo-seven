"""Summaries and final-test figures; selection is read, never changed by test scores."""
from collections import defaultdict
import csv
import gzip
import json
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
from make_figures import ROOT, FIG, TABLE, COLORS, SEEDS, NAMES, save, style

DATA = ROOT/'results/million/final-2026-10-15'


def write_csv(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def build():
    snap=json.loads((DATA/'snapshot.json').read_text())
    sel=json.loads((DATA/'selection.json').read_text())
    tests=json.loads((DATA/'test-results.json').read_text())
    fits={r['case_id']:r for r in snap['fits']}; test={r['case_id']:r for r in tests['metrics']}
    assert len(fits)==576 and len(test)==96
    groups={(g['encoding'],g['variant']):g for g in sel['selected']}
    cells=[]; selected_stats=[]
    for encoding in ['bits','scalar']:
        for architecture in NAMES:
            g=groups[(encoding,architecture)]
            for lr in [.0003,.003]:
                for wd in [0.,.01,.1]:
                    rows=sorted([r for r in fits.values() if r['case']['encoding']==encoding
                        and r['case']['variant']['name']==architecture and r['case']['lr']==lr
                        and r['case']['weight_decay']==wd],key=lambda r:r['case']['seed'])
                    assert [r['case']['seed'] for r in rows]==SEEDS
                    cell=dict(encoding=encoding,architecture=architecture,learning_rate=lr,weight_decay=wd,n=3,
                              selected=(lr==g['lr'] and wd==g['weight_decay']))
                    for domain in ['val_within','val_outside']:
                        for metric in ['loss','accuracy']:
                            vals=[r['best'][domain][metric] for r in rows]
                            cell[domain+'_'+metric+'_mean']=mean(vals)
                            cell[domain+'_'+metric+'_sd']=stdev(vals)
                    cells.append(cell)
            stat=dict(encoding=encoding,architecture=architecture,learning_rate=g['lr'],weight_decay=g['weight_decay'],n=3)
            for domain in ['val_within','val_outside','test_within','test_outside']:
                for metric in ['loss','accuracy']:
                    vals=[(fits[i]['best'] if domain.startswith('val') else test[i])[domain][metric] for i in g['case_ids']]
                    stat[domain+'_'+metric+'_mean']=mean(vals)
                    stat[domain+'_'+metric+'_sd']=stdev(vals)
            selected_stats.append(stat)
    write_csv(DATA/'recipe-statistics.csv',cells); write_csv(DATA/'selected-statistics.csv',selected_stats)
    seed_rows=[]
    for cid in sorted(test):
        f,t=fits[cid],test[cid]
        row=dict(case_id=cid,encoding=f['case']['encoding'],architecture=f['case']['variant']['name'],
            seed=f['case']['seed'],learning_rate=f['case']['lr'],weight_decay=f['case']['weight_decay'],
            best_samples=t['best_samples'],parameters=f['parameters'],width=f['width'])
        for domain in ['val_within','val_outside','test_within','test_outside']:
            for metric in ['loss','accuracy']:
                row[domain+'_'+metric]=(f['best'] if domain.startswith('val') else t)[domain][metric]
        seed_rows.append(row)
    write_csv(DATA/'selected-per-seed.csv',seed_rows)
    summary=dict(completed_fits=576,selected_models=96,complete_recipe_cells=192,
                 overall_validation_winners=sel['overall_validation_winners'],encodings={},selected_groups=selected_stats,
                 threshold_status='99 percent is a post-hoc descriptive threshold, not a selection rule.')
    for encoding in ['bits','scalar']:
        allfits=[r for r in fits.values() if r['case']['encoding']==encoding]
        chosen=[r for r in seed_rows if r['encoding']==encoding]
        recipe_success=[]
        for cell in cells:
            if cell['encoding']!=encoding:continue
            rr=[r for r in allfits if r['case']['variant']['name']==cell['architecture'] and r['case']['lr']==cell['learning_rate'] and r['case']['weight_decay']==cell['weight_decay']]
            recipe_success.append(all(min(r['best'][d]['accuracy'] for d in ['val_within','val_outside'])>=.99 for r in rr))
        summary['encodings'][encoding]=dict(completed=len(allfits),selected=len(chosen),
            validation_within_ge99=sum(r['best']['val_within']['accuracy']>=.99 for r in allfits),
            validation_both_ge99=sum(min(r['best'][d]['accuracy'] for d in ['val_within','val_outside'])>=.99 for r in allfits),
            recipes_all_three_validation_both_ge99=sum(recipe_success),
            test_within_ge99=sum(r['test_within_accuracy']>=.99 for r in chosen),
            test_both_ge99=sum(min(r['test_within_accuracy'],r['test_outside_accuracy'])>=.99 for r in chosen),
            selected_recipes_all_three_test_both_ge99=sum(all(min(r['test_within_accuracy'],r['test_outside_accuracy'])>=.99 for r in chosen if r['architecture']==name) for name in NAMES),
            validation_loss_min=min(r['best']['val_within']['loss'] for r in allfits),
            validation_loss_max=max(r['best']['val_within']['loss'] for r in allfits),
            validation_accuracy_min=min(r['best']['val_within']['accuracy'] for r in allfits),
            validation_accuracy_max=max(r['best']['val_within']['accuracy'] for r in allfits))
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    # Individual seed points are essential because means conceal success/failure mixtures.
    fig,axes=plt.subplots(1,2,figsize=(7,5.0),sharey=True)
    for ax,domain,title in zip(axes,['test_within','test_outside'],['Within-range test','Outside-range test']):
        for i,name in enumerate(NAMES):
            rr=[r for r in seed_rows if r['encoding']=='bits' and r['architecture']==name]
            for j,(seed,color,marker) in enumerate(zip(SEEDS,COLORS,['o','s','^'])):
                r=next(r for r in rr if r['seed']==seed)
                ax.scatter(100*r[domain+'_accuracy'],i+(j-1)*.16,color=color,marker=marker,s=21,zorder=3)
        ax.axvline(100/7,color='#777',ls=':',lw=.8);ax.set_xlim(0,104);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x')
        ax.set_title(title,fontweight='bold');ax.set_xlabel('Accuracy (%)');ax.set_yticks(range(16),list(NAMES.values()),fontsize=8)
    axes[0].invert_yaxis()
    handles=[axes[0].scatter([],[],color=c,marker=m,label=f'Seed {s}',s=22) for s,c,m in zip(SEEDS,COLORS,['o','s','^'])]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,'million-final-test')

    curves=defaultdict(list)
    with gzip.open(DATA/'selected-curves.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);curves[r['case_id']].append(r)
    # Four informative architectures, selected post hoc for illustration; fixed
    # hyperparameters and all seeds are exactly those frozen before test access.
    names=['hyper_g2_f3','resmlp20','resmlp50','hyper_g8_f5']
    fig,axes=plt.subplots(4,3,figsize=(7,6.2),sharex=True)
    for i,name in enumerate(names):
        group=groups[('bits',name)]
        for seed,color,ls in zip(SEEDS,COLORS,['-','--',':']):
            cid=next(cid for cid in group['case_ids'] if fits[cid]['case']['seed']==seed)
            points=curves[cid];x=np.array([v['samples']/1e6 for v in points])
            for j,key in enumerate(['train_loss','val_within','val_outside']):
                y=np.array([v[key] if key=='train_loss' else v[key]['loss'] for v in points])
                axes[i,j].plot(x,np.maximum(y,1e-6),color=color,ls=ls,lw=.65,label=f'Seed {seed}')
                at=next(k for k,v in enumerate(points) if v['samples']==fits[cid]['best']['samples'])
                axes[i,j].scatter([x[at]],[max(y[at],1e-6)],color=color,s=10,zorder=3)
        axes[i,0].set_ylabel(NAMES[name]+f"\nLR {group['lr']:g}, decay {group['weight_decay']:g}",fontsize=8)
        for ax in axes[i]:
            ax.set_yscale('log');ax.axhline(np.log(7),color='#777',ls=':',lw=.7);ax.grid(axis='y')
            ax.set_xticks([0,5,10]);ax.tick_params(labelsize=7.5)
            if i==3:ax.set_xlabel('Presentations (M)',fontsize=8)
    for ax,title in zip(axes[0],['Training-window CE','Within-validation CE','Outside-validation CE']):ax.set_title(title,fontsize=8.5,fontweight='bold')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'million-selected-loss-curves')

    for encoding in ['bits','scalar']:
        lines=[]; accuracy_lines=[]
        for s in [s for s in selected_stats if s['encoding']==encoding]:
            vals=[f"{s[d+'_accuracy_mean']*100:.2f} $\\pm$ {s[d+'_accuracy_sd']*100:.2f}" for d in ['test_within','test_outside']]
            losses=[f"{s[d+'_loss_mean']:.4f} $\\pm$ {s[d+'_loss_sd']:.4f}" for d in ['test_within','test_outside']]
            lines.append(NAMES[s['architecture']]+f" & {s['learning_rate']:g} & {s['weight_decay']:g} & "+' & '.join(vals+losses)+r' \\')
            accuracy_lines.append(NAMES[s['architecture']]+f" & {s['learning_rate']:g} & {s['weight_decay']:g} & "+' & '.join(vals)+r' \\')
        (TABLE/('million-final-'+encoding+'.tex')).write_text('\n'.join(lines)+'\n\\bottomrule\n')
        (TABLE/('million-final-'+encoding+'-accuracy.tex')).write_text('\n'.join(accuracy_lines)+'\n\\bottomrule\n')
    print(json.dumps(summary['encodings'],indent=2))
    for s in selected_stats:
        if s['architecture']==sel['overall_validation_winners'][s['encoding']]:print(json.dumps(s,indent=2))


if __name__=='__main__':
    style();build()
