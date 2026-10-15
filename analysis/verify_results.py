"""Recompute public selections/statistics and validate the released metric chains."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)

def csvrows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def verify_screen(kind, expected):
    folder=ROOT/'results'/kind
    raw=list(rows(folder/'all-fits.jsonl.gz'))
    assert len(raw)==expected and len({r['case_id'] for r in raw})==expected
    groups=defaultdict(list)
    for r in raw:
        assert r['status']=='completed' and r['optimizer_steps']==1200
        groups[(r['task'],r['variant']['name'],r['training_samples'],r['seed'])].append(r)
    selected={}
    for group in groups.values():
        assert {r['lr'] for r in group}=={.0003,.003}
        winner=min(group,key=lambda r:(r['validation_loss'],r['lr']))
        selected[winner['case_id']]=winner
    table=csvrows(folder/'selected-fits.csv')
    assert {r['case_id'] for r in table}==set(selected)
    for record in table:
        fit=selected[record['case_id']]
        for split,metrics in fit['metrics'].items():
            for metric,value in metrics.items():
                assert math.isclose(float(record[split+'_'+metric]), value, rel_tol=1e-12,abs_tol=1e-12)
    for record in csvrows(folder/'summary.csv'):
        matches=[r for r in selected.values() if r['task']==record['task'] and r['variant']['name']==record['architecture'] and r['training_samples']==int(record['training_samples'])]
        assert sorted(r['seed'] for r in matches)==[101,202,303]
        for split,metrics in matches[0]['metrics'].items():
            for metric in metrics:
                values=[r['metrics'][split][metric] for r in matches]
                for suffix,value in [('mean',mean(values)),('std',stdev(values))]:
                    assert math.isclose(float(record[f'{split}_{metric}_{suffix}']),value,rel_tol=1e-10,abs_tol=1e-10)
    return {'fits':len(raw),'selected_fits':len(selected),'summary_cells':len(csvrows(folder/'summary.csv'))}

def verify_million():
    folder=ROOT/'results/million'
    snapshot=json.loads((folder/'validation-snapshot.json').read_text())
    assert snapshot['partial_sweep'] and snapshot['test_evaluated'] is False
    results={r['case_id']:r for r in snapshot['results']}
    validations=defaultdict(list)
    for r in rows(folder/'validation-curves.jsonl.gz'):
        assert r['case_id'] in results
        validations[r['case_id']].append(r)
    for key,records in validations.items():
        records.sort(key=lambda r:r['samples'])
        assert [r['samples'] for r in records]==list(range(5000,10_000_001,5000))
        best=min(records,key=lambda r:(r['val_within']['loss'],-r['val_within']['accuracy'],r['samples']))
        assert best['samples']==results[key]['best_samples']
        assert best['val_within']==results[key]['best']
        assert best['val_outside']==results[key]['outside_at_selected_checkpoint']
    window={key:[0,0.,0,0] for key in results}
    total=0
    for batch in rows(folder/'training-batches.jsonl.gz'):
        key=batch['case_id'];w=window[key]
        assert batch['step']==w[2]+1
        assert batch['samples']==w[0]+batch['batch_size']
        assert batch['batch_size']==(8 if batch['samples']%5000==0 else 128)
        w[0]=batch['samples'];w[1]+=batch['loss']*batch['batch_size'];w[2]+=1;total+=1
        if w[0]%5000==0:
            val=validations[key][w[3]]
            assert abs(w[1]/5000-val['train_loss'])<1e-10
            w[1]=0.;w[3]+=1
    assert all(w==[10_000_000,0.,80_000,2000] for w in window.values())
    return {'completed_fits':len(results),'validation_observations':sum(map(len,validations.values())), 'batch_observations':total,'test_evaluated':False}

def verify_million_draft():
    folder=ROOT/'results/million/draft-2026-10-15'
    snap=json.loads((folder/'snapshot.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    assert snap['status']=='partial_validation_draft'
    assert snap['test_evaluated'] is False and snap['sweep_selection_frozen'] is False
    fits={r['case_id']:r for r in snap['fits']}
    assert len(fits)==len(snap['fits'])==snap['completed_fits']==560
    config=json.loads((ROOT/'experiments/million/config.json').read_text())
    expected={f"{e}-{v['name']}-wd{w}-lr{l}-s{s}" for e in ['bits','scalar']
              for v in config['variants'] for w in range(3) for l in range(2) for s in [101,202,303]}
    assert set(fits).isdisjoint(snap['missing_case_ids'])
    assert set(fits)|set(snap['missing_case_ids'])==expected
    assert len(snap['missing_case_ids'])==summary['missing_fits']==16
    def finite(value):
        if isinstance(value,dict):
            for v in value.values(): finite(v)
        elif isinstance(value,list):
            for v in value: finite(v)
        elif isinstance(value,(float,int)):
            assert math.isfinite(value)
    finite(snap)
    for r in fits.values():
        assert r['samples']==r['final']['samples']==10_000_000 and r['steps']==80_000
        assert r['best']['samples']%5000==0 and 0<r['best']['samples']<=10_000_000
        assert r['best']['val_within']['loss']<=r['final']['val_within']['loss']
        for point in [r['best'],r['final']]:
            for split in ['val_within','val_outside']:
                assert point[split]['count']==8192 and 0<=point[split]['accuracy']<=1
    for encoding in ['bits','scalar']:
        rr=[r for r in fits.values() if r['case']['encoding']==encoding]
        result=summary['encodings'][encoding]
        assert result['completed']==len(rr)
        assert result['within_ge99']==sum(r['best']['val_within']['accuracy']>=.99 for r in rr)
        assert result['both_ge99']==sum(r['best']['val_within']['accuracy']>=.99 and
                                       r['best']['val_outside']['accuracy']>=.99 for r in rr)
        for metric in ['loss','accuracy']:
            values=[r['best']['val_within'][metric] for r in rr]
            assert result['within_'+metric+'_min']==min(values)
            assert result['within_'+metric+'_max']==max(values)
    cells=csvrows(folder/'recipe-statistics.csv')
    assert len(cells)==192
    seen=set(); complete=0; all_high=0
    for c in cells:
        key=(c['encoding'],c['architecture'],float(c['learning_rate']),float(c['weight_decay']))
        assert key not in seen;seen.add(key)
        rr=[r for r in fits.values() if (r['case']['encoding'],r['case']['variant']['name'],
             r['case']['lr'],r['case']['weight_decay'])==key]
        assert len(rr)==int(c['n'])
        is_complete=len(rr)==3
        assert (c['complete']=='True')==is_complete
        if is_complete:
            assert {r['case']['seed'] for r in rr}=={101,202,303}
            complete+=1
            all_high+=all(r['best']['val_within']['accuracy']>=.99 and
                          r['best']['val_outside']['accuracy']>=.99 for r in rr)
        for split in ['val_within','val_outside']:
            for metric in ['loss','accuracy']:
                for suffix,fn in [('mean',mean),('sd',stdev)]:
                    value=c[f'{split}_{metric}_{suffix}']
                    if is_complete:
                        assert math.isclose(float(value),fn(r['best'][split][metric] for r in rr),abs_tol=1e-12)
                    else:
                        assert value==''
    assert complete==summary['complete_recipe_cells']
    assert all_high==summary['recipe_cells_with_all_seeds_both_ge99']==0
    curves=defaultdict(list)
    for row in rows(folder/'illustrative-curves.jsonl.gz'):
        finite(row);curves[row['case_id']].append(row)
    assert len(curves)==12
    for key,rr in curves.items():
        assert [r['samples'] for r in rr]==list(range(5000,10_000_001,5000))
        assert all(r['step']==r['samples']//125 for r in rr)
        best=min(rr,key=lambda r:(r['val_within']['loss'],-r['val_within']['accuracy'],r['samples']))
        assert best==fits[key]['best'] and rr[-1]==fits[key]['final']
    early=json.loads((ROOT/'results/million/validation-snapshot.json').read_text())
    for r in early['results']:
        best=fits[r['case_id']]['best']
        assert best['samples']==r['best_samples']
        assert best['val_within']==r['best'] and best['val_outside']==r['outside_at_selected_checkpoint']
    lines=[]
    for lr in [.0003,.003]:
        for wd in [0.,.01,.1]:
            rr=sorted([r for r in fits.values() if r['case']['encoding']=='bits'
                and r['case']['variant']['name']=='hyper_g2_f3'
                and r['case']['lr']==lr and r['case']['weight_decay']==wd],key=lambda r:r['case']['seed'])
            values=[f"{100*r['best'][split]['accuracy']:.2f}" for split in ['val_within','val_outside'] for r in rr]
            lines.append(f"{lr:g} & {wd:g} & "+' & '.join(values)+r' \\')
    assert (ROOT/'report/tables/million-hyper-settings.tex').read_text()=='\n'.join(lines)+'\n\\bottomrule\n'
    return {'completed_fits':len(fits),'missing_fits':16,'complete_three_seed_cells':complete,
            'illustrative_validation_rows':sum(map(len,curves.values())), 'early_results_unchanged':12,
            'test_evaluated':False, 'sweep_selection_frozen':False}

def main():
    origins=json.loads((ROOT/'provenance/source-origins.json').read_text())
    for path,record in origins.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==record['sha256']
    print(json.dumps({'unchanged_scientific_files':len(origins),
        'screen':verify_screen('screen',1728),'observed_bits':verify_screen('observed_bits',96),
        'million_snapshot':verify_million(), 'million_draft':verify_million_draft()},indent=2))

if __name__=='__main__':
    main()
