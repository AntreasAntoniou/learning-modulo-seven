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

def main():
    origins=json.loads((ROOT/'provenance/source-origins.json').read_text())
    for path,record in origins.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==record['sha256']
    print(json.dumps({'unchanged_scientific_files':len(origins),
        'screen':verify_screen('screen',1728),'observed_bits':verify_screen('observed_bits',96),
        'million_snapshot':verify_million()},indent=2))

if __name__=='__main__':
    main()
