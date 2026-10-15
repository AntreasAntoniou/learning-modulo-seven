"""Verify final public statistics and optionally every downloaded release asset.

No training or model selection is executed. Asset verification independently
recomputes batch windows and test metrics from the released numeric records.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/million/final-2026-10-15'


def load(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(assets=None):
    snap,selection,test=map(load,[DATA/'snapshot.json',DATA/'selection.json',DATA/'test-results.json'])
    fits={r['case_id']:r for r in snap['fits']};tests={r['case_id']:r for r in test['metrics']}
    config=load(ROOT/'experiments/million/config.json')
    expected={f"{e}-{v['name']}-wd{w}-lr{l}-s{s}" for e in ['bits','scalar'] for v in config['variants']
              for w in range(3) for l in range(2) for s in [101,202,303]}
    assert set(fits)==expected and len(fits)==len(snap['fits'])==576 and snap['test_evaluated']
    by_group=defaultdict(list)
    for r in fits.values():
        c=r['case'];assert r['samples']==10000000 and r['steps']==80000
        by_group[(c['encoding'],c['variant']['name'],c['lr'],c['weight_decay'])].append(r)
    assert len(by_group)==192 and all(len(v)==3 for v in by_group.values())
    best={}
    for (e,v,lr,wd),rows in by_group.items():
        key=(math.fsum(r['best']['val_within']['loss'] for r in rows)/3,
             -math.fsum(r['best']['val_within']['accuracy'] for r in rows)/3,wd,lr)
        if (e,v) not in best or key<best[(e,v)][0]:best[(e,v)]=(key,rows)
    chosen=set()
    for g in selection['selected']:
        key,rows=best[(g['encoding'],g['variant'])]
        assert set(g['case_ids'])=={r['case_id'] for r in rows}
        assert g['lr']==key[3] and g['weight_decay']==key[2]
        assert math.isclose(g['mean_validation_loss'],key[0],abs_tol=1e-12)
        chosen.update(g['case_ids'])
    assert len(chosen)==96 and chosen==set(tests)
    for e in ['bits','scalar']:
        winner=min((g for g in selection['selected'] if g['encoding']==e),
                   key=lambda g:(g['mean_validation_loss'],-g['mean_validation_accuracy'],g['variant']))
        assert winner['variant']==selection['overall_validation_winners'][e]
    with (DATA/'selected-statistics.csv').open() as stream:
        stats=list(csv.DictReader(stream))
    for row in stats:
        rr=next(g for g in selection['selected'] if g['encoding']==row['encoding'] and g['variant']==row['architecture'])
        for split in ['val_within','val_outside','test_within','test_outside']:
            for metric in ['loss','accuracy']:
                values=[(fits[i]['best'] if split.startswith('val') else tests[i])[split][metric] for i in rr['case_ids']]
                for suffix,value in [('mean',mean(values)),('sd',stdev(values))]:
                    assert math.isclose(float(row[split+'_'+metric+'_'+suffix]),value,abs_tol=1e-12)
    curves=defaultdict(list)
    with gzip.open(DATA/'selected-curves.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);curves[row['case_id']].append(row)
    assert set(curves)==chosen
    for cid,rows in curves.items():
        assert [r['samples'] for r in rows]==list(range(5000,10000001,5000))
        assert min(rows,key=lambda r:(r['val_within']['loss'],-r['val_within']['accuracy'],r['samples']))==fits[cid]['best']
    # The dated twelve-fit and 560-fit releases remain exact subsets.
    for row in load(ROOT/'results/million/draft-2026-10-15/snapshot.json')['fits']:
        assert row==fits[row['case_id']]
    result=dict(fits=576,complete_recipes=192,selected_models=96,selected_curve_rows=192000,
                all_selected_statistics_recomputed=True,prior_560_results_unchanged=True)
    if assets:
        manifest=load(DATA/'release-assets.json')['assets'];member_hashes=0
        for a in manifest:
            path=assets/a['name'];assert path.stat().st_size==a['bytes'] and sha(path)==a['sha256']
            with tarfile.open(path) as archive:
                assert archive.getnames()==[r['name'] for r in a['members']]
                for r in a['members']:
                    stream=archive.extractfile(r['name'])
                    assert hashlib.file_digest(stream,'sha256').hexdigest()==r['sha256'];member_hashes+=1
        batches=validations=prediction_rows=0
        for a in manifest:
            if not a['name'].startswith('million-history-'):continue
            with tarfile.open(assets/a['name']) as archive:
                ids=sorted({n.split('/')[0] for n in archive.getnames()})
                for cid in ids:
                    with gzip.GzipFile(fileobj=archive.extractfile(cid+'/validation.jsonl.gz')) as stream:
                        vals=[json.loads(line) for line in stream]
                    assert len(vals)==2000
                    n=0;step=0;loss=acc=0.;index=0
                    with gzip.GzipFile(fileobj=archive.extractfile(cid+'/training.jsonl.gz')) as stream:
                        for line in stream:
                            r=json.loads(line);step+=1;count=8 if step%40==0 else 128;n+=count
                            assert (r['step'],r['batch_size'],r['samples'])==(step,count,n)
                            assert all(math.isfinite(r[k]) for k in ['loss','accuracy','gradient_norm'])
                            loss+=count*r['loss'];acc+=count*r['accuracy']
                            if step%40==0:
                                v=vals[index];assert (v['step'],v['samples'])==(step,n)
                                assert math.isclose(loss/5000,v['train_loss'],abs_tol=1e-12,rel_tol=1e-12)
                                assert math.isclose(acc/5000,v['train_accuracy'],abs_tol=1e-12,rel_tol=1e-12)
                                index+=1;loss=acc=0.
                    assert n==10000000 and step==80000
                    bestrow=min(vals,key=lambda r:(r['val_within']['loss'],-r['val_within']['accuracy'],r['samples']))
                    for k,v in fits[cid]['best'].items():assert bestrow[k]==v
                    batches+=step;validations+=len(vals)
            print(a['name']+' batch windows verified',flush=True)
        with tarfile.open(assets/'million-data.tar') as tar:
            with np.load(io.BytesIO(tar.extractfile('test.npz').read()),allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with tarfile.open(assets/'million-predictions.tar') as tar:
            for cid,r in tests.items():
                with np.load(io.BytesIO(tar.extractfile(cid+'.npz').read()),allow_pickle=False) as z:
                    for split in ['test_within','test_outside']:
                        ids=z[split+'_ids'];logits=z[split+'_logits'];assert np.array_equal(ids,data[split])
                        assert logits.shape==(65536,7) and np.isfinite(logits).all()
                        accuracy=np.mean(logits.argmax(1)==ids%7);assert accuracy==r[split]['accuracy']
                        x=logits.astype(np.float64);x-=x.max(1,keepdims=True)
                        loss=np.mean(np.log(np.exp(x).sum(1))-x[np.arange(len(x)),ids%7])
                        assert math.isclose(loss,r[split]['loss'],rel_tol=1e-6,abs_tol=1e-6)
                        prediction_rows+=len(ids)
        assert (batches,validations,prediction_rows)==(46080000,1152000,12582912)
        result.update(release_assets_verified=len(manifest),release_members_hashed=member_hashes,
                      training_batch_windows_recomputed=batches,validation_rows_verified=validations,
                      saved_prediction_rows_rescored=prediction_rows)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--assets',type=Path)
    parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    result=verify(args.assets)
    if args.receipt:args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
