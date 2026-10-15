"""Publish an explicit scientific allowlist from the completed, audited sweep.

Private source paths are supplied at execution time and are never exported.
All compressed metric histories are schema-checked before release packaging.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile

from export_million_snapshot import point, metrics


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, (float, int)):
        assert math.isfinite(value)
    else: raise AssertionError(type(value))


def pack(path, members):
    records = []
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as tar:
        for name, source in sorted(members):
            assert not name.startswith('/') and '..' not in Path(name).parts
            info = tarfile.TarInfo(name)
            info.size = source.stat().st_size
            info.mode = 0o644
            with source.open('rb') as stream: tar.addfile(info, stream)
            records.append(dict(name=name, bytes=info.size, sha256=sha(source)))
    return dict(name=path.name, bytes=path.stat().st_size, sha256=sha(path), members=records,
                url='https://github.com/AntreasAntoniou/learning-modulo-seven/releases/download/v1.0.0/' + path.name)


def export(args):
    read = lambda p: json.loads(p.read_text())
    results, index, config = map(read, [args.completed_results, args.history_index, args.config])
    selection, tests, audit = map(read, [args.selection, args.test_results, args.prediction_audit])
    assert index['full_sweep_history_audit_complete'] and index['cases_verified'] == 576
    assert audit['saved_prediction_metrics_verified'] and audit['selected_checkpoints_verified'] == 96
    assert audit['dataset_disjoint_and_hashes_verified'] and audit['prediction_rows_verified'] == 12582912
    assert audit['all_cpu_replay_argmaxes_identical']
    expected = {f"{enc}-{v['name']}-wd{wd}-lr{lr}-s{seed}"
                for enc in ['scalar', 'bits'] for v in config['variants']
                for wd in range(3) for lr in range(2) for seed in [101, 202, 303]}
    assert len(expected) == len(results) == 576
    assert {r['case_id'] for r in results} == expected
    identities = {(r['identity']['source_commit'], r['identity']['trainval_sha256'],
                   r['identity']['config_digest']) for r in results}
    assert len(identities) == 1
    source, data_sha, config_digest = identities.pop()
    selected = {r['case_id']: r for g in selection['selected'] for r in g['models']}
    assert len(selected) == 96 and len(selection['selected']) == 32
    assert set(selected) == {r['case_id'] for r in tests['metrics']}
    assert sha(args.selection) == audit['selection_sha256']
    assert sha(args.test_results) == audit['test_results_sha256']
    audits = {r['case_id']: r for r in index['cases']}
    args.output.mkdir(parents=True, exist_ok=True)
    args.assets.mkdir(parents=True, exist_ok=True)
    fits, curves, histories = [], [], defaultdict(list)
    for number, result in enumerate(sorted(results, key=lambda r:r['case_id']), 1):
        key, case = result['case_id'], result['identity']['case']
        assert set(case) == {'id', 'encoding', 'variant', 'lr', 'weight_decay', 'seed'}
        assert case['id'] == key and case['variant'] in config['variants']
        assert result['status'] == 'complete' and result['samples'] == 10000000 and result['steps'] == 80000
        values = None
        for record in audits[key]['artifacts']:
            name = record['name']; assert name in ['training.jsonl.gz', 'validation.jsonl.gz']
            path = args.history_root / key / name
            assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
            count = 0; last = 0; vpoints = []
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    if name.startswith('training'):
                        assert set(row) == {'accuracy', 'batch_size', 'gradient_norm', 'loss', 'samples', 'step'}
                        finite(row)
                        assert row['step'] == count + 1 and row['samples'] - last == row['batch_size']
                        assert row['batch_size'] == (8 if (count + 1) % 40 == 0 else 128)
                        assert row['loss'] >= 0 and 0 <= row['accuracy'] <= 1 and row['gradient_norm'] >= 0
                    else:
                        assert set(row) == {'case_id', 'samples', 'step', 'best_samples', 'elapsed_seconds',
                                            'train_loss', 'train_accuracy', 'val_within', 'val_outside'}
                        assert row['case_id'] == key
                        finite({k:v for k,v in row.items() if k != 'case_id'})
                        assert row['samples'] == (count+1)*5000 and row['step'] == (count+1)*40
                        vpoints.append(point(row))
                    last = row['samples']; count += 1
            assert count == (80000 if name.startswith('training') else 2000) and last == 10000000
            histories[case['encoding']+'-'+case['variant']['name']].append((key+'/'+name, path))
            if vpoints: values = vpoints
        assert values is not None
        best = min(values, key=lambda v:(v['val_within']['loss'], -v['val_within']['accuracy'], v['samples']))
        assert best['samples'] == result['best_samples'] and best['val_within'] == result['best']
        fits.append(dict(case_id=key, case=case, parameters=result['parameters'], width=result['width'],
                         samples=result['samples'], steps=result['steps'], best=best, final=values[-1],
                         validation_history_sha256=sha(args.history_root/key/'validation.jsonl.gz'),
                         checkpoint_archive_sha256=result['checkpoint']['sha256']))
        if key in selected: curves.extend(values)
        if number % 18 == 0: print(f'Schema, hashes and selection verified: {number}/576 histories', flush=True)
    assert len(curves) == 192000
    snapshot = dict(schema=1, status='final_test_complete', completed_fits=576, expected_fits=576,
                    test_evaluated=True, sweep_selection_frozen=True, missing_case_ids=[],
                    source_commit=source, trainval_sha256=data_sha, config_digest=config_digest,
                    config_file_sha256=sha(args.config), input_completed_results_sha256=sha(args.completed_results),
                    input_history_audit_index_sha256=sha(args.history_index), fits=fits)
    write(args.output/'snapshot.json', snapshot)
    safe_selection = {k:selection[k] for k in ['config_digest','expected_fit_count',
        'overall_validation_winners','selection_rule','source_commit','trainval_sha256']}
    safe_selection['original_selection_sha256'] = sha(args.selection)
    safe_selection['selected'] = [dict(encoding=g['encoding'], variant=g['variant'], lr=g['lr'],
        weight_decay=g['weight_decay'], mean_validation_loss=g['mean_validation_loss'],
        mean_validation_accuracy=g['mean_validation_accuracy'],
        case_ids=[r['case_id'] for r in g['models']]) for g in selection['selected']]
    write(args.output/'selection.json', safe_selection)
    safe_test = dict(selection_sha256=sha(args.selection), original_results_sha256=sha(args.test_results),
                    sealed_test_sha256=tests['test']['sha256'], metrics=[])
    for row in tests['metrics']:
        safe = {k:row[k] for k in ['case_id','best_samples','test_within','test_outside']}
        for split in ['test_within','test_outside']:
            assert set(safe[split]) == {'count','accuracy','loss'} and safe[split]['count'] == 65536
            finite(safe[split])
        safe.update(checkpoint_archive_sha256=row['checkpoint']['sha256'], predictions_sha256=row['predictions']['sha256'])
        assert sha(args.prediction_root/(row['case_id']+'.npz')) == safe['predictions_sha256']
        safe_test['metrics'].append(safe)
    write(args.output/'test-results.json', safe_test)
    with (args.output/'selected-curves.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            for row in curves: gz.write((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode())
    # The remaining audit fields contain only scientific counts, hashes and measurements.
    safe_audit = {k:audit[k] for k in ['checked_utc','selection_sha256','test_results_sha256',
        'dataset_disjoint_and_hashes_verified','selected_checkpoints_verified','prediction_rows_verified',
        'saved_prediction_metrics_verified','cpu_replay_rows_verified','cpu_replay_scope',
        'cpu_replay_logit_tolerance','all_cpu_replay_argmaxes_identical','cpu_replay_summary',
        'cpu_replay_probability_absolute_tolerance','initial_strict_logit_check_passed',
        'initial_probability_check_passed','comparison_revision','replay_environment','cases']}
    write(args.output/'prediction-audit.json', safe_audit)
    safe_history = {k:index[k] for k in ['cases_verified','checkpoint_segments','compressed_metric_bytes',
        'training_batch_records','validation_records','all_compressed_artifact_hashes_and_gzip_crc_verified',
        'all_metric_file_record_counts_and_endpoints_verified']}
    safe_history.update(all_exported_records_schema_checked=True, raw_histories_in_release_assets=True)
    write(args.output/'history-audit.json', safe_history)
    assets = [pack(args.assets/('million-history-'+name+'.tar'),members) for name,members in sorted(histories.items())]
    for dirname, root, suffixes in [('predictions',args.prediction_root,['.npz']),
                                   ('selected-checkpoints',args.weights_root,['.json','.npz'])]:
        members=[]
        for cid in sorted(selected):
            for suffix in suffixes:
                path=root/(cid+suffix)
                if suffix=='.json':
                    metadata=read(path)
                    assert set(metadata)=={'case','meta','width','parameters','best_samples','source_commit',
                        'config_digest','trainval_sha256','checkpoint_archive_sha256','tensor_file_sha256'}
                    assert metadata['case']['id']==cid and sha(root/(cid+'.npz'))==metadata['tensor_file_sha256']
                members.append((cid+suffix,path))
        assets.append(pack(args.assets/('million-'+dirname+'.tar'),members))
    assets.append(pack(args.assets/'million-data.tar', [('trainval.npz',args.data_root/'trainval.npz'),
        ('test.npz',args.data_root/'sealed-test.npz'),('split-manifest.json',args.data_root/'split-manifest.json')]))
    write(args.output/'release-assets.json', dict(release='v1.0.0', assets=assets))
    print(json.dumps(dict(fits=len(fits), selected_curves=len(curves), release_assets=len(assets),
                         release_bytes=sum(x['bytes'] for x in assets))), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['completed-results','history-index','history-root','config','selection','test-results',
                 'prediction-audit','prediction-root','weights-root','data-root','output','assets']:
        parser.add_argument('--'+name, type=Path, required=True)
    export(parser.parse_args())
