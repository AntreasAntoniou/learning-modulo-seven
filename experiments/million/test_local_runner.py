"""Exercise the portable storage/evaluation path on a separate tiny dataset."""
import json
from pathlib import Path
import numpy as np
import pytest
from data import prepare, split_ids
from run import final_test
from train import cases, fit
from test_protocol import tiny

def test_test_split_cannot_enter_training(tmp_path):
    config=tiny()
    ids=split_ids(config)
    path=tmp_path/'wrong.npz'
    np.savez(path, **ids)
    with pytest.raises(ValueError, match='Only training and validation'):
        fit(config,cases(config)[0],path,tmp_path/'wrong-fit')

def test_complete_local_reproduction_and_one_time_test_guard(tmp_path):
    config=tiny()
    with pytest.raises(ValueError, match='every preregistered case'):
        final_test(config,tmp_path,'cpu')
    assert not (tmp_path/'final-test-fence.json').exists()
    prepare(config,tmp_path/'data')
    for case in cases(config):
        fit(config,case,tmp_path/'data/trainval.npz',tmp_path/'fits'/case['id'])
    final_test(config,tmp_path,'cpu')
    selected=json.loads((tmp_path/'selection.json').read_text())
    result=json.loads((tmp_path/'test-results.json').read_text())
    assert len(selected['selected'])==1
    assert len(result['metrics'])==3
    for row in result['metrics']:
        for key in ['test_within','test_outside']:
            assert row[key]['count']==14
            assert 0<=row[key]['accuracy']<=1 and np.isfinite(row[key]['loss'])
    with pytest.raises(FileExistsError):
        final_test(config,tmp_path,'cpu')
