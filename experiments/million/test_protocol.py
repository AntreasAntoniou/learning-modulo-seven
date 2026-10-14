import copy

import json

from pathlib import Path

import tarfile

import numpy as np

import pytest

import torch

from common import digest

from data import encode, split_ids

from models import choose_width, model_for, parameter_count

from selection import select

from train import batch_spans, better, cases, fit, order_for, unpack

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())

def tiny():
    c = copy.deepcopy(CONFIG)
    c.update(training_samples=100, validation_samples_per_domain=14, test_samples_per_domain=14,
             within_stop=384, outside_stop=512, input_bits=9, epochs=2, batch_size=16,
             validation_every_samples=50, checkpoint_every_samples=50, evaluation_batch_size=14,
             parameter_target=128, encodings=["bits"], variants=[CONFIG["variants"][0]])
    return c

def test_actual_splits_no_leakage_and_inputs_distinct():
    ids = split_ids(CONFIG)
    assert len(ids["train"]) == 1_000_000
    all_ids = np.concatenate(list(ids.values()))
    assert len(np.unique(all_ids)) == len(all_ids)
    assert len(np.unique(encode(all_ids, "scalar", CONFIG))) == len(all_ids)
    for name, values in ids.items():
        assert len(np.unique(values % 7)) == 7
        assert (values >= CONFIG["within_stop"]).all() if "outside" in name else (values < CONFIG["within_stop"]).all()
    bits = encode(ids["train"], "bits", CONFIG)
    assert (bits.min(0) == -1).all() and (bits.max(0) == 1).all()
    assert not ((bits[:, -1] == 1) & (bits[:, -2] == 1)).any()

def test_sample_boundaries_and_no_omissions():
    spans = list(batch_spans(1_000_000, 128, 5000))
    assert len(spans) == 8000
    assert sum(b-a for a, b in spans) == 1_000_000
    assert [b for a, b in spans if b % 5000 == 0] == list(range(5000, 1_000_001, 5000))
    assert set(b-a for a, b in spans) == {8, 128}
    assert all(a == previous[1] for previous, (a, b) in zip(spans, spans[1:]))

def test_cases_and_paired_order():
    assert len(cases(CONFIG)) == len({c["id"] for c in cases(CONFIG)}) == 576
    np.testing.assert_array_equal(order_for(100, 101, 1), order_for(100, 101, 1))
    assert set(order_for(100, 101, 1)) == set(range(100))
    assert not np.array_equal(order_for(100, 101, 0), order_for(100, 101, 1))

@pytest.mark.parametrize("variant", CONFIG["variants"], ids=lambda v: v["name"])
@pytest.mark.parametrize("input_dim", [1, 22])
def test_models_and_parameters(variant, input_dim):
    torch.set_num_threads(1)
    meta = {"input_dim": input_dim, "output_dim": 7}
    width = choose_width(variant, meta, CONFIG["parameter_target"])
    model = model_for(variant, meta, width)
    assert parameter_count(variant, meta, width) == sum(p.numel() for p in model.parameters())
    loss = torch.nn.functional.cross_entropy(model(torch.randn(4, input_dim)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if variant["family"] == "hyper":
        assert all(name.startswith("g.") for name, _ in model.named_parameters())

def test_exact_resume_and_metric_segment_chain(tmp_path):
    config = tiny()
    ids = split_ids(config)
    data = tmp_path / "trainval.npz"
    np.savez(data, **{k: v for k, v in ids.items() if not k.startswith("test")})
    case = cases(config)[0]
    full, full_bundle = fit(config, case, data, tmp_path / "full")
    part, part_bundle = fit(config, case, data, tmp_path / "part", stop_after=50)
    resumed, resumed_bundle = fit(config, case, data, tmp_path / "resumed", resume=part_bundle)
    states = []
    for i, bundle in enumerate((full_bundle, resumed_bundle)):
        directory = tmp_path / str(i)
        directory.mkdir()
        unpack(bundle, directory)
        states.append(torch.load(directory / "state.pt", weights_only=False))
    for key in states[0]["model"]:
        torch.testing.assert_close(states[0]["model"][key], states[1]["model"][key], rtol=0, atol=0)
    assert full["best"] == resumed["best"]
    assert full["samples"] == resumed["samples"] == 200
    assert full["steps"] == resumed["steps"] == 16
    with tarfile.open(part_bundle) as tar:
        rows = [json.loads(line) for line in tar.extractfile("train.jsonl")]
        vals = [json.loads(line) for line in tar.extractfile("validation.jsonl")]
    assert sum(r["batch_size"] for r in rows) == 50
    assert vals[0]["train_loss"] == pytest.approx(sum(r["loss"] * r["batch_size"] for r in rows) / 50)

def fake_results(config):
    return [{"case_id": c["id"], "identity": {"case": c, "config_digest": digest(config),
             "source_commit": "a" * 40, "trainval_sha256": "b" * 64},
             "status": "complete", "samples": config["training_samples"] * config["epochs"],
             "best": {"loss": 1 + c["weight_decay"], "accuracy": 0.5}, "checkpoint": {}}
             for c in cases(config)]

def test_selection_all_seeds_validation_only_and_test_gate():
    results = fake_results(CONFIG)
    selected = select(CONFIG, results)
    assert len(selected["selected"]) == 32
    assert all(len(r["models"]) == 3 and r["weight_decay"] == 0 for r in selected["selected"])
    for r in results:
        r["test_loss"] = -100 * r["identity"]["case"]["weight_decay"]
    assert [r["weight_decay"] for r in select(CONFIG, results)["selected"]] == [0] * 32
    with pytest.raises(ValueError):
        select(CONFIG, results[:-1])
    with pytest.raises(ValueError):
        select(CONFIG, results + results[:1])
    results[0]["status"] = "paused"
    with pytest.raises(ValueError):
        select(CONFIG, results)

def test_loss_primary_accuracy_only_tiebreak():
    assert better({"loss": 1, "accuracy": .1}, {"loss": 2, "accuracy": .9})
    assert better({"loss": 1, "accuracy": .9}, {"loss": 1, "accuracy": .1})
    assert not better({"loss": 1, "accuracy": .9}, {"loss": 1, "accuracy": .9})
