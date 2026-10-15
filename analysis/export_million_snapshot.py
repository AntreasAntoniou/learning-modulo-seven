"""Export a dated, validation-only draft from already audited local histories.

Supply the private input locations explicitly. Only the scientific allowlist below
is exported; storage locators, attempts and infrastructure metadata are excluded.
This does not freeze a sweep selection or read test data.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


CURVE_RECIPES = [
    ("hyper_g2_f3", 0.0003, 0.0),
    ("resmlp20", 0.003, 0.0),
    ("resmlp50", 0.003, 0.0),
    ("hyper_g8_f5", 0.003, 0.0),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(value):
    assert set(value) == {"accuracy", "loss", "count"}
    assert value["count"] == 8192
    assert math.isfinite(value["loss"]) and value["loss"] >= 0
    assert math.isfinite(value["accuracy"]) and 0 <= value["accuracy"] <= 1
    return dict(value)


def point(value):
    assert math.isfinite(value["train_loss"]) and value["train_loss"] >= 0
    assert math.isfinite(value["train_accuracy"]) and 0 <= value["train_accuracy"] <= 1
    return {"case_id": value["case_id"], "samples": value["samples"],
            "step": value["step"], "train_loss": value["train_loss"],
            "train_accuracy": value["train_accuracy"],
            "val_within": metrics(value["val_within"]),
            "val_outside": metrics(value["val_outside"])}


def export(args):
    results = json.loads(args.completed_results.read_text())
    index = json.loads(args.history_index.read_text())
    audits = {r["case_id"]: r for r in index["cases"]}
    config = json.loads(args.config.read_text())
    expected = {f"{enc}-{v['name']}-wd{wd}-lr{lr}-s{seed}"
                for enc in ["scalar", "bits"] for v in config["variants"]
                for wd in range(3) for lr in range(2) for seed in [101, 202, 303]}
    assert len(expected) == 576
    assert len(results) == args.expected_complete
    assert len({r["case_id"] for r in results}) == len(results)
    assert {r["case_id"] for r in results} <= expected
    identities = {(r["identity"]["source_commit"], r["identity"]["trainval_sha256"],
                   r["identity"]["config_digest"]) for r in results}
    assert len(identities) == 1
    source, data_sha, config_digest = identities.pop()
    fits, curves = [], []
    for result in sorted(results, key=lambda r: r["case_id"]):
        key = result["case_id"]
        assert result["status"] == "complete" and result["samples"] == 10_000_000
        assert result["steps"] == 80_000
        case = result["identity"]["case"]
        assert set(case) == {"id", "encoding", "variant", "lr", "weight_decay", "seed"}
        assert case["id"] == key and case["variant"] in config["variants"]
        wi = [0.0, 0.01, 0.1].index(case["weight_decay"])
        li = [0.0003, 0.003].index(case["lr"])
        assert key == f"{case['encoding']}-{case['variant']['name']}-wd{wi}-lr{li}-s{case['seed']}"
        path = args.history_root / key / "validation.jsonl.gz"
        record = next(a for a in audits[key]["artifacts"] if a["name"] == path.name)
        assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]
        with gzip.open(path, "rt") as stream:
            values = [point(json.loads(line)) for line in stream]
        assert [v["samples"] for v in values] == list(range(5000, 10_000_001, 5000))
        assert all(v["case_id"] == key and v["step"] == v["samples"] // 125 for v in values)
        best = min(values, key=lambda v: (v["val_within"]["loss"],
                   -v["val_within"]["accuracy"], v["samples"]))
        assert best["samples"] == result["best_samples"]
        assert best["val_within"] == result["best"]
        fits.append({"case_id": key, "case": case, "parameters": result["parameters"],
                     "width": result["width"], "samples": result["samples"],
                     "steps": result["steps"], "best": best, "final": values[-1],
                     "validation_history_sha256": record["sha256"],
                     "checkpoint_archive_sha256": result["checkpoint"]["sha256"]})
        recipe = (case["variant"]["name"], case["lr"], case["weight_decay"])
        if case["encoding"] == "bits" and recipe in CURVE_RECIPES:
            curves.extend(values)
    assert len(curves) == len(CURVE_RECIPES) * 3 * 2000
    snapshot = {"schema": 1, "snapshot_utc": args.snapshot_utc,
                "status": "partial_validation_draft", "expected_fits": 576,
                "completed_fits": len(fits), "test_evaluated": False,
                "sweep_selection_frozen": False,
                "missing_case_ids": sorted(expected - {r["case_id"] for r in fits}),
                "source_commit": source, "trainval_sha256": data_sha,
                "config_digest": config_digest, "config_file_sha256": sha(args.config),
                "input_completed_results_sha256": sha(args.completed_results),
                "input_history_audit_index_sha256": sha(args.history_index),
                "curve_recipes": CURVE_RECIPES,
                "curve_choice": "Post-hoc illustrative recipes; all three seeds retained. Not sweep selection.",
                "fits": fits}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    with (args.output / "illustrative-curves.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for value in curves:
                stream.write((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
    print(json.dumps({"completed_fits": len(fits), "missing_fits": len(snapshot["missing_case_ids"]),
                      "illustrative_curve_records": len(curves), "test_evaluated": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["completed-results", "history-index", "history-root", "config", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-complete", type=int, required=True)
    parser.add_argument("--snapshot-utc", required=True)
    export(parser.parse_args())
