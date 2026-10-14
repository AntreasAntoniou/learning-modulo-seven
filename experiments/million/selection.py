"""Freeze validation-only choices after every preregistered fit completes."""
from collections import defaultdict
from statistics import mean
from common import digest
from train import cases

def select(config, results):
    expected = {case["id"]: case for case in cases(config)}
    indexed = {r["case_id"]: r for r in results}
    if len(indexed) != len(results) or set(indexed) != set(expected):
        raise ValueError("Exactly one completed result for every preregistered case is required")
    groups = defaultdict(list)
    source_commits, dataset_hashes = set(), set()
    for case_id, result in indexed.items():
        identity = result["identity"]
        if (identity["case"] != expected[case_id] or identity["config_digest"] != digest(config)
            or result["status"] != "complete"
            or result["samples"] != config["training_samples"] * config["epochs"]):
            raise ValueError("Incomplete or mismatched fit cannot participate in selection")
        source_commits.add(identity["source_commit"])
        dataset_hashes.add(identity["trainval_sha256"])
        case = expected[case_id]
        groups[(case["encoding"], case["variant"]["name"], case["weight_decay"], case["lr"])].append(result)
    if len(source_commits) != 1 or len(dataset_hashes) != 1:
        raise ValueError("Source or data changed within the sweep")
    candidates = defaultdict(list)
    for (encoding, variant, wd, lr), rows in groups.items():
        if sorted(r["identity"]["case"]["seed"] for r in rows) != sorted(config["seeds"]):
            raise ValueError("Every configuration requires all three seeds")
        candidates[(encoding, variant)].append({"encoding": encoding, "variant": variant,
            "weight_decay": wd, "lr": lr, "mean_validation_loss": mean(r["best"]["loss"] for r in rows),
            "mean_validation_accuracy": mean(r["best"]["accuracy"] for r in rows),
            "models": sorted(rows, key=lambda r: r["identity"]["case"]["seed"])})
    key = lambda r: (r["mean_validation_loss"], -r["mean_validation_accuracy"], r["weight_decay"], r["lr"], r["variant"])
    winners = [min(rows, key=key) for _, rows in sorted(candidates.items())]
    overall = {encoding: min([r for r in winners if r["encoding"] == encoding], key=key)["variant"]
               for encoding in config["encodings"]}
    return {"selection_rule": "Mean best within-validation cross-entropy across seeds; accuracy tie-break",
            "config_digest": digest(config), "source_commit": source_commits.pop(),
            "trainval_sha256": dataset_hashes.pop(), "expected_fit_count": len(expected),
            "overall_validation_winners": overall, "selected": winners}
