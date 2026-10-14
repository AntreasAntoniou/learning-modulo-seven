"""Disjoint integer domains; test values never enter the training data file."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import sha, write_json

def split_ids(config):
    n, v, t = (config[k] for k in ("training_samples", "validation_samples_per_domain", "test_samples_per_domain"))
    low, high = config["within_stop"], config["outside_stop"]
    if not (n + v + t <= low and v + t <= high - low):
        raise ValueError("Domains cannot hold disjoint splits")
    rng = np.random.default_rng(config["data_seed"])
    inside = rng.permutation(low).astype(np.int64)
    outside = rng.permutation(high - low).astype(np.int64) + low
    return {"train": inside[:n], "val_within": inside[n:n+v],
            "test_within": inside[n+v:n+v+t], "val_outside": outside[:v],
            "test_outside": outside[v:v+t]}

def encode(ids, encoding, config):
    if encoding == "scalar":
        # Compute in float64, round once. No learned or split-dependent scaler.
        return (ids[:, None].astype(np.float64) / ((config["within_stop"] - 1) / 2) - 1).astype(np.float32)
    if encoding == "bits":
        return (((ids[:, None] >> np.arange(config["input_bits"])) & 1) * 2 - 1).astype(np.float32)
    raise ValueError(encoding)

def prepare(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ids = split_ids(config)
    trainval = {k: v for k, v in ids.items() if not k.startswith("test_")}
    tests = {k: v for k, v in ids.items() if k.startswith("test_")}
    np.savez_compressed(output / "trainval.npz", **trainval)
    np.savez_compressed(output / "sealed-test.npz", **tests)
    manifest = {"splits": {}, "trainval_sha256": sha(output / "trainval.npz"),
                "sealed_test_sha256": sha(output / "sealed-test.npz")}
    import hashlib
    for name, values in ids.items():
        manifest["splits"][name] = {"count": len(values), "min": int(values.min()), "max": int(values.max()),
            "ids_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "class_counts": np.bincount(values % 7, minlength=7).tolist()}
    joined = np.concatenate(list(ids.values()))
    if len(np.unique(joined)) != len(joined):
        raise AssertionError("Split leakage")
    if len(np.unique(encode(joined, "scalar", config))) != len(joined):
        raise AssertionError("Float32 scalar inputs alias integers")
    bits = encode(ids["train"], "bits", config)
    if not np.all((bits.min(0) == -1) & (bits.max(0) == 1)):
        raise AssertionError("An input bit never varies during training")
    manifest["audit"] = {"disjoint": True, "scalar_float32_unique": True, "all_training_bits_vary": True}
    write_json(output / "manifest.json", manifest)
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(Path(args.config).read_text()), args.output), indent=2))
