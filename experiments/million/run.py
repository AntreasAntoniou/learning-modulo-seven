"""Local reproduction: prepare, train, then select and test exactly once."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from common import digest, sha, write_json
from data import prepare, encode
from models import model_for
from selection import select
from train import cases, evaluate, fit, unpack

ROOT = Path(__file__).resolve().parent

def final_test(config, root, device):
    results = [json.loads(p.read_text()) for p in sorted((root / "fits").glob("*/result.json"))]
    chosen = select(config, results)  # Refuses missing, duplicate, or mismatched cases.
    manifest = json.loads((root / "data/manifest.json").read_text())
    if chosen["trainval_sha256"] != manifest["trainval_sha256"]:
        raise ValueError("Selected models belong to another dataset")
    test_path = root / "data/sealed-test.npz"
    if sha(test_path) != manifest["sealed_test_sha256"]:
        raise ValueError("Test input has changed")
    # Create-only fence precedes reading test values or predicting from them.
    with (root / "final-test-fence.json").open("x") as stream:
        json.dump({"selection_digest": digest(chosen), "test_sha256": sha(test_path)}, stream)
    write_json(root / "selection.json", chosen)
    tests = np.load(test_path, allow_pickle=False)
    if set(tests.files) != {"test_within", "test_outside"}:
        raise ValueError("Unexpected test splits")
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for group in chosen["selected"]:
        for record in group["models"]:
            checkpoint = Path(record["checkpoint"]["local"])
            if sha(checkpoint) != record["checkpoint"]["sha256"]:
                raise ValueError("Selected checkpoint has changed")
            with tempfile.TemporaryDirectory() as folder:
                unpack(checkpoint, folder)
                state = torch.load(Path(folder) / "state.pt", map_location="cpu", weights_only=False)
            if state["identity"] != record["identity"] or state["best_samples"] != record["best_samples"]:
                raise ValueError("Selected state identity differs")
            case = state["identity"]["case"]
            model = model_for(case["variant"], state["meta"], state["width"]).to(device)
            model.load_state_dict(state["best_model"])
            row = {"case_id": case["id"], "best_samples": state["best_samples"]}
            for key in tests.files:
                x = torch.from_numpy(encode(tests[key], case["encoding"], config)).to(device)
                y = torch.from_numpy(tests[key] % 7).long().to(device)
                row[key] = evaluate(model, x, y, config["evaluation_batch_size"])
            rows.append(row)
    write_json(root / "test-results.json", {"selection_digest": digest(chosen), "metrics": rows})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "list", "train", "test"])
    parser.add_argument("--root", type=Path, default=Path("runs/million"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--case-id")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.root.resolve()
    if args.action == "list":
        for case in cases(config):
            print(case["id"])
    elif args.action == "prepare":
        root.mkdir(parents=True, exist_ok=True)
        prepare(config, root / "data")
    elif args.action == "train":
        matches = [c for c in cases(config) if c["id"] == args.case_id]
        if len(matches) != 1:
            parser.error("--case-id must identify a case from the list command")
        # Reproduction identity describes the exported scientific files, not a new result claim.
        source_id = digest({p: sha(ROOT / p) for p in ["models.py", "data.py", "train.py", "selection.py", "common.py"]})
        fit(config, matches[0], root / "data/trainval.npz", root / "fits" / args.case_id,
            device=args.device, source_commit=source_id,
            resume=str(args.resume.resolve()) if args.resume else None)
    else:
        final_test(config, root, args.device)

if __name__ == "__main__":
    main()
