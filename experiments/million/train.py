"""Exact sample-indexed training, with recoverable immutable metric segments."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import tarfile
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import digest, download, publish, publish_json, sha, write_json
from data import encode
from models import choose_width, model_for

def cases(config):
    return [{"id": f"{encoding}-{v['name']}-wd{wi}-lr{li}-s{seed}", "encoding": encoding,
             "variant": v, "seed": seed, "weight_decay": wd, "lr": lr}
            for encoding in config["encodings"] for v in config["variants"]
            for wi, wd in enumerate(config["weight_decays"])
            for li, lr in enumerate(config["learning_rates"]) for seed in config["seeds"]]

def batch_spans(n, batch, interval, offset=0):
    while offset < n:
        end = min(n, offset + batch, (offset // interval + 1) * interval)
        yield offset, end
        offset = end

def order_for(n, seed, epoch):
    return np.random.default_rng(seed + 90000 + 1000003 * epoch).permutation(n)

def better(metrics, best):
    return best is None or (metrics["loss"], -metrics["accuracy"]) < (best["loss"], -best["accuracy"])

def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def evaluate(model, x, y, batch):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model(x[start:start+batch])
            loss += float(F.cross_entropy(logits, y[start:start+batch], reduction="sum"))
            correct += int((logits.argmax(-1) == y[start:start+batch]).sum())
    return {"loss": loss / len(x), "accuracy": correct / len(x), "count": len(x)}

def unpack(archive, directory):
    with tarfile.open(archive) as tar:
        tar.extractall(directory, filter="data")

def fit(config, case, data_path, output, *, device="cpu", source_commit="local-test",
        remote_prefix=None, resume=None, stop_after=None):
    """stop_after is a validation-boundary canary limit, never a production override."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(case["seed"])
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity = {"source_commit": source_commit, "config_digest": digest(config),
                "case": case, "trainval_sha256": sha(data_path)}
    loaded = np.load(data_path, allow_pickle=False)
    if set(loaded.files) != {"train", "val_within", "val_outside"}:
        raise ValueError("Only training and validation splits may be opened")
    data = {key: (torch.from_numpy(encode(loaded[key], case["encoding"], config)).to(device),
                  torch.from_numpy(loaded[key] % 7).long().to(device)) for key in loaded.files}
    meta = {"input_dim": data["train"][0].shape[1], "output_dim": 7}
    width = choose_width(case["variant"], meta, config["parameter_target"])
    model = model_for(case["variant"], meta, width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=case["lr"], weight_decay=case["weight_decay"],
                                 betas=tuple(config["betas"]), eps=config["epsilon"])
    samples = step = 0
    best = best_model = None
    best_samples = None
    previous = None
    if resume:
        directory = output / "resume"
        directory.mkdir()
        if isinstance(resume, dict):
            download(resume, directory / "bundle.tar.gz")
            previous = resume
            unpack(directory / "bundle.tar.gz", directory)
        else:
            unpack(resume, directory)
            previous = {"local": str(resume), "sha256": sha(resume)}
        state = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume identity differs from this experiment")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        samples, step = state["samples"], state["step"]
        best, best_model, best_samples = state["best"], state["best_model"], state["best_samples"]
        torch.set_rng_state(state["cpu_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    n = len(data["train"][0])
    if n != config["training_samples"] or n % config["validation_every_samples"]:
        raise ValueError("Every epoch must end at a validation boundary")
    total = n * config["epochs"]
    target = min(total, stop_after) if stop_after is not None else total
    if target % config["validation_every_samples"] or target <= samples:
        raise ValueError("Stop point must be a later validation boundary")
    requested_stop = [False]
    old_handler = signal.signal(signal.SIGTERM, lambda *_: requested_stop.__setitem__(0, True))
    batch_log = open(output / "train.jsonl", "w")
    val_log = open(output / "validation.jsonl", "w")
    window_loss = window_correct = window_n = 0
    segment_start = samples
    checkpoint_pins = []
    started = time.time()
    last_bundle = None
    try:
        while samples < target:
            epoch, offset = divmod(samples, n)
            order = order_for(n, case["seed"], epoch)
            for lo, hi in batch_spans(n, config["batch_size"], config["validation_every_samples"], offset):
                index = torch.from_numpy(order[lo:hi]).to(device)
                x, y = data["train"][0][index], data["train"][1][index]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                value = float(loss.detach())
                correct = int((logits.detach().argmax(-1) == y).sum())
                if not math.isfinite(value):
                    raise RuntimeError("Nonfinite training loss")
                samples += len(y)
                step += 1
                window_loss += value * len(y)
                window_correct += correct
                window_n += len(y)
                batch_log.write(json.dumps({"samples": samples, "step": step, "batch_size": len(y),
                    "loss": value, "accuracy": correct / len(y), "gradient_norm": float(norm)}) + "\n")
                if samples % config["validation_every_samples"] == 0:
                    within = evaluate(model, *data["val_within"], config["evaluation_batch_size"])
                    outside = evaluate(model, *data["val_outside"], config["evaluation_batch_size"])
                    if not all(math.isfinite(m["loss"]) for m in (within, outside)):
                        raise RuntimeError("Nonfinite validation loss")
                    if better(within, best):
                        best, best_model, best_samples = within, cpu_state(model), samples
                    row = {"case_id": case["id"], "samples": samples, "step": step,
                           "train_loss": window_loss / window_n, "train_accuracy": window_correct / window_n,
                           "val_within": within, "val_outside": outside, "best_samples": best_samples,
                           "elapsed_seconds": time.time() - started}
                    val_log.write(json.dumps(row) + "\n")
                    for stream in (batch_log, val_log):
                        stream.flush()
                        os.fsync(stream.fileno())
                    print(json.dumps({"event": "validation", **row}), flush=True)
                    window_loss = window_correct = window_n = 0
                    should_stop = requested_stop[0] or samples >= target
                    if samples % config["checkpoint_every_samples"] == 0 or should_stop:
                        state = {"identity": identity, "model": cpu_state(model), "optimizer": optimizer.state_dict(),
                                 "best_model": best_model, "best": best, "best_samples": best_samples,
                                 "samples": samples, "step": step, "width": width, "meta": meta,
                                 "cpu_rng": torch.get_rng_state(),
                                 "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []}
                        torch.save(state, output / "state.pt")
                        write_json(output / "segment.json", {"start": segment_start, "end": samples,
                            "previous": previous, "identity": identity, "step": step})
                        bundle = output / f"checkpoint-{samples:09d}.tar.gz"
                        with tarfile.open(bundle, "w:gz") as tar:
                            for name in ("state.pt", "segment.json", "train.jsonl", "validation.jsonl"):
                                tar.add(output / name, arcname=name)
                        last_bundle = str(bundle)
                        if remote_prefix:
                            previous = publish(bundle, f"{remote_prefix}/checkpoint-{samples:09d}.tar.gz")
                            receipt = {"case_id": case["id"], "samples": samples, "checkpoint": previous,
                                       "identity": identity, "best": best, "best_samples": best_samples}
                            publish_json(receipt, f"{remote_prefix}/checkpoint-{samples:09d}.json", output)
                            checkpoint_pins.append(previous)
                            bundle.unlink()
                        else:
                            previous = {"local": str(bundle), "sha256": sha(bundle)}
                        segment_start = samples
                        for stream in (batch_log, val_log):
                            stream.seek(0)
                            stream.truncate()
                    if should_stop:
                        break
            if requested_stop[0]:
                break
        result = {"identity": identity, "case_id": case["id"], "samples": samples, "steps": step,
                  "status": "complete" if samples == total else "paused", "best": best,
                  "best_samples": best_samples, "checkpoint": previous,
                  "parameters": sum(p.numel() for p in model.parameters()), "width": width,
                  "elapsed_seconds_this_attempt": time.time() - started,
                  "device": device, "torch_version": torch.__version__}
        write_json(output / "result.json", result)
        if remote_prefix:
            publish(output / "result.json", remote_prefix + "/result.json")
        return result, last_bundle
    finally:
        batch_log.close()
        val_log.close()
        signal.signal(signal.SIGTERM, old_handler)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    a = p.parse_args()
    request = json.loads(Path(a.request).read_text())
    fit(**request)
