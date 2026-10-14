"""Finite nonlinear-function screen. No target-derived input features."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def event(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def targets(task, raw):
    if task == "sphere":
        return 4 * np.pi / 3 * raw[:, 0] ** 3
    if task == "product":
        return raw[:, 0] * raw[:, 1]
    if task == "chirp":
        return np.sin(6 * np.pi * raw[:, 0] ** 2)
    if task == "pairwise":
        return np.stack([(raw[:, i] - raw[:, j]) ** 2
                         for i in range(4) for j in range(i + 1, 4)], 1).mean(1)
    raise ValueError(task)


def make_data(task, seed=1729, n_train=4096):
    """Each task has a deterministic, isolated stream and train-only scaling."""
    task_seed = seed + int(hashlib.sha256(task.replace("_bits", "_scalar").encode())
                           .hexdigest()[:8], 16)
    rng = np.random.default_rng(task_seed)
    if not 1 <= n_train <= 65536:
        raise ValueError("Training sample count must be 1..65536")
    data = {}
    if task.startswith("modulo"):
        ids = rng.permutation(131072)
        outside = rng.permutation(131072) + 131072
        groups = {"train": ids[:n_train], "val": ids[65536:69632],
                  "test": ids[69632:77824], "ood_val": outside[:4096],
                  "ood": outside[4096:12288]}
        for split, n in groups.items():
            if task == "modulo_scalar":
                x = (n[:, None] / 65535.5 - 1).astype(np.float32)
            else:
                x = (((n[:, None] >> np.arange(18)) & 1) * 2 - 1).astype(np.float32)
            data[split] = {"x": x, "y": (n % 7).astype(np.int64),
                           "raw": n[:, None].astype(np.float64)}
        meta = {"task": task, "classification": True, "input_dim": x.shape[1],
                "output_dim": 7, "y_mean": 0.0, "y_std": 1.0}
    else:
        dim = {"sphere": 1, "product": 2, "chirp": 1, "pairwise": 4}[task]
        # Fixed pools give nested training sets and unchanged evaluation inputs.
        # Dyadic grid points remain distinct after float32 input encoding.
        def pool(count):
            return (np.column_stack([rng.choice(2**22, count, replace=False)
                                     for _ in range(dim)]) + .5) / 2**22
        inside, outside = pool(65536 + 1024 + 2048), pool(1024 + 2048)
        groups = {"train": inside[:n_train], "val": inside[65536:66560],
                  "test": inside[66560:], "ood_val": outside[:1024], "ood": outside[1024:]}
        # Signs are fixed before subsetting, too.
        signs = rng.choice([-1, 1], outside.shape)
        for split, unit in groups.items():
            raw = unit.copy()
            ood = split.startswith("ood")
            if task == "sphere":
                raw = 2 * raw + (2 if ood else 0)
                x = raw - 1
            elif task == "chirp":
                raw = raw + int(ood)
                x = 2 * raw - 1
            else:
                raw = ((1 + raw) * (signs[:1024] if split == "ood_val" else signs[1024:])
                       if ood else 2 * raw - 1)
                x = raw
            data[split] = {"x": x.astype(np.float32), "raw": raw,
                           "y_raw": targets(task, raw)}
        mean = float(data["train"]["y_raw"].mean())
        std = float(data["train"]["y_raw"].std())
        for values in data.values():
            values["y"] = ((values["y_raw"] - mean) / std).astype(np.float32)[:, None]
        meta = {"task": task, "classification": False, "input_dim": dim,
                "output_dim": 1, "y_mean": mean, "y_std": std}
    meta["training_samples"] = n_train
    meta["selection_policy"] = "within_range_validation_only; outside_validation_diagnostic"
    return data, meta


class MLP(nn.Module):
    def __init__(self, inputs, outputs, width, depth, residual=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(inputs, width)] +
            [nn.Linear(width, width) for _ in range(depth - 2)] + [nn.Linear(width, outputs)])
        self.residual = residual

    def forward(self, x):
        h = F.gelu(self.layers[0](x))
        for layer in self.layers[1:-1]:
            z = F.gelu(layer(h))
            h = h + z / math.sqrt(len(self.layers) - 2) if self.residual else z
        return self.layers[-1](h)


class Transformer(nn.Module):
    def __init__(self, inputs, outputs, width, depth):
        super().__init__()
        self.embedding = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, inputs + 1, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, outputs)

    def forward(self, x):
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1),
                       self.embedding(x.unsqueeze(-1))], 1) + self.position
        for block in self.blocks:
            h = block(h)
        return self.output(self.norm(h[:, 0]))


class Relational(nn.Module):
    def __init__(self, inputs, outputs, width, depth):
        super().__init__()
        self.embedding = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, inputs, width) * 0.02)
        self.relations = nn.ModuleList([nn.Sequential(nn.Linear(2 * width, width),
            nn.GELU(), nn.Linear(width, width), nn.GELU()) for _ in range(depth)])
        self.output = nn.Linear(width, outputs)

    def forward(self, x):
        h = self.embedding(x.unsqueeze(-1)) + self.position
        n = h.shape[1]
        for relation in self.relations:
            pairs = torch.cat([h[:, :, None, :].expand(-1, -1, n, -1),
                               h[:, None, :, :].expand(-1, n, -1, -1)], -1)
            h = h + relation(pairs).mean(2) / math.sqrt(len(self.relations))
        return self.output(h.mean(1))


class Hypernetwork(nn.Module):
    """Only theta=g.parameters() is learned. f has no stored Parameters."""
    def __init__(self, inputs, outputs, width, depth, f_depth=3):
        super().__init__()
        dims = [inputs] + [8] * (f_depth - 1) + [outputs]
        self.shapes = list(zip(dims[:-1], dims[1:]))
        self.generated_count = sum(a * b + b for a, b in self.shapes)
        self.g = MLP(inputs, self.generated_count, width, depth)
        with torch.no_grad():
            self.g.layers[-1].weight.mul_(0.1)
            offset = 0
            for a, b in self.shapes:
                self.g.layers[-1].bias[offset:offset+a*b].normal_(0, 1 / math.sqrt(a))
                offset += a * b
                self.g.layers[-1].bias[offset:offset+b].zero_()
                offset += b

    def forward(self, x):
        weights = self.g(x)
        h = x
        offset = 0
        for i, (a, b) in enumerate(self.shapes):
            w = weights[:, offset:offset+a*b].reshape(-1, a, b)
            offset += a * b
            bias = weights[:, offset:offset+b]
            offset += b
            h = torch.bmm(h.unsqueeze(1), w).squeeze(1) + bias
            if i + 1 < len(self.shapes):
                h = F.gelu(h)
        return h


def model_for(variant, meta, width):
    args = (meta["input_dim"], meta["output_dim"], width, variant["depth"])
    family = variant["family"]
    if family in ("mlp", "resmlp"):
        return MLP(*args, residual=family == "resmlp")
    if family == "transformer":
        return Transformer(*args)
    if family == "relational":
        return Relational(*args)
    if family == "hyper":
        return Hypernetwork(*args, f_depth=variant["f_depth"])
    raise ValueError(family)


def parameter_count(variant, meta, width):
    i, o, w, d = meta["input_dim"], meta["output_dim"], width, variant["depth"]
    family = variant["family"]
    if family == "hyper":
        dims = [i] + [8] * (variant["f_depth"] - 1) + [o]
        o = sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
    if family in ("mlp", "resmlp", "hyper"):
        return (i + 1) * w + (d - 2) * (w + 1) * w + (w + 1) * o
    if family == "transformer":
        return 2*w + (i+1)*w + w + d*(8*w*w + 11*w) + 2*w + (w+1)*o
    if family == "relational":
        return 2*w + i*w + d*(3*w*w + 2*w) + (w+1)*o
    raise ValueError(family)


def choose_width(variant, meta, target):
    step = 4 if variant["family"] == "transformer" else 1
    return min(range(8, 513, step), key=lambda w: abs(parameter_count(variant, meta, w) - target))


def loss_fn(pred, target, classification):
    return F.cross_entropy(pred, target) if classification else F.mse_loss(pred, target)


@torch.no_grad()
def predict(model, x, batch_size=256):
    model.eval()
    return torch.cat([model(part) for part in x.split(batch_size)])


def scores(pred, truth, meta):
    p, y = pred.detach().cpu(), truth.detach().cpu()
    if meta["classification"]:
        return {"loss": float(F.cross_entropy(p, y)),
                "accuracy": float((p.argmax(1) == y).float().mean())}
    error = (p.double() - y.double()).numpy().ravel()
    target = y.double().numpy().ravel()
    mse = float(np.mean(error ** 2))
    return {"loss": mse, "nrmse": math.sqrt(mse),
            "rmse": math.sqrt(mse) * meta["y_std"],
            "mae": float(np.abs(error).mean()) * meta["y_std"],
            "r2": float(1 - mse / np.var(target))}


def cases(config):
    return [{"task": task, "variant": variant, "seed": seed, "lr": lr,
             "training_samples": n,
             "case_id": f"{task}--{variant['name']}--n{n}--s{seed}--lr{lr:g}"}
            for seed in config["seeds"] for variant in config["variants"]
            for task in config["tasks"] for n in config.get("training_samples", [4096])
            for lr in config["learning_rates"]]


def train_case(config, case, output, device="cuda"):
    torch.set_num_threads(1)
    torch.manual_seed(case["seed"])
    np.random.seed(case["seed"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU run requires CUDA; CPU fallback is forbidden")
    if device == "cuda":
        torch.cuda.manual_seed_all(case["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # Disable fused encoder inference so eval and train use the same block path.
    torch.backends.mha.set_fastpath_enabled(False)
    directory = Path(output) / case["case_id"]
    directory.mkdir(parents=True, exist_ok=False)
    history = directory / "metrics.jsonl"
    started = time.time()
    event(history, {"event": "started", "case": case, "unix_time": started})
    data, meta = make_data(case["task"], config["data_seed"], case.get("training_samples", 4096))
    tensors = {split: {key: torch.from_numpy(values[key]).to(device) for key in ("x", "y")}
               for split, values in data.items()}
    width = choose_width(case["variant"], meta, config["parameter_target"])
    model = model_for(case["variant"], meta, width).to(device)
    params = sum(p.numel() for p in model.parameters())
    assert params == parameter_count(case["variant"], meta, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=case["lr"], weight_decay=0.0)
    # Dedicated, identical batch stream across architecture and LR for each seed.
    sampler = torch.Generator(device=device).manual_seed(case["seed"] + 90000)
    best, best_state, best_step = float("inf"), None, 0
    train = tensors["train"]
    visited = torch.zeros(len(train["x"]), dtype=torch.bool, device=device)
    for step in range(1, config["steps"] + 1):
        model.train()
        ids = torch.randint(len(train["x"]), (config["batch_size"],), generator=sampler, device=device)
        visited[ids] = True
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train["x"][ids]), train["y"][ids], meta["classification"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % config["eval_every"] == 0 or step == config["steps"]:
            val = tensors["val"]
            val_loss = float(loss_fn(predict(model, val["x"]), val["y"], meta["classification"]))
            outside_val = tensors["ood_val"]
            outside_val_loss = float(loss_fn(predict(model, outside_val["x"]),
                                             outside_val["y"], meta["classification"]))
            if val_loss < best:
                best, best_step = val_loss, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            event(history, {"event": "optimizer_progress", "step": step, "train_batch_loss": float(loss.detach()),
                            "validation_loss": val_loss, "gradient_norm": float(norm),
                            "outside_validation_loss": outside_val_loss,
                            "elapsed_seconds": time.time() - started})
    model.load_state_dict(best_state)
    # No held-out test or OOD data is touched during optimization or selection.
    metrics, predictions = {}, {}
    for split, values in tensors.items():
        pred = predict(model, values["x"])
        metrics[split] = scores(pred, values["y"], meta)
        if split in ("val", "test", "ood_val", "ood"):
            predictions[f"{split}_prediction"] = pred.cpu().numpy()
            predictions[f"{split}_target"] = values["y"].cpu().numpy()
    np.savez_compressed(directory / "predictions.npz", **predictions)
    torch.save({"model": best_state, "variant": case["variant"], "meta": meta,
                "width": width, "step": best_step, "case": case}, directory / "best.pt")
    result = {**case, "meta": meta, "width": width, "parameters": params,
              "generated_parameters": getattr(model, "generated_count", 0),
              "architecture_sha256": hashlib.sha256(json.dumps({"variant": case["variant"],
                  "input_dim": meta["input_dim"], "output_dim": meta["output_dim"], "width": width,
                  "suite_sha256": sha256(__file__)}, sort_keys=True).encode()).hexdigest(),
              "unique_training_examples_seen": int(visited.sum()),
              "training_examples_seen": config["steps"] * config["batch_size"],
              "exposure_unit": "numeric_examples", "text_tokens": None,
              "optimizer_steps": config["steps"], "best_step": best_step,
              "validation_loss": best, "metrics": metrics, "device": device,
              "duration_seconds": time.time() - started, "status": "completed",
              "checkpoint_sha256": sha256(directory / "best.pt"),
              "predictions_sha256": sha256(directory / "predictions.npz")}
    write_json(directory / "result.json", result)
    event(history, {"event": "completed", "step": config["steps"],
                    "validation_loss": best, "elapsed_seconds": result["duration_seconds"]})
    return result


def baseline_results(config, n_train=4096):
    result = {}
    for task in config["tasks"]:
        data, meta = make_data(task, config["data_seed"], n_train)
        group = {}
        if meta["classification"]:
            counts = np.bincount(data["train"]["y"], minlength=7)
            majority = int(counts.argmax())
            for split, values in data.items():
                group[split] = {"accuracy": float((values["y"] == majority).mean()),
                                "uniform_chance": 1/7}
        else:
            for split, values in data.items():
                group[split] = scores(torch.zeros_like(torch.from_numpy(values["y"])),
                                      torch.from_numpy(values["y"]), meta)
            if task == "sphere":
                coef = np.polynomial.polynomial.polyfit(data["train"]["raw"].ravel(),
                                                        data["train"]["y_raw"], 3)
                group["analytic_cubic_control"] = {split: float(np.sqrt(np.mean((
                    np.polynomial.polynomial.polyval(values["raw"].ravel(), coef) - values["y_raw"])**2)))
                    for split, values in data.items()}
        result[task] = group
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.inventory:
        rows = []
        for task in config["tasks"]:
            _, meta = make_data(task, config["data_seed"])
            for variant in config["variants"]:
                width = choose_width(variant, meta, config["parameter_target"])
                rows.append({"task": task, "variant": variant["name"], "width": width,
                             "parameters": parameter_count(variant, meta, width)})
        write_json(args.output, {"models": rows, "fits": len(cases(config)),
                                 "baselines": {f"n{n}": baseline_results(config, n)
                                               for n in config.get("training_samples", [4096])}})
        return
    if args.case_index is None:
        parser.error("--case-index is required for training")
    result = train_case(config, cases(config)[args.case_index], args.output, args.device)
    print(json.dumps({k: result[k] for k in ("case_id", "status", "duration_seconds", "metrics")}), flush=True)


if __name__ == "__main__":
    main()
