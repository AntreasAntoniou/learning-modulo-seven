"""Figures and descriptive summaries for the dated, incomplete 560-fit snapshot."""
from collections import defaultdict
import csv
import gzip
import json
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
from make_figures import ROOT, FIG, TABLE, COLORS, SEEDS, NAMES, save, style

DATA = ROOT / "results/million/draft-2026-10-15"


def build():
    snapshot = json.loads((DATA / "snapshot.json").read_text())
    assert snapshot["status"] == "partial_validation_draft"
    assert not snapshot["test_evaluated"] and not snapshot["sweep_selection_frozen"]
    rows = snapshot["fits"]
    summary = {"completed_fits": len(rows), "expected_fits": 576,
               "missing_fits": len(snapshot["missing_case_ids"]),
               "descriptive_accuracy_threshold": 0.99,
               "threshold_status": "Post-hoc descriptive count, not a selection rule or hypothesis test.",
               "encodings": {}, "architectures": {}, "complete_recipe_cells": 0,
               "recipe_cells_with_all_seeds_both_ge99": 0}
    for encoding in ["bits", "scalar"]:
        subset = [r for r in rows if r["case"]["encoding"] == encoding]
        within = [r["best"]["val_within"] for r in subset]
        outside = [r["best"]["val_outside"] for r in subset]
        summary["encodings"][encoding] = {
            "completed": len(subset), "expected": 288,
            "within_ge99": sum(v["accuracy"] >= .99 for v in within),
            "both_ge99": sum(w["accuracy"] >= .99 and o["accuracy"] >= .99
                             for w, o in zip(within, outside)),
            "within_accuracy_min": min(v["accuracy"] for v in within),
            "within_accuracy_max": max(v["accuracy"] for v in within),
            "within_loss_min": min(v["loss"] for v in within),
            "within_loss_max": max(v["loss"] for v in within)}
        for architecture in NAMES:
            rr = [r for r in subset if r["case"]["variant"]["name"] == architecture]
            summary["architectures"][encoding + "/" + architecture] = {
                "completed": len(rr), "expected": 18,
                "within_ge99": sum(r["best"]["val_within"]["accuracy"] >= .99 for r in rr),
                "both_ge99": sum(min(r["best"][s]["accuracy"] for s in ["val_within", "val_outside"]) >= .99 for r in rr)}
    # All 192 planned recipe cells are represented. Incomplete cells have no mean.
    cells = []
    for encoding in ["bits", "scalar"]:
        for architecture in NAMES:
            for lr in [.0003, .003]:
                for wd in [0., .01, .1]:
                    rr = sorted([r for r in rows if r["case"]["encoding"] == encoding
                        and r["case"]["variant"]["name"] == architecture
                        and r["case"]["lr"] == lr and r["case"]["weight_decay"] == wd],
                        key=lambda r: r["case"]["seed"])
                    cell = {"encoding": encoding, "architecture": architecture,
                            "learning_rate": lr, "weight_decay": wd, "n": len(rr),
                            "complete": len(rr) == 3}
                    for split in ["val_within", "val_outside"]:
                        for metric in ["loss", "accuracy"]:
                            values = [r["best"][split][metric] for r in rr]
                            cell[split + "_" + metric + "_mean"] = mean(values) if len(rr) == 3 else None
                            cell[split + "_" + metric + "_sd"] = stdev(values) if len(rr) == 3 else None
                    cells.append(cell)
                    if len(rr) == 3:
                        summary["complete_recipe_cells"] += 1
                        summary["recipe_cells_with_all_seeds_both_ge99"] += int(all(
                            min(r["best"][s]["accuracy"] for s in ["val_within", "val_outside"]) >= .99 for r in rr))
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (DATA / "recipe-statistics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader(); writer.writerows(cells)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.7), sharey=True)
    for ax, split, title in zip(axes, ["val_within", "val_outside"],
                                ["Within-range validation", "Outside-range validation"]):
        for i, architecture in enumerate(NAMES):
            rr = sorted([r for r in rows if r["case"]["encoding"] == "bits"
                         and r["case"]["variant"]["name"] == architecture],
                        key=lambda r: (r["case"]["lr"], r["case"]["weight_decay"], r["case"]["seed"]))
            assert len(rr) == 18
            for j, r in enumerate(rr):
                seed = SEEDS.index(r["case"]["seed"])
                ax.scatter(100 * r["best"][split]["accuracy"], i + (j - 8.5) * .037,
                           color=COLORS[seed], marker=["o", "s", "^"][seed],
                           s=10, linewidths=0, alpha=.85)
        ax.axvline(100 / 7, color="#777", ls=":", lw=.8)
        ax.set_xlim(0, 104); ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis="x")
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Accuracy at within-selected checkpoint (%)")
        ax.set_yticks(range(16), list(NAMES.values()), fontsize=8)
    axes[0].invert_yaxis()
    handles = [axes[0].scatter([], [], color=color, marker=marker, s=18, label=f"Seed {seed}")
               for seed, color, marker in zip(SEEDS, COLORS, ["o", "s", "^"])]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .055, 1, 1)); save(fig, "million-binary-overview")

    groups = defaultdict(list)
    with gzip.open(DATA / "illustrative-curves.jsonl.gz", "rt") as stream:
        for line in stream:
            value = json.loads(line); groups[value["case_id"]].append(value)
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 6.4), sharex=True)
    for i, (architecture, lr, wd) in enumerate(snapshot["curve_recipes"]):
        for seed, color in zip(SEEDS, COLORS):
            r = next(r for r in rows if r["case"]["encoding"] == "bits"
                     and r["case"]["variant"]["name"] == architecture
                     and r["case"]["lr"] == lr and r["case"]["weight_decay"] == wd
                     and r["case"]["seed"] == seed)
            values = groups[r["case_id"]]
            x = np.array([v["samples"] / 1e6 for v in values])
            for j, extract in enumerate([lambda v: v["train_loss"],
                    lambda v: v["val_within"]["loss"], lambda v: 100*v["val_outside"]["accuracy"]]):
                axes[i,j].plot(x, [extract(v) for v in values], color=color, lw=.65, alpha=.85, label=f"Seed {seed}")
        for j in [0, 1]:
            axes[i,j].set_yscale("log"); axes[i,j].set_ylim(1e-6, 5)
            axes[i,j].set_yticks([1e-4, 1e-2, 1])
            axes[i,j].axhline(np.log(7), color="#777", ls=":", lw=.7)
        axes[i,2].set_ylim(0, 105); axes[i,2].set_yticks([0, 50, 100])
        axes[i,2].axhline(100/7, color="#777", ls=":", lw=.7)
        axes[i,0].set_ylabel(NAMES[architecture] + f"\nLR {lr:g}, decay {wd:g}", fontsize=8)
        for ax in axes[i]:
            ax.grid(axis="y"); ax.set_xticks([0, 5, 10]); ax.tick_params(labelsize=8)
            if i == 3: ax.set_xlabel("Presentations (M)", fontsize=8)
    for ax, title in zip(axes[0], ["Training-window CE", "Within-validation CE", "Outside-validation accuracy (%)"]):
        ax.set_title(title, fontsize=8.2, fontweight="bold")
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .035, 1, 1)); save(fig, "million-draft-learning-curves")

    lines = []
    for lr in [.0003, .003]:
        for wd in [0., .01, .1]:
            rr = sorted([r for r in rows if r["case"]["encoding"] == "bits"
                         and r["case"]["variant"]["name"] == "hyper_g2_f3"
                         and r["case"]["lr"] == lr and r["case"]["weight_decay"] == wd], key=lambda r: r["case"]["seed"])
            values = [f"{100*r['best'][split]['accuracy']:.2f}" for split in ["val_within", "val_outside"] for r in rr]
            lines.append(f"{lr:g} & {wd:g} & " + " & ".join(values) + r" \\")
    (TABLE / "million-hyper-settings.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    print(json.dumps(summary["encodings"], indent=2))


if __name__ == "__main__":
    style(); build()
