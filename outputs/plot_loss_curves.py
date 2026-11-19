"""Learning and validation loss curves of a UNET.py run, one panel per fold.

Usage:
    python plot_loss_curves.py RUN_DIR [--log train.log] [--out DIR] [--compare OTHER_RUN_DIR --compare-log other.log]

Reads RUN_DIR/fold_*/history.csv (written by UNET.py since 2026-11-19). For older runs, which only
printed the losses, pass the training log with --log ("Fold k | epoch n | train MSE x | val MSE y" lines).
Writes 13_loss_curves.png (+ 14_loss_curves_compare.png with --compare) and loss_history.csv
to ./plots/<run_name>_analysis/ by default.
Losses are MSE in normalized units (target standardized with the fold's training mean and std).
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_TRAIN, C_VAL, C_BEST = "#2a78d6", "#eb6834", "#3d3d3a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False,
})
LINE = re.compile(r"Fold\s+(\d+)\s*\|\s*epoch\s+(\d+)\s*\|\s*train MSE\s+([\d.eE+-]+)\s*\|\s*val MSE\s+([\d.eE+-]+)")


def from_histories(run_dir):
    frames = []
    for f in sorted(glob.glob(f"{run_dir}/fold_*/history.csv")):
        d = pd.read_csv(f)
        d.insert(0, "fold", int(re.search(r"fold_(\d+)", f).group(1)))
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def from_log(log):
    rows = [(int(a), int(b), float(c), float(d)) for a, b, c, d in LINE.findall(open(log).read())]
    if not rows:
        raise SystemExit(f"no epoch lines found in {log}")
    d = pd.DataFrame(rows, columns=["fold", "epoch", "train_mse", "val_mse"])
    return d.drop_duplicates(["fold", "epoch"], keep="last")      # a restarted fold keeps its last attempt


def load(run_dir, log):
    d = from_histories(run_dir) if run_dir else None
    if d is None:
        if not log:
            raise SystemExit(f"no fold_*/history.csv in {run_dir}: pass the training log with --log")
        d = from_log(log)
    return d.sort_values(["fold", "epoch"]).reset_index(drop=True)


def best_epochs(d):
    return d.loc[d.groupby("fold")["val_mse"].idxmin(), ["fold", "epoch", "train_mse", "val_mse"]]


def plot_folds(d, title, path):
    folds = sorted(d.fold.unique())
    ncol = 5
    nrow = int(np.ceil(len(folds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow + 0.6), sharey=True, squeeze=False)
    hi = np.nanpercentile(np.r_[d.train_mse, d.val_mse], 99) * 1.15
    lo = min(d.train_mse.min(), d.val_mse.min()) * 0.85
    best = best_epochs(d).set_index("fold")
    for ax, k in zip(axes.flat, folds):
        f = d[d.fold == k]
        ax.plot(f.epoch, f.train_mse, color=C_TRAIN, label="train")
        ax.plot(f.epoch, f.val_mse, color=C_VAL, label="validation")
        b = best.loc[k]
        ax.axvline(b.epoch, color=C_BEST, lw=0.9, ls="--")
        ax.plot(b.epoch, b.val_mse, "o", color=C_BEST, ms=4)
        ax.set_title(f"fold {k}  ·  best epoch {int(b.epoch)} / {int(f.epoch.max())}")
        ax.text(0.97, 0.93, f"val {b.val_mse:.4f}", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (normalized)")
    for ax in list(axes.flat)[len(folds):]:
        ax.axis("off")
    axes.flat[0].legend(loc="upper right", bbox_to_anchor=(1, 0.85))
    fig.suptitle(title + "   (dashed: epoch kept as best.pt = lowest validation MSE)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_compare(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colours = [C_TRAIN, C_VAL, "#1baf7a", "#e34948"]
    for (name, d), c in zip(runs, colours):
        for k, f in d.groupby("fold"):
            axes[0].plot(f.epoch, f.train_mse, color=c, lw=0.8, alpha=0.45)
            axes[1].plot(f.epoch, f.val_mse, color=c, lw=0.8, alpha=0.45)
        for ax, col in zip(axes, ["train_mse", "val_mse"]):
            m = d.groupby("epoch")[col].median()
            ax.plot(m.index, m.values, color=c, lw=2.2, label=f"{name} (median of folds)")
    for ax, t in zip(axes, ["train MSE", "validation MSE"]):
        ax.set_title(t)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("MSE (normalized)")
    axes[1].legend()
    fig.suptitle("Learning curves compared (thin: each fold)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--log", default=None, help="training log, used when the folds have no history.csv")
    ap.add_argument("--out", default=None, help="output dir (default: ./plots/<run_name>_analysis)")
    ap.add_argument("--compare", default=None, help="another run dir to overlay")
    ap.add_argument("--compare-log", default=None)
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.run_dir))
    out = a.out or f"plots/{name}_analysis"
    os.makedirs(out, exist_ok=True)

    d = load(a.run_dir, a.log)
    d.to_csv(f"{out}/loss_history.csv", index=False)
    plot_folds(d, f"{name}: learning and validation loss", f"{out}/13_loss_curves.png")
    print("-> 13_loss_curves")

    b = best_epochs(d).merge(d.groupby("fold").epoch.max().rename("last_epoch"), on="fold")
    b["gap_val_minus_train"] = b.val_mse - b.train_mse
    print(b.round(4).to_string(index=False))

    if a.compare:
        cname = os.path.basename(os.path.normpath(a.compare))
        plot_compare([(name, d), (cname, load(a.compare, a.compare_log))], f"{out}/14_loss_curves_compare.png")
        print("-> 14_loss_curves_compare")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
