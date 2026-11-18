"""
predictor_analysis.py - descriptive statistics and predictor selection for UNET.py
==================================================================================

Purpose
-------
Before training the U-Net, find out which variables are worth feeding it as predictors
for a given target. Works for ANY target and ANY set of candidate predictors found in
ALADIN_CORRIGE: the data are read exactly the way UNET.py reads them (same loader, same
GCM-grid upscaling, same cache), so the features analysed are the 1D vector the model gets
(one value per variable and per GCM cell, e.g. "tasminAdjust@-20.71N,56.25E").

Only non-test days are used (the fixed test periods of UNET.py are left out), so the choice
of predictors does not leak information from the test set.

Analyses (one numbered folder each in --out)
--------------------------------------------
  1_descriptive   per-feature statistics (mean, std, quantiles, skewness, kurtosis, missing),
                  historical vs projection shift, linear trend per decade, histograms,
                  seasonal cycles and annual means.
  2_relevance     link with the target: Pearson / Spearman on raw values and on deseasonalised
                  anomalies, mutual information (non-linear), lagged correlations (lag 0..3 days),
                  maps of the correlation between each variable and the target at every 3 km pixel.
  3_redundancy    correlation matrix between predictors (Spearman), hierarchical clustering,
                  variance inflation factors (VIF), list of strongly correlated pairs.
  4_pca           ACP / PCA of the standardised predictors: scree plot, loadings, correlation
                  circle with the target as supplementary variable, PC scores vs target;
                  EOF (PCA) of the 3 km target field and correlation of predictors with its modes.
  5_models        model-based selection with blocked cross-validation on years:
                    - ridge regression of the whole 3 km field (single variable, drop one variable,
                      greedy forward selection),
                    - gradient boosting on the domain-mean target (forward selection,
                      grouped permutation importance),
                    - LassoCV coefficients.
  summary.md      one table per variable with every score, a combined rank and a recommendation.

A "variable" (group) = all the GCM-cell columns of one predictor. "season" (sin/cos of the day of
the year) is added as a baseline group: it shows what a predictor adds beyond the seasonal cycle.

Usage
-----
  python predictor_analysis.py                                  # target tasmaxAdjust, every other variable as candidate
  python predictor_analysis.py --target tasminAdjust
  python predictor_analysis.py --target prtotAdjust --predictors tasminAdjust tasmaxAdjust rsdsAdjust hursminAdjust
  python predictor_analysis.py --cache-dir ~/unet_cache         # reuse / fill UNET.py's cache of prepared arrays
  python predictor_analysis.py --npz ~/unet_cache/data_xxx.npz  # read a UNET.py cache file directly (no NAS)
  python predictor_analysis.py --fast                           # skip the gradient-boosting part (slowest)

Paths follow UNET.py (flag > environment variable > default): --data-dir / UNET_DATA_DIR, --cache-dir /
UNET_CACHE_DIR, ... Extra dependencies compared with UNET.py: scikit-learn, matplotlib.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster import hierarchy
from scipy.ndimage import uniform_filter1d

import UNET

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_MAIN, C_ALT = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.4, "legend.frameon": False,
})
SEASON = "season"


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def available_variables(aladin_dir, scenarios):
    # variables that have a folder in every scenario: the candidates when --predictors is not given
    sets = [{p.name for p in (Path(aladin_dir) / s / "day").iterdir() if p.is_dir()} for s in scenarios]
    return sorted(set.intersection(*sets))


def load(args):
    if args.npz:
        d = np.load(Path(args.npz).expanduser(), allow_pickle = False)
        print(f"Loaded {args.npz}")
        return d["features"], d["Y"], d["times"], d["lat"], d["lon"], d["names"].tolist(), d["dem"]

    data_dir = Path(args.data_dir).expanduser()
    aladin_dir = Path(args.aladin_dir).expanduser() if args.aladin_dir else data_dir / UNET.ALADIN_SUBDIR
    predictors = args.predictors or available_variables(aladin_dir, args.scenarios)
    predictors = [v for v in predictors if v != args.target]          # the target is never its own predictor
    print(f"Target {args.target} | candidate predictors: {', '.join(predictors)}")
    ns = argparse.Namespace(data_dir = args.data_dir, aladin_dir = args.aladin_dir, dem_path = args.dem_path,
                            land_frac_path = getattr(args, "land_frac_path", None),
                            gcm_grid_file = args.gcm_grid_file, models_dir = args.out, cache_dir = args.cache_dir,
                            scenarios = args.scenarios, predictors = predictors, target = args.target)
    ns = UNET.resolve_paths(ns)
    return UNET.load_data_cached(ns)


def day_of_year(times):
    return pd.DatetimeIndex(times).dayofyear.values


def anomalies(A, doy, ref, window = 31):
    # removes a smoothed day-of-year climatology computed on the reference (non-test) days
    clim = np.full((366,) + A.shape[1:], np.nan)
    for d in range(1, 367):
        sel = ref & (doy == d)
        if sel.any():
            clim[d - 1] = np.nanmean(A[sel], axis = 0)
    clim = pd.DataFrame(clim.reshape(366, -1)).interpolate(limit_direction = "both").values.reshape(clim.shape)
    clim = uniform_filter1d(clim, size = window, axis = 0, mode = "wrap")
    return A - clim[doy - 1]


def standardize(A):
    m, s = np.nanmean(A, axis = 0), np.nanstd(A, axis = 0) + 1e-12
    return np.nan_to_num((A - m) / s)


def corr_columns(A, b):
    # Pearson correlation of each column of A (T, F) with b (T,) or with each column of B (T, P) -> (F,) / (F, P)
    Za = standardize(A)
    Zb = standardize(b if b.ndim > 1 else b[:, None])
    r = Za.T @ Zb / len(Za)
    return r[:, 0] if b.ndim == 1 else r


# ---------------------------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------------------------

def group_colors(groups):
    cmap = plt.get_cmap("tab10")
    return {g: cmap(i % 10) for i, g in enumerate(groups)}


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def show_map(ax, field, lat, lon, title, cmap = "RdBu_r", vmin = -1, vmax = 1):
    im = ax.pcolormesh(lon, lat, field, cmap = cmap, vmin = vmin, vmax = vmax, shading = "auto")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.tick_params(labelsize = 7)
    return im


def md_table(df, floatfmt = "{:.3f}"):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# 1. Descriptive statistics
# ---------------------------------------------------------------------------------------------

def descriptive(ctx, out):
    X, names, ref, years = ctx["X"], ctx["names"], ctx["ref"], ctx["years"]
    rows = []
    hist = ref & (years < ctx["projection_start"])
    proj = ref & (years >= ctx["projection_start"])
    cols = list(zip(names, X.T)) + [(f"TARGET {ctx['target']} (land mean)", ctx["y"])]
    for name, x in cols:
        v = x[ref]
        f = v[np.isfinite(v)]
        annual = pd.Series(x[ref]).groupby(years[ref]).mean()
        slope = stats.linregress(annual.index, annual.values).slope * 10 if len(annual) > 2 else np.nan
        sd = np.nanstd(v)
        rows.append({"feature": name, "n": len(f), "missing_%": 100 * (1 - len(f) / len(v)),
                     "mean": f.mean(), "std": sd, "cv": sd / abs(f.mean()) if f.mean() else np.nan,
                     "min": f.min(), "q05": np.quantile(f, .05), "q25": np.quantile(f, .25),
                     "median": np.median(f), "q75": np.quantile(f, .75), "q95": np.quantile(f, .95), "max": f.max(),
                     "skewness": stats.skew(f), "kurtosis": stats.kurtosis(f),
                     "skew_log1p": stats.skew(np.log1p(f)) if f.min() >= 0 else np.nan,
                     "zero_%": 100 * np.mean(f == 0),
                     "shift_proj-hist_in_std": (np.nanmean(x[proj]) - np.nanmean(x[hist])) / sd if hist.any() and proj.any() else np.nan,
                     "trend_per_decade": slope})
    table = pd.DataFrame(rows).set_index("feature")
    table.to_csv(out / "descriptive_stats.csv", float_format = "%.5g")

    # per variable (mean over the GCM cells) + target: histogram, seasonal cycle, annual mean
    series = {g: np.nanmean(X[:, ctx["group_cols"][g]], axis = 1) for g in ctx["var_groups"]}
    series[f"TARGET {ctx['target']}"] = ctx["y"]
    n = len(series)
    month = pd.DatetimeIndex(ctx["times"]).month.values
    fig, axes = plt.subplots(n, 3, figsize = (11, 2.1 * n))
    axes = np.atleast_2d(axes)
    for i, (g, x) in enumerate(series.items()):
        v = x[ref]
        color = C_ALT if g.startswith("TARGET") else C_MAIN
        skewed = abs(stats.skew(v[np.isfinite(v)])) > 2
        axes[i, 0].hist(v[np.isfinite(v)], bins = 60, color = color, log = skewed)
        axes[i, 0].set_title(f"{g}: distribution" + (" (log count)" if skewed else ""))
        clim = pd.Series(v).groupby(month[ref]).agg(["mean", lambda s: s.quantile(.1), lambda s: s.quantile(.9)])
        axes[i, 1].fill_between(clim.index, clim.iloc[:, 1], clim.iloc[:, 2], color = color, alpha = .25, lw = 0)
        axes[i, 1].plot(clim.index, clim.iloc[:, 0], color = color)
        axes[i, 1].set_xticks(range(1, 13))
        axes[i, 1].set_title("seasonal cycle (mean, q10-q90)")
        annual = pd.Series(v).groupby(years[ref]).mean()
        axes[i, 2].plot(annual.index, annual.values, ".", color = color, ms = 3)
        axes[i, 2].set_title("annual mean (test years left out)")
    fig.tight_layout()
    save(fig, out / "distributions_seasonal_annual.png")
    return table


# ---------------------------------------------------------------------------------------------
# 2. Relevance: link between each predictor and the target
# ---------------------------------------------------------------------------------------------

def relevance(ctx, out, max_samples, seed):
    from sklearn.feature_selection import mutual_info_regression

    X, Xa, y, ya, ref, names = ctx["X"], ctx["Xa"], ctx["y"], ctx["ya"], ctx["ref"], ctx["names"]
    Xr, Xar, yr, yar = X[ref], Xa[ref], y[ref], ya[ref]
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(yr), size = min(max_samples, len(yr)), replace = False)

    table = pd.DataFrame(index = pd.Index(names, name = "feature"))
    table["group"] = [ctx["group_of"][n] for n in names]
    table["pearson_raw"] = corr_columns(Xr, yr)
    table["spearman_raw"] = [stats.spearmanr(Xr[:, j], yr, nan_policy = "omit")[0] for j in range(len(names))]
    table["pearson_anom"] = corr_columns(Xar, yar)
    table["spearman_anom"] = [stats.spearmanr(Xar[:, j], yar, nan_policy = "omit")[0] for j in range(len(names))]
    table["mutual_info_raw"] = mutual_info_regression(np.nan_to_num(Xr[sub]), yr[sub], random_state = seed)
    table["mutual_info_anom"] = mutual_info_regression(np.nan_to_num(Xar[sub]), yar[sub], random_state = seed)

    # lagged correlation of anomalies: predictor at day t - lag vs target at day t (consecutive days only)
    times = ctx["times"].astype("datetime64[D]")
    for lag in range(1, 4):
        ok = np.zeros(len(y), dtype = bool)
        ok[lag:] = (times[lag:] - times[:-lag] == np.timedelta64(lag, "D")) & ref[lag:] & ref[:-lag]
        idx = np.where(ok)[0]
        table[f"pearson_anom_lag{lag}"] = corr_columns(Xa[idx - lag], ya[idx])
    table.to_csv(out / "relevance_per_feature.csv", float_format = "%.4f")

    fig, axes = plt.subplots(1, 3, figsize = (13, max(3, .22 * len(names) + 1)), sharey = True)
    order = table["pearson_anom"].abs().sort_values().index
    colors = [ctx["colors"][table.loc[n, "group"]] for n in order]
    for ax, col, title in zip(axes, ["pearson_raw", "pearson_anom", "mutual_info_anom"],
                              ["Pearson r (raw values)", "Pearson r (deseasonalised anomalies)", "Mutual information (anomalies)"]):
        ax.barh(range(len(order)), table.loc[order, col], color = colors)
        ax.axvline(0, color = INK2, lw = .8)
        ax.set_title(title)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize = 7)
    fig.tight_layout()
    save(fig, out / "relevance_bars.png")

    # 3 km maps: correlation of each variable (mean over GCM cells, anomalies) with the target at every land pixel
    land, lat, lon = ctx["land_mask"], ctx["lat"], ctx["lon"]
    Ya = ctx["Ya_land"][ref]
    groups = ctx["var_groups"]
    pix = {}
    ncol = min(4, len(groups))
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize = (3.3 * ncol, 3 * nrow), squeeze = False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, g in zip(axes.ravel(), groups):
        xg = np.nanmean(Xa[ref][:, ctx["group_cols"][g]], axis = 1)
        r = corr_columns(Ya, xg[:, None])[:, 0]            # (P,)
        pix[g] = r
        field = np.full(land.shape, np.nan)
        field[land] = r
        ax.set_visible(True)
        im = show_map(ax, field, lat, lon, f"{g}\nmean |r| = {np.mean(np.abs(r)):.2f}")
    fig.colorbar(im, ax = axes.ravel().tolist(), shrink = .7, label = "r with target anomaly (3 km)")
    save(fig, out / "pixel_correlation_maps.png")

    per_var = pd.DataFrame({
        "max_abs_pearson_raw": table.groupby("group")["pearson_raw"].apply(lambda s: s.abs().max()),
        "max_abs_pearson_anom": table.groupby("group")["pearson_anom"].apply(lambda s: s.abs().max()),
        "max_abs_spearman_anom": table.groupby("group")["spearman_anom"].apply(lambda s: s.abs().max()),
        "max_mutual_info_anom": table.groupby("group")["mutual_info_anom"].max(),
        "best_lag_abs_r": table.groupby("group")[[f"pearson_anom_lag{l}" for l in range(1, 4)]].apply(lambda d: d.abs().max().max()),
        "pixel_mean_abs_r": pd.Series({g: np.mean(np.abs(r)) for g, r in pix.items()}),
        "pixel_share_abs_r>0.3": pd.Series({g: np.mean(np.abs(r) > .3) for g, r in pix.items()}),
    })
    return table, per_var


# ---------------------------------------------------------------------------------------------
# 3. Redundancy between predictors
# ---------------------------------------------------------------------------------------------

def vif(A):
    R = np.corrcoef(standardize(A), rowvar = False)
    return np.diag(np.linalg.pinv(R))


def redundancy(ctx, out):
    X, ref, names = ctx["X"][ctx["ref"]], ctx["ref"], ctx["names"]
    ranks = np.apply_along_axis(stats.rankdata, 0, np.nan_to_num(X))
    R = np.corrcoef(ranks, rowvar = False)                       # Spearman
    pd.DataFrame(R, index = names, columns = names).to_csv(out / "spearman_matrix_features.csv", float_format = "%.3f")

    dist = np.clip(1 - np.abs(R), 0, None)
    np.fill_diagonal(dist, 0)
    link = hierarchy.linkage(dist[np.triu_indices(len(names), 1)], method = "average")
    order = hierarchy.leaves_list(link)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (14, max(4, .25 * len(names) + 1)),
                                   gridspec_kw = {"width_ratios": [1, 2.4]})
    hierarchy.dendrogram(link, labels = names, orientation = "left", ax = ax0, color_threshold = .3,
                         above_threshold_color = INK2, leaf_font_size = 7)
    ax0.set_title("clustering, distance = 1 - |Spearman r|")
    ax0.grid(False)
    im = ax1.imshow(R[np.ix_(order, order)], cmap = "RdBu_r", vmin = -1, vmax = 1)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(np.array(names)[order], rotation = 90, fontsize = 6)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(np.array(names)[order], fontsize = 6)
    ax1.grid(False)
    ax1.set_title("Spearman correlation between predictors")
    fig.colorbar(im, ax = ax1, shrink = .7)
    fig.tight_layout()
    save(fig, out / "predictor_correlation_clustered.png")

    # per variable (mean over GCM cells): correlation, VIF
    groups = ctx["var_groups"]
    G = np.column_stack([np.nanmean(X[:, ctx["group_cols"][g]], axis = 1) for g in groups])
    Rg = pd.DataFrame(stats.spearmanr(G)[0] if len(groups) > 2 else np.corrcoef(G, rowvar = False),
                      index = groups, columns = groups)
    Rg.to_csv(out / "spearman_matrix_variables.csv", float_format = "%.3f")
    fig, ax = plt.subplots(figsize = (1 + .7 * len(groups), .8 + .6 * len(groups)))
    ax.imshow(Rg.values, cmap = "RdBu_r", vmin = -1, vmax = 1)
    for i in range(len(groups)):
        for j in range(len(groups)):
            ax.text(j, i, f"{Rg.values[i, j]:.2f}", ha = "center", va = "center", fontsize = 7,
                    color = "white" if abs(Rg.values[i, j]) > .6 else INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation = 45, ha = "right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.grid(False)
    ax.set_title("Spearman r between variables (mean over GCM cells)")
    save(fig, out / "variable_correlation.png")

    pairs = [(names[i], names[j], R[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))
             if abs(R[i, j]) > .9]
    pd.DataFrame(pairs, columns = ["feature_1", "feature_2", "spearman_r"]).sort_values(
        "spearman_r", key = np.abs, ascending = False).to_csv(out / "pairs_abs_r_above_0.9.csv", index = False, float_format = "%.3f")
    pd.DataFrame({"vif": vif(X)}, index = pd.Index(names, name = "feature")).to_csv(out / "vif_features.csv", float_format = "%.2f")

    offdiag = Rg.abs().where(~np.eye(len(groups), dtype = bool))
    per_var = pd.DataFrame({"vif_variable_means": vif(G) if len(groups) > 1 else [1.0],
                            "max_abs_r_other_variable": offdiag.max(axis = 1).values,
                            "most_correlated_with": offdiag.idxmax(axis = 1).values if len(groups) > 1 else ["-"],
                            "within_variable_mean_abs_r": [np.mean(np.abs(R[np.ix_(ctx["group_cols"][g], ctx["group_cols"][g])]))
                                                           for g in groups]}, index = groups)
    return per_var


# ---------------------------------------------------------------------------------------------
# 4. PCA / ACP of the predictors, EOF of the target field
# ---------------------------------------------------------------------------------------------

def pca_analysis(ctx, out):
    from sklearn.decomposition import PCA

    ref, names = ctx["ref"], ctx["names"]
    Z = standardize(ctx["X"][ref])
    pca = PCA().fit(Z)
    scores = pca.transform(Z)
    ev = pca.explained_variance_ratio_
    k90, k95 = int(np.searchsorted(np.cumsum(ev), .90) + 1), int(np.searchsorted(np.cumsum(ev), .95) + 1)
    loadings_r = pca.components_.T * np.sqrt(pca.explained_variance_)       # corr(feature, PC) for standardised data
    npc = min(10, len(ev))
    pc_cols = [f"PC{i + 1}" for i in range(npc)]
    pd.DataFrame(loadings_r[:, :npc], index = pd.Index(names, name = "feature"), columns = pc_cols).to_csv(
        out / "pca_loadings_correlation.csv", float_format = "%.3f")
    r_y = corr_columns(scores[:, :npc], ctx["y"][ref])
    r_ya = corr_columns(scores[:, :npc], ctx["ya"][ref])
    pd.DataFrame({"explained_variance_%": 100 * ev[:npc], "cumulative_%": 100 * np.cumsum(ev)[:npc],
                  "r_with_target": r_y, "r_with_target_anomaly": r_ya}, index = pc_cols).to_csv(
        out / "pca_components.csv", float_format = "%.3f")

    fig, axes = plt.subplots(1, 3, figsize = (15, 4.6))
    ax = axes[0]
    ax.bar(range(1, npc + 1), 100 * ev[:npc], color = C_MAIN)
    ax.plot(range(1, npc + 1), 100 * np.cumsum(ev)[:npc], "o-", color = C_ALT, ms = 3, label = "cumulative")
    ax.axhline(90, color = INK2, lw = .8, ls = "--")
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance (%)")
    ax.set_title(f"Scree: {k90} PCs for 90 %, {k95} for 95 %")
    ax.legend()

    ax = axes[1]                                                             # correlation circle, PC1-PC2
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color = INK2, lw = .8)
    for j, n in enumerate(names):
        c = ctx["colors"][ctx["group_of"][n]]
        ax.arrow(0, 0, loadings_r[j, 0], loadings_r[j, 1] if npc > 1 else 0, color = c, alpha = .8,
                 head_width = .02, length_includes_head = True)
    for g in ctx["var_groups"] + ([SEASON] if ctx["season"] is not None else []):
        if g in ctx["var_groups"]:
            ax.plot([], [], color = ctx["colors"][g], label = g)
    ty = [r_y[0], r_y[1] if npc > 1 else 0]
    ax.arrow(0, 0, *ty, color = INK, lw = 2, head_width = .04, length_includes_head = True)
    ax.text(ty[0] * 1.08, ty[1] * 1.08, "target\n(suppl.)", fontsize = 8, ha = "center")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f} %)" if npc > 1 else "")
    ax.set_title("Correlation circle (target = supplementary variable)")
    ax.legend(fontsize = 7, loc = "lower left")

    ax = axes[2]                                                             # individuals coloured by the target
    sub = np.random.default_rng(0).choice(len(scores), size = min(5000, len(scores)), replace = False)
    sc = ax.scatter(scores[sub, 0], scores[sub, 1] if npc > 1 else 0, c = ctx["y"][ref][sub], s = 3, cmap = "viridis")
    fig.colorbar(sc, ax = ax, label = ctx["target"])
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("Days in the PC1-PC2 plane, coloured by target")
    fig.tight_layout()
    save(fig, out / "pca_scree_circle_scores.png")

    fig, ax = plt.subplots(figsize = (1.2 + .55 * npc, max(3, .22 * len(names) + 1)))
    ax.imshow(loadings_r[:, :npc], cmap = "RdBu_r", vmin = -1, vmax = 1, aspect = "auto")
    ax.set_xticks(range(npc))
    ax.set_xticklabels([f"{c}\n{100 * e:.0f}%\nr_y={r:.2f}" for c, e, r in zip(pc_cols, ev, r_y)], fontsize = 7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize = 7)
    ax.grid(False)
    ax.set_title("corr(feature, PC); r_y = corr(PC, target)")
    save(fig, out / "pca_loadings.png")

    # EOF of the 3 km target anomalies: how many spatial modes the predictors must explain
    Ya = np.nan_to_num(ctx["Ya_land"][ref])
    n_eof = min(10, Ya.shape[1])
    eof = PCA(n_components = n_eof).fit(Ya)
    epc = eof.transform(Ya)
    r_eof = corr_columns(ctx["Xa"][ref], epc[:, :min(3, n_eof)])           # (F, 3)
    eof_cols = [f"EOF{i + 1} ({100 * v:.0f}%)" for i, v in enumerate(eof.explained_variance_ratio_[:r_eof.shape[1]])]
    pd.DataFrame(r_eof, index = pd.Index(names, name = "feature"), columns = eof_cols).to_csv(
        out / "target_eof_vs_predictor_anomalies.csv", float_format = "%.3f")
    land = ctx["land_mask"]
    fig, axes = plt.subplots(1, r_eof.shape[1] + 1, figsize = (4 * (r_eof.shape[1] + 1), 3.6))
    axes[0].bar(range(1, n_eof + 1), 100 * eof.explained_variance_ratio_, color = C_ALT)
    axes[0].set_title("Target field EOFs: explained variance (%)")
    for i in range(r_eof.shape[1]):
        field = np.full(land.shape, np.nan)
        field[land] = eof.components_[i]
        m = np.nanmax(np.abs(field))
        show_map(axes[i + 1], field, ctx["lat"], ctx["lon"], eof_cols[i], vmin = -m, vmax = m)
    fig.tight_layout()
    save(fig, out / "target_eofs.png")

    per_var = pd.DataFrame({
        "max_abs_r_target_EOF1": pd.Series(np.abs(r_eof[:, 0]), index = names).groupby(ctx["group_of"]).max(),
        "max_abs_loading_PC_best_for_target": pd.Series(np.abs(loadings_r[:, np.argmax(np.abs(r_y))]),
                                                        index = names).groupby(ctx["group_of"]).max(),
    })
    return per_var, {"k90": k90, "k95": k95, "n_features": len(names),
                     "eof_var": eof.explained_variance_ratio_, "best_pc": int(np.argmax(np.abs(r_y))) + 1,
                     "best_pc_r": float(r_y[np.argmax(np.abs(r_y))])}


# ---------------------------------------------------------------------------------------------
# 5. Model-based selection (blocked CV on years)
# ---------------------------------------------------------------------------------------------

def year_folds(years, n_folds):
    groups = UNET.split_years(years, n_folds)
    return [(np.where(~np.isin(years, g))[0], np.where(np.isin(years, g))[0]) for g in groups]


def design(ctx, groups):
    cols = [ctx["Xfull"][:, ctx["full_cols"][g]] for g in groups]
    return np.concatenate(cols, axis = 1)


def cv_ridge_field(A, Y, folds, alpha = 1.0):
    # R2 of the whole 3 km field (all land pixels pooled), ridge on standardised inputs
    from sklearn.linear_model import Ridge
    sse = sst = 0.0
    for tr, va in folds:
        m, s = A[tr].mean(0), A[tr].std(0) + 1e-12
        model = Ridge(alpha = alpha).fit((A[tr] - m) / s, Y[tr])
        pred = model.predict((A[va] - m) / s)
        sse += np.sum((pred - Y[va]) ** 2)
        sst += np.sum((Y[va] - Y[tr].mean(0)) ** 2)
    return 1 - sse / sst


def hgb_model(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter = 200, learning_rate = .08, early_stopping = False, random_state = seed)


def cv_hgb(A, y, folds, seed):
    sse = sst = 0.0
    for tr, va in folds:
        pred = hgb_model(seed).fit(A[tr], y[tr]).predict(A[va])
        sse += np.sum((pred - y[va]) ** 2)
        sst += np.sum((y[va] - y[tr].mean()) ** 2)
    return 1 - sse / sst


def forward_selection(candidates, score_fn):
    chosen, path, rest = [], [], list(candidates)
    while rest:
        scores = {g: score_fn(chosen + [g]) for g in rest}
        best = max(scores, key = scores.get)
        chosen.append(best)
        rest.remove(best)
        path.append((best, scores[best]))
        print(f"    + {best:<20s} R2 = {scores[best]:.4f}")
    return path


def models(ctx, out, n_folds, max_samples, seed, fast):
    from sklearn.linear_model import LassoCV

    ref = ctx["ref"]
    idx = np.where(ref)[0]
    years = ctx["years"][idx]
    folds = year_folds(years, n_folds)
    Y = np.nan_to_num(ctx["Y_land"][idx])
    y = ctx["y"][idx]
    var_groups = ctx["var_groups"]
    all_groups = var_groups + ([SEASON] if ctx["season"] is not None else [])
    A = lambda gs: design(ctx, gs)[idx]
    res = pd.DataFrame(index = pd.Index(all_groups, name = "variable"))

    print("  ridge, whole 3 km field ...")
    full = cv_ridge_field(A(all_groups), Y, folds)
    season_only = cv_ridge_field(A([SEASON]), Y, folds) if SEASON in all_groups else 0.0
    res["ridge_field_R2_alone"] = [cv_ridge_field(A([g]), Y, folds) for g in all_groups]
    res["ridge_field_R2_with_season"] = [cv_ridge_field(A([g, SEASON]), Y, folds) - season_only if g != SEASON and SEASON in all_groups
                                         else np.nan for g in all_groups]
    res["ridge_field_drop_one_loss"] = [full - cv_ridge_field(A([h for h in all_groups if h != g]), Y, folds)
                                        if len(all_groups) > 1 else full for g in all_groups]
    print("  ridge forward selection (field R2):")
    path_ridge = forward_selection(all_groups, lambda gs: cv_ridge_field(A(gs), Y, folds))
    res["ridge_forward_step"] = pd.Series({g: i + 1 for i, (g, _) in enumerate(path_ridge)})

    # LassoCV on the domain-mean target: which individual columns survive the L1 penalty
    Afull = A(all_groups)
    names_full = [n for g in all_groups for n in ctx["full_names"][g]]
    Zf = standardize(Afull)
    sub = np.random.default_rng(seed).choice(len(y), size = min(max_samples, len(y)), replace = False)
    split = [(np.intersect1d(tr, sub, assume_unique = True), np.intersect1d(va, sub, assume_unique = True)) for tr, va in folds]
    pos = {v: i for i, v in enumerate(sub)}
    split = [(np.array([pos[i] for i in tr]), np.array([pos[i] for i in va])) for tr, va in split]
    lasso = LassoCV(cv = split, alphas = 50, max_iter = 5000, random_state = seed).fit(Zf[sub], y[sub])
    coef = pd.Series(lasso.coef_, index = names_full)
    coef.to_csv(out / "lasso_coefficients.csv", header = ["coef_standardised"], float_format = "%.4f")
    group_of_full = {n: g for g in all_groups for n in ctx["full_names"][g]}
    res["lasso_sum_abs_coef"] = coef.abs().groupby(group_of_full).sum()
    res["lasso_n_nonzero"] = (coef != 0).groupby(group_of_full).sum()

    path_hgb = None
    if not fast:
        print("  gradient boosting forward selection (domain-mean target, subsample) ...")
        path_hgb = forward_selection(all_groups, lambda gs: cv_hgb(A(gs)[sub], y[sub], split, seed))
        res["hgb_forward_step"] = pd.Series({g: i + 1 for i, (g, _) in enumerate(path_hgb)})

        print("  grouped permutation importance ...")
        rng = np.random.default_rng(seed)
        imp = {g: [] for g in all_groups}
        Asub, ysub = Afull[sub], y[sub]
        cols_of = {}
        start = 0
        for g in all_groups:
            cols_of[g] = np.arange(start, start + len(ctx["full_names"][g]))
            start += len(ctx["full_names"][g])
        for tr, va in split:
            model = hgb_model(seed).fit(Asub[tr], ysub[tr])
            base = np.mean((model.predict(Asub[va]) - ysub[va]) ** 2)
            var = np.var(ysub[va])
            for g in all_groups:
                for _ in range(3):
                    Ap = Asub[va].copy()
                    Ap[:, cols_of[g]] = Ap[rng.permutation(len(va))][:, cols_of[g]]   # rows shuffled together: keeps within-group structure
                    imp[g].append((np.mean((model.predict(Ap) - ysub[va]) ** 2) - base) / var)
        res["hgb_permutation_R2_loss"] = pd.Series({g: np.mean(v) for g, v in imp.items()})
        res["hgb_permutation_R2_loss_std"] = pd.Series({g: np.std(v) for g, v in imp.items()})
    res.to_csv(out / "model_based_scores.csv", float_format = "%.4f")

    fig, axes = plt.subplots(1, 2 if path_hgb else 1, figsize = (12 if path_hgb else 6, 4), squeeze = False)
    for ax, path, title in zip(axes[0], [path_ridge, path_hgb], ["Ridge, whole 3 km field", "Gradient boosting, domain-mean target"]):
        if path is None:
            continue
        ax.plot(range(1, len(path) + 1), [s for _, s in path], "o-", color = C_MAIN)
        ax.set_xticks(range(1, len(path) + 1))
        ax.set_xticklabels([g for g, _ in path], rotation = 45, ha = "right")
        ax.set_ylabel("cross-validated R2")
        ax.set_title(f"Forward selection: {title}")
    fig.tight_layout()
    save(fig, out / "forward_selection.png")

    if not fast:
        fig, ax = plt.subplots(figsize = (6, .4 * len(all_groups) + 1))
        r = res["hgb_permutation_R2_loss"].sort_values()
        ax.barh(r.index, r.values, xerr = res.loc[r.index, "hgb_permutation_R2_loss_std"],
                color = [ctx["colors"].get(g, INK2) for g in r.index])
        ax.set_xlabel("R2 lost when the variable is shuffled")
        ax.set_title("Grouped permutation importance (gradient boosting)")
        save(fig, out / "permutation_importance.png")
    return res, {"ridge_full_R2": full, "ridge_season_only_R2": season_only, "path_ridge": path_ridge, "path_hgb": path_hgb}


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------

def summarize(ctx, out, rel, red, pca_v, mod, info_pca, info_mod, desc):
    var_groups = ctx["var_groups"]
    S = pd.concat([rel, red, pca_v, mod.reindex(var_groups)], axis = 1).loc[var_groups]
    S.index.name = "variable"
    relevance_cols = [c for c in ["max_abs_pearson_anom", "max_mutual_info_anom", "pixel_mean_abs_r",
                                  "ridge_field_R2_with_season", "ridge_field_drop_one_loss", "hgb_permutation_R2_loss"]
                      if c in S and S[c].notna().any()]
    S["combined_rank"] = S[relevance_cols].rank(ascending = False).mean(axis = 1)
    S = S.sort_values("combined_rank")

    def advice(row):
        drop = row.get("ridge_field_drop_one_loss", np.nan)
        perm = row.get("hgb_permutation_R2_loss", np.nan)
        unique = (np.nan_to_num(drop) > .005) or (np.nan_to_num(perm) > .01)
        if unique:
            return "keep (adds information no other variable carries)"
        if row["max_abs_r_other_variable"] > .8:
            return f"redundant with {row['most_correlated_with']} (keep only one?)"
        if row["max_abs_pearson_anom"] < .1 and row["max_mutual_info_anom"] < .02:
            return "weak link with the target: candidate to drop"
        return "small marginal gain: test with/without in UNET.py"
    S["recommendation"] = S.apply(advice, axis = 1)
    S.to_csv(out / "summary_per_variable.csv", float_format = "%.4f")

    skewed = desc[(desc["skewness"].abs() > 2) & desc.index.str.contains("@")]
    skewed_vars = sorted({n.split("@")[0] for n in skewed.index})
    shifted = desc[(desc["shift_proj-hist_in_std"].abs() > .5) & desc.index.str.contains("@")]
    shifted_vars = sorted({n.split("@")[0] for n in shifted.index})

    lines = [f"# Predictor analysis for target `{ctx['target']}`", "",
             f"- days analysed (test periods {', '.join(f'{a}-{b}' for a, b in ctx['test_periods'])} left out): "
             f"{int(ctx['ref'].sum())}",
             f"- 1D features: {len(ctx['names'])} ({len(var_groups)} variables x {len(ctx['names']) // max(1, len(var_groups))} GCM cells)",
             f"- PCA: {info_pca['k90']} components explain 90 % of the predictor variance, {info_pca['k95']} explain 95 % "
             f"(out of {info_pca['n_features']}); PC{info_pca['best_pc']} is the most correlated with the target "
             f"(r = {info_pca['best_pc_r']:.2f})",
             f"- target 3 km field: EOF1 = {100 * info_pca['eof_var'][0]:.0f} % of the anomaly variance, "
             f"EOF1-3 = {100 * info_pca['eof_var'][:3].sum():.0f} %",
             f"- ridge, whole field, cross-validated R2: season only {info_mod['ridge_season_only_R2']:.3f}, "
             f"all variables + season {info_mod['ridge_full_R2']:.3f}",
             f"- ridge forward order: {' > '.join(g for g, _ in info_mod['path_ridge'])}"]
    if info_mod["path_hgb"]:
        lines.append(f"- gradient boosting forward order: {' > '.join(g for g, _ in info_mod['path_hgb'])}")
    if skewed_vars:
        lines.append(f"- strongly skewed (|skewness| > 2), consider log1p before normalisation: {', '.join(skewed_vars)}")
    if shifted_vars:
        lines.append(f"- mean shift projection vs historical > 0.5 std (non-stationary, check the model extrapolates): "
                     f"{', '.join(shifted_vars)}")
    keep = S.index[S["recommendation"].str.startswith("keep")].tolist()
    lines += ["", f"**Suggested predictor set:** `--predictors {' '.join(keep) if keep else ' '.join(S.index[:3])}`", "",
              "## Scores per variable (sorted by combined rank)", "",
              md_table(S[["combined_rank"] + relevance_cols + ["max_abs_r_other_variable", "most_correlated_with",
                                                               "vif_variable_means", "recommendation"]]), "",
              "Column guide: `max_abs_pearson_anom` / `max_mutual_info_anom` = strongest link of any GCM cell of the variable "
              "with the target anomaly (linear / non-linear); `pixel_mean_abs_r` = mean |r| over the 3 km land pixels; "
              "`ridge_field_R2_with_season` = R2 gained over the seasonal cycle alone; `ridge_field_drop_one_loss` / "
              "`hgb_permutation_R2_loss` = R2 lost without the variable when all others are present (its unique information); "
              "`vif_variable_means` > 10 = strong multicollinearity.", "",
              "Figures and full tables: see the numbered folders next to this file."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:12]))
    return S


# ---------------------------------------------------------------------------------------------

def main(args):
    t0 = time.time()
    args.out = Path(args.out or f"predictor_analysis/{args.target}").expanduser()
    dirs = {k: args.out / k for k in ["1_descriptive", "2_relevance", "3_redundancy", "4_pca", "5_models"]}
    for d in dirs.values():
        d.mkdir(parents = True, exist_ok = True)

    features, Y, times, lat, lon, names, dem = load(args)
    names = [str(n) for n in names]
    if args.drop_features:
        keep = [i for i, n in enumerate(names) if n.split("@")[0] not in args.drop_features]
        features, names = features[:, keep], [names[i] for i in keep]
    land_mask = ~np.isnan(Y[0, 0])
    Y_land = Y[:, 0][:, land_mask].astype(np.float64)              # (T, P)
    y = np.nanmean(Y_land, axis = 1)                               # domain-mean target (T,)
    X = features.astype(np.float64)

    years = UNET.years_of(times)
    test_periods = UNET.parse_periods(args.test_periods)
    is_test = np.zeros(len(times), dtype = bool)
    for a, b in test_periods:
        is_test |= (years >= a) & (years <= b)
    ref = ~is_test if not args.include_test else np.ones(len(times), dtype = bool)
    ref &= np.isfinite(y) & np.all(np.isfinite(X), axis = 1)

    doy = day_of_year(times)
    group_of = {n: n.split("@")[0] for n in names}
    var_groups = list(dict.fromkeys(group_of.values()))
    group_cols = {g: [i for i, n in enumerate(names) if group_of[n] == g] for g in var_groups}

    season = np.column_stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]) if not args.no_season else None
    Xfull = X if season is None else np.concatenate([X, season], axis = 1)
    full_cols = dict(group_cols)
    full_names = {g: [names[i] for i in group_cols[g]] for g in var_groups}
    if season is not None:
        full_cols[SEASON] = [X.shape[1], X.shape[1] + 1]
        full_names[SEASON] = ["season_sin", "season_cos"]

    print("Deseasonalising ...")
    ctx = dict(target = args.target, X = X, Xa = anomalies(X, doy, ref), y = y, ya = anomalies(y[:, None], doy, ref)[:, 0],
               Y_land = Y_land, Ya_land = anomalies(Y_land, doy, ref), times = times, years = years, ref = ref,
               names = names, group_of = group_of, var_groups = var_groups, group_cols = group_cols,
               colors = group_colors(var_groups), land_mask = land_mask, lat = lat, lon = lon, season = season,
               Xfull = Xfull, full_cols = full_cols, full_names = full_names, test_periods = test_periods,
               projection_start = args.projection_start)
    print(f"{int(ref.sum())} days | {len(names)} features | {len(var_groups)} variables | {int(land_mask.sum())} land pixels")

    print("1. descriptive statistics")
    desc = descriptive(ctx, dirs["1_descriptive"])
    print("2. relevance")
    _, rel = relevance(ctx, dirs["2_relevance"], args.max_samples, args.seed)
    print("3. redundancy")
    red = redundancy(ctx, dirs["3_redundancy"])
    print("4. PCA / EOF")
    pca_v, info_pca = pca_analysis(ctx, dirs["4_pca"])
    print("5. model-based selection")
    mod, info_mod = models(ctx, dirs["5_models"], args.n_folds, args.max_samples, args.seed, args.fast)
    summarize(ctx, args.out, rel, red, pca_v, mod, info_pca, info_mod, desc)
    print(f"\nDone in {time.time() - t0:.0f} s -> {args.out}/summary.md")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering = True)
    env = os.environ.get
    p = argparse.ArgumentParser(description = "Descriptive statistics, PCA and predictor selection for UNET.py",
                                formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    paths = p.add_argument_group("paths (same as UNET.py)")
    paths.add_argument("--data-dir", default = env("UNET_DATA_DIR", UNET.DEFAULT_DATA_DIR))
    paths.add_argument("--aladin-dir", default = None)
    paths.add_argument("--dem-path", default = None)
    paths.add_argument("--land-frac-path", default = None)
    paths.add_argument("--gcm-grid-file", default = None)
    paths.add_argument("--cache-dir", default = env("UNET_CACHE_DIR", UNET.DEFAULT_CACHE_DIR),
                       help = "UNET.py cache of prepared arrays, '' = off")
    paths.add_argument("--npz", default = None, help = "read a UNET.py cache file directly instead of the NetCDF files")
    paths.add_argument("--out", default = None, help = "output folder (default predictor_analysis/<target>)")

    exp = p.add_argument_group("analysis")
    exp.add_argument("--target", default = UNET.TARGET)
    exp.add_argument("--predictors", nargs = "+", default = None,
                     help = "candidates (default: every variable present in all scenarios, target excluded)")
    exp.add_argument("--drop-features", nargs = "+", default = None, help = "variables to remove after loading (e.g. from an --npz)")
    exp.add_argument("--scenarios", nargs = "+", default = UNET.SCENARIOS)
    exp.add_argument("--test-periods", nargs = "+", default = UNET.TEST_PERIODS, help = "left out of every statistic")
    exp.add_argument("--include-test", action = "store_true", help = "use the test periods too (not recommended)")
    exp.add_argument("--projection-start", type = int, default = UNET.PROJECTION_START)
    exp.add_argument("--n-folds", type = int, default = 5, help = "blocked CV folds on years for the model-based scores")
    exp.add_argument("--max-samples", type = int, default = 20000, help = "subsample for mutual information, Lasso and boosting")
    exp.add_argument("--no-season", action = "store_true", help = "do not add the sin/cos day-of-year baseline")
    exp.add_argument("--fast", action = "store_true", help = "skip gradient boosting (forward selection + permutation)")
    exp.add_argument("--seed", type = int, default = 0)
    main(p.parse_args())
