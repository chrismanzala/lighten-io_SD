"""Evaluation of U-Net downscaling outputs (tasmaxAdjust) against ALADIN-corrected truth.

Usage:
    python analyze_run.py /mnt/nas/Data/christian_data/run1 [--scenario ssp370]

Writes PNG figures + CSV tables to ./plots/<run_name>_analysis/ (override with --out).
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

ALADIN = "/mnt/nas/Data/METEO_FRANCE/DONNEES_SIMULATIONS/ALADIN/ALADIN_CORRIGE"
DEM = "/mnt/nas/Data/DEM/gebco_2025_n-20.82_s-21.45_w55.15_e55.9.nc"
PERIODS = {"2011-2014": ("2011", "2014"), "2046-2050": ("2046", "2050"), "2096-2100": ("2096", "2100"),
           "All test": ("2011", "2100")}  # predictions only exist on test days -> all three periods pooled
SUB = ["2011-2014", "2046-2050", "2096-2100"]

# palette (validated reference instance): truth = slot 1 blue, prediction = slot 2 orange
C_TRUTH, C_PRED, C_FOLD = "#2a78d6", "#eb6834", "#b9b8b3"
C_PER = ["#2a78d6", "#1baf7a", "#e34948", "#3d3d3a"]  # one colour per period (slots 1,3,8) + neutral ink for pooled
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False,
})
K2C = 273.15


def load(run_dir, scenario):
    pred = xr.open_dataset(f"{run_dir}/tasmaxAdjust_test_pred_ensemble.nc")["tasmaxAdjust"] - K2C
    hist = glob.glob(f"{ALADIN}/historical/day/tasmaxAdjust/*.nc")[0]
    fut = glob.glob(f"{ALADIN}/{scenario}/day/tasmaxAdjust/*.nc")[0]
    truth = xr.concat([xr.open_dataset(hist)["tasmaxAdjust"], xr.open_dataset(fut)["tasmaxAdjust"]], "time")
    truth = truth.sel(time=pred.time).load() - K2C
    mask = np.load(f"{run_dir}/land_mask.npy")
    folds = sorted(glob.glob(f"{run_dir}/fold_*/tasmaxAdjust_test_pred.nc"))
    fold_preds = [xr.open_dataset(f)["tasmaxAdjust"].load() - K2C for f in folds]
    pred = pred.load()
    m = xr.DataArray(mask, dims=("lat", "lon"))
    return pred.where(m), truth.where(m), [f.where(m) for f in fold_preds], mask


def land(da, mask):
    """(time, n_land) array."""
    return da.values[:, mask]


def wd(a, b):
    a, b = np.ravel(a), np.ravel(b)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return stats.wasserstein_distance(a, b)


def basic_metrics(p, t):
    e = p - t
    e = e[np.isfinite(e)]
    tt, pp = t[np.isfinite(t)], p[np.isfinite(p)]
    return dict(rmse=np.sqrt(np.mean(e ** 2)), mae=np.mean(np.abs(e)), bias=e.mean(),
                r=np.corrcoef(pp, tt)[0, 1], wasserstein=wd(pp, tt),
                ks=stats.ks_2samp(pp, tt).statistic,
                std_ratio=pp.std() / tt.std(),
                p01_err=np.percentile(pp, 1) - np.percentile(tt, 1),
                p99_err=np.percentile(pp, 99) - np.percentile(tt, 99))


def save(fig, out, name):
    fig.savefig(f"{out}/{name}.png")
    plt.close(fig)
    print("  ->", name)


def map_panel(ax, da2d, cmap, vmin, vmax, title, lon, lat):
    im = ax.pcolormesh(lon, lat, da2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.tick_params(labelsize=7)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--scenario", default="ssp370")
    ap.add_argument("--out", default=None, help="output dir (default: ./plots/<run_name>_analysis)")
    args = ap.parse_args()
    out = args.out or os.path.join("plots", os.path.basename(os.path.normpath(args.run_dir)) + "_analysis")
    os.makedirs(out, exist_ok=True)

    print("loading...")
    pred, truth, fpreds, mask = load(args.run_dir, args.scenario)
    lon, lat = pred.lon.values, pred.lat.values
    sel = {k: dict(time=slice(*v)) for k, v in PERIODS.items()}

    # ------------------------------------------------------------------ metrics table
    rows = []
    for name in PERIODS:
        s = sel[name]
        t = land(truth.sel(**s), mask)
        rows.append(dict(model="ensemble", period=name, **basic_metrics(land(pred.sel(**s), mask), t)))
        for i, fp in enumerate(fpreds):
            rows.append(dict(model=f"fold_{i:02d}", period=name, **basic_metrics(land(fp.sel(**s), mask), t)))
    met = pd.DataFrame(rows)
    met.to_csv(f"{out}/metrics_full.csv", index=False)
    print(met[met.model == "ensemble"].round(3).to_string(index=False))

    # ------------------------------------------------------------------ 1. density curves
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, name in zip(axes, PERIODS):
        t = land(truth.sel(**sel[name]), mask).ravel()
        p = land(pred.sel(**sel[name]), mask).ravel()
        x = np.linspace(min(t.min(), p.min()) - 1, max(t.max(), p.max()) + 1, 400)
        for fp in fpreds:
            ax.plot(x, stats.gaussian_kde(land(fp.sel(**sel[name]), mask).ravel()[::7])(x),
                    color=C_FOLD, lw=0.7, alpha=0.7)
        ax.plot(x, stats.gaussian_kde(t[::3])(x), color=C_TRUTH, lw=2, label="ALADIN truth")
        ax.plot(x, stats.gaussian_kde(p[::3])(x), color=C_PRED, lw=2, label="U-Net ensemble")
        ax.plot([], [], color=C_FOLD, lw=0.7, label="individual folds")
        ax.set_title(f"{name}   W1 = {wd(p, t):.3f} °C")
        ax.set_xlabel("tasmax (°C)")
    axes[0].set_ylabel("density (all land pixels × days)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Density of daily tasmax: truth vs prediction", x=0.07, ha="left", fontsize=11)
    save(fig, out, "01_density_by_period")

    # ------------------------------------------------------------------ 2. QQ plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    q = np.linspace(0.001, 0.999, 300)
    for ax, name, c in zip(axes, PERIODS, C_PER):
        t = land(truth.sel(**sel[name]), mask).ravel()
        p = land(pred.sel(**sel[name]), mask).ravel()
        qt, qp = np.quantile(t, q), np.quantile(p, q)
        ax.plot([qt.min(), qt.max()], [qt.min(), qt.max()], color=INK2, lw=1, ls="--")
        ax.plot(qt, qp, color=C_PRED, lw=2)
        for pc in (0.01, 0.5, 0.99):
            ax.plot(np.quantile(t, pc), np.quantile(p, pc), "o", ms=6, color=C_PRED, mec="white", mew=1.5)
            ax.annotate(f"P{int(pc*100)}", (np.quantile(t, pc), np.quantile(p, pc)),
                        xytext=(6, -10), textcoords="offset points", fontsize=7, color=INK2)
        ax.set_title(name); ax.set_xlabel("truth quantile (°C)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("prediction quantile (°C)")
    fig.suptitle("Quantile–quantile comparison", x=0.07, ha="left", fontsize=11)
    save(fig, out, "02_qq_by_period")

    # ------------------------------------------------------------------ 3. per-pixel maps
    def per_pixel(fn, s):
        T, P = truth.sel(**s).values, pred.sel(**s).values
        res = np.full(mask.shape, np.nan)
        for i, j in zip(*np.where(mask)):
            res[i, j] = fn(P[:, i, j], T[:, i, j])
        return res

    maps = {}
    for name in PERIODS:
        maps[(name, "W1")] = per_pixel(wd, sel[name])
        maps[(name, "bias")] = per_pixel(lambda p, t: np.mean(p - t), sel[name])
        maps[(name, "rmse")] = per_pixel(lambda p, t: np.sqrt(np.mean((p - t) ** 2)), sel[name])
        maps[(name, "r")] = per_pixel(lambda p, t: np.corrcoef(p, t)[0, 1], sel[name])
        maps[(name, "std_ratio")] = per_pixel(lambda p, t: p.std() / t.std(), sel[name])
    specs = [("W1", "Blues", 0, None, "Wasserstein-1 (°C)"),
             ("bias", "RdBu_r", None, None, "mean bias pred−truth (°C)"),
             ("rmse", "Blues", 0, None, "RMSE (°C)"),
             ("r", "Blues_r", None, 1, "Pearson r (daily)"),
             ("std_ratio", "RdBu_r", None, None, "σ_pred / σ_truth")]
    fig, axes = plt.subplots(len(specs), 4, figsize=(14.5, 3.1 * len(specs)))
    for r, (key, cmap, vmin, vmax, lab) in enumerate(specs):
        allv = np.concatenate([maps[(n, key)][mask] for n in PERIODS])
        if key == "bias":
            a = np.nanmax(np.abs(allv)); vmin, vmax = -a, a
        elif key == "std_ratio":
            a = np.nanmax(np.abs(allv - 1)); vmin, vmax = 1 - a, 1 + a
        vmin = np.nanmin(allv) if vmin is None else vmin
        vmax = np.nanmax(allv) if vmax is None else vmax
        for c, name in enumerate(PERIODS):
            im = map_panel(axes[r, c], maps[(name, key)], cmap, vmin, vmax,
                           f"{name}  (land mean {np.nanmean(maps[(name, key)]):.2f})", lon, lat)
        fig.colorbar(im, ax=axes[r, :], shrink=0.85, label=lab)
    save(fig, out, "03_spatial_error_maps")

    # ------------------------------------------------------------------ 4. W1 vs elevation
    dem = xr.open_dataset(DEM)["elevation"].interp(lat=lat, lon=lon).values
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, c in zip(PERIODS, C_PER):
        axes[0].scatter(dem[mask], maps[(name, "W1")][mask], s=14, color=c, alpha=0.75,
                        edgecolor="white", lw=0.4, label=name)
        axes[1].scatter(dem[mask], maps[(name, "bias")][mask], s=14, color=c, alpha=0.75,
                        edgecolor="white", lw=0.4, label=name)
    axes[0].set_ylabel("per-pixel W1 (°C)"); axes[1].set_ylabel("per-pixel bias (°C)")
    axes[1].axhline(0, color=INK2, lw=0.8)
    for ax in axes: ax.set_xlabel("elevation (m, GEBCO)")
    axes[0].legend()
    fig.suptitle("Error vs elevation", x=0.07, ha="left", fontsize=11)
    save(fig, out, "04_error_vs_elevation")

    # ------------------------------------------------------------------ 5. W1 by month / by year
    tl, pl = land(truth, mask), land(pred, mask)
    dfw = []
    times = pd.DatetimeIndex(pred.time.values)
    for (y, mth), idx in pd.Series(np.arange(len(times)), index=times).groupby([times.year, times.month]):
        ii = idx.values
        dfw.append(dict(year=y, month=mth, W1=wd(pl[ii], tl[ii]), bias=np.nanmean(pl[ii] - tl[ii]),
                        rmse=np.sqrt(np.nanmean((pl[ii] - tl[ii]) ** 2))))
    dfw = pd.DataFrame(dfw)
    dfw["period"] = pd.cut(dfw.year, [2010, 2014, 2050, 2100], labels=SUB)
    dfw.to_csv(f"{out}/monthly_W1.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for name, c in zip(PERIODS, C_PER):
        g = (dfw if name == "All test" else dfw[dfw.period == name]).groupby("month")
        for ax, k in zip(axes, ["W1", "bias", "rmse"]):
            ax.plot(g[k].mean().index, g[k].mean().values, "-o", color=c, ms=4, label=name)
    for ax, k in zip(axes, ["Wasserstein-1 (°C)", "bias (°C)", "RMSE (°C)"]):
        ax.set_xticks(range(1, 13)); ax.set_xticklabels("JFMAMJJASOND"); ax.set_ylabel(k)
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[0].legend()
    fig.suptitle("Seasonality of error (computed on each month's land-pixel distribution)",
                 x=0.07, ha="left", fontsize=11)
    save(fig, out, "05_error_by_month")

    # ------------------------------------------------------------------ 6. W1 per fold
    fm = met[(met.model != "ensemble") & (met.period != "all")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    x = np.arange(len(fpreds)); w = 0.2
    for k, (name, c) in enumerate(zip(PERIODS, C_PER)):
        d = fm[fm.period == name]
        axes[0].bar(x + (k - 1.5) * w, d.wasserstein, w * 0.92, color=c, label=name)
        axes[1].bar(x + (k - 1.5) * w, d.rmse, w * 0.92, color=c, label=name)
        e = met[(met.model == "ensemble") & (met.period == name)]
        axes[0].axhline(e.wasserstein.item(), color=c, ls="--", lw=1)
        axes[1].axhline(e.rmse.item(), color=c, ls="--", lw=1)
    for ax, lab in zip(axes, ["Wasserstein-1 (°C)", "RMSE (°C)"]):
        ax.set_xticks(x); ax.set_xticklabels([f"f{i}" for i in x]); ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    axes[0].legend(title="dashed = ensemble")
    fig.suptitle("Per-fold skill vs ensemble", x=0.07, ha="left", fontsize=11)
    save(fig, out, "06_fold_skill")

    # ------------------------------------------------------------------ 7. temporal: spatial-mean series
    tm, pm = truth.mean(("lat", "lon")), pred.mean(("lat", "lon"))
    fstack = xr.concat(fpreds, "fold").mean(("lat", "lon"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharey="row",
                             gridspec_kw=dict(height_ratios=[2.2, 1], width_ratios=[4, 5, 5]))
    for c, name in enumerate(SUB):
        s = sel[name]
        tt, pp, ff = tm.sel(**s), pm.sel(**s), fstack.sel(**s)
        axes[0, c].fill_between(tt.time, ff.min("fold"), ff.max("fold"), color=C_PRED, alpha=0.18, lw=0,
                                label="fold min–max")
        axes[0, c].plot(tt.time, tt, color=C_TRUTH, lw=0.9, label="truth")
        axes[0, c].plot(pp.time, pp, color=C_PRED, lw=0.9, label="ensemble")
        axes[0, c].set_title(name)
        axes[1, c].plot(tt.time, pp - tt, color=INK2, lw=0.6)
        axes[1, c].plot(tt.time, (pp - tt).rolling(time=30, center=True).mean(), color=C_PRED, lw=1.6)
        axes[1, c].axhline(0, color=INK, lw=0.8)
        for ax in axes[:, c]:
            ax.tick_params(axis="x", labelsize=7)
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes[0, 0].set_ylabel("island-mean tasmax (°C)")
    axes[1, 0].set_ylabel("pred − truth (°C)\n(30-day mean in orange)")
    axes[0, 0].legend(loc="lower left", ncol=3)
    fig.suptitle("Daily island-mean tasmax time series", x=0.07, ha="left", fontsize=11)
    save(fig, out, "07_timeseries_island_mean")

    # ------------------------------------------------------------------ 8. seasonal cycle
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=True)
    for ax, name in zip(axes, PERIODS):
        tt = land(truth.sel(**sel[name]), mask); pp = land(pred.sel(**sel[name]), mask)
        mo = pd.DatetimeIndex(truth.sel(**sel[name]).time.values).month
        for arr, col, lab in [(tt, C_TRUTH, "truth"), (pp, C_PRED, "ensemble")]:
            g = pd.DataFrame(arr).groupby(mo)
            mean = g.mean().mean(1); flat = pd.Series(arr.ravel(), index=np.repeat(mo, arr.shape[1]))
            lo = flat.groupby(level=0).quantile(0.1); hi = flat.groupby(level=0).quantile(0.9)
            ax.fill_between(mean.index, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(mean.index, mean, "-o", color=col, ms=4, label=f"{lab} (mean, P10–P90)")
        ax.set_xticks(range(1, 13)); ax.set_xticklabels("JFMAMJJASOND"); ax.set_title(name)
    axes[0].set_ylabel("tasmax over land (°C)"); axes[0].legend(loc="lower left")
    fig.suptitle("Seasonal cycle", x=0.07, ha="left", fontsize=11)
    save(fig, out, "08_seasonal_cycle")

    # ------------------------------------------------------------------ 9. conditional bias (error vs truth)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(4, 40, 1.0)
    for name, c in zip(PERIODS, C_PER):
        t = land(truth.sel(**sel[name]), mask).ravel(); p = land(pred.sel(**sel[name]), mask).ravel()
        b = np.digitize(t, bins)
        df = pd.DataFrame(dict(b=bins[np.clip(b - 1, 0, len(bins) - 1)] + 0.5, e=p - t)).groupby("b")["e"]
        n = df.size(); keep = n[n > 200].index
        axes[0].plot(keep, df.mean()[keep], "-", color=c, label=name)
        axes[0].fill_between(keep, df.quantile(0.1)[keep], df.quantile(0.9)[keep], color=c, alpha=0.12, lw=0)
        e = p - t
        axes[1].hist(e, bins=np.linspace(-6, 6, 121), density=True, histtype="step", color=c, lw=1.6,
                     label=f"{name} (σ={e.std():.2f})")
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xlabel("truth tasmax (°C)"); axes[0].set_ylabel("pred − truth (°C), mean & P10–P90")
    axes[0].set_title("Conditional bias: does the model shrink extremes?")
    axes[1].set_xlabel("pred − truth (°C)"); axes[1].set_ylabel("density"); axes[1].set_title("Error distribution")
    axes[0].legend(); axes[1].legend()
    save(fig, out, "09_conditional_bias_error_hist")

    # ------------------------------------------------------------------ 10. temporal variability / persistence
    def lag1(a):  # mean over pixels of lag-1 autocorr of anomalies (deseasonalised by month)
        out_ = []
        for k in range(a.shape[1]):
            x = a[:, k]; out_.append(np.corrcoef(x[:-1], x[1:])[0, 1])
        return np.array(out_)

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for name, c in zip(PERIODS, C_PER):
        tt_da, pp_da = truth.sel(**sel[name]), pred.sel(**sel[name])
        ta = land(tt_da.groupby("time.month") - tt_da.groupby("time.month").mean(), mask)
        pa = land(pp_da.groupby("time.month") - pp_da.groupby("time.month").mean(), mask)
        dt, dp = np.diff(land(tt_da, mask), axis=0).ravel(), np.diff(land(pp_da, mask), axis=0).ravel()
        lt, lp = lag1(ta), lag1(pa)
        ss_t, ss_p = land(tt_da, mask).std(1), land(pp_da, mask).std(1)
        rows.append(dict(period=name, lag1_truth=lt.mean(), lag1_pred=lp.mean(),
                         std_dayToDay_truth=dt.std(), std_dayToDay_pred=dp.std(),
                         W1_dayToDay=wd(dp, dt), spatial_std_truth=ss_t.mean(), spatial_std_pred=ss_p.mean(),
                         anomaly_std_truth=ta.std(), anomaly_std_pred=pa.std()))
        axes[0].scatter(lt, lp, s=12, color=c, alpha=0.7, edgecolor="white", lw=0.4, label=name)
        xx = np.linspace(-6, 6, 200)
        axes[1].plot(xx, stats.gaussian_kde(dt[::5])(xx), color=c, lw=1.6, label=f"{name} truth")
        axes[1].plot(xx, stats.gaussian_kde(dp[::5])(xx), color=c, lw=1.6, ls="--", label=f"{name} pred")
        axes[2].scatter(ss_t, ss_p, s=4, color=c, alpha=0.3, lw=0, label=name)
    lim = axes[0].get_xlim(); axes[0].plot(lim, lim, color=INK2, ls="--", lw=1)
    axes[0].set_xlabel("lag-1 autocorr of anomalies (truth)"); axes[0].set_ylabel("(prediction)")
    axes[0].set_title("Day-to-day persistence per pixel"); axes[0].legend()
    axes[1].set_xlabel("tasmax(t) − tasmax(t−1) (°C)"); axes[1].set_ylabel("density")
    axes[1].set_title("Day-to-day change distribution"); axes[1].legend(fontsize=7)
    lim = axes[2].get_xlim(); axes[2].plot(lim, lim, color=INK2, ls="--", lw=1)
    axes[2].set_xlabel("spatial σ across island (truth, °C)"); axes[2].set_ylabel("(prediction)")
    axes[2].set_title("Daily spatial variability")
    save(fig, out, "10_temporal_variability")
    pd.DataFrame(rows).to_csv(f"{out}/temporal_stats.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ------------------------------------------------------------------ 11. climate-change signal
    base_t = truth.sel(**sel["2011-2014"]).mean("time"); base_p = pred.sel(**sel["2011-2014"]).mean("time")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    sig_rows = []
    for r, name in enumerate(["2046-2050", "2096-2100"]):
        dT = truth.sel(**sel[name]).mean("time") - base_t
        dP = pred.sel(**sel[name]).mean("time") - base_p
        vmax = float(max(dT.max(), dP.max())); vmin = float(min(dT.min(), dP.min()))
        im = map_panel(axes[r, 0], dT, "Reds", vmin, vmax, f"truth Δ {name} (mean {float(dT.mean()):.2f})", lon, lat)
        map_panel(axes[r, 1], dP, "Reds", vmin, vmax, f"pred Δ {name} (mean {float(dP.mean()):.2f})", lon, lat)
        fig.colorbar(im, ax=axes[r, :2], shrink=0.85, label="warming vs 2011-2014 (°C)")
        d = dP - dT; a = float(np.abs(d).max())
        im2 = map_panel(axes[r, 2], d, "RdBu_r", -a, a, f"pred Δ − truth Δ (mean {float(d.mean()):.2f})", lon, lat)
        fig.colorbar(im2, ax=axes[r, 2], shrink=0.85, label="°C")
        for pc in (50, 90, 99):
            sig_rows.append(dict(period=name, stat=f"P{pc}",
                                 delta_truth=np.percentile(land(truth.sel(**sel[name]), mask), pc)
                                 - np.percentile(land(truth.sel(**sel["2011-2014"]), mask), pc),
                                 delta_pred=np.percentile(land(pred.sel(**sel[name]), mask), pc)
                                 - np.percentile(land(pred.sel(**sel["2011-2014"]), mask), pc)))
        sig_rows.append(dict(period=name, stat="mean", delta_truth=float(dT.mean()), delta_pred=float(dP.mean())))
    fig.suptitle("Climate-change signal preservation (relative to 2011-2014)", x=0.07, ha="left", fontsize=11)
    save(fig, out, "11_climate_signal")
    pd.DataFrame(sig_rows).to_csv(f"{out}/climate_signal.csv", index=False)
    print(pd.DataFrame(sig_rows).round(3).to_string(index=False))

    # ------------------------------------------------------------------ 12. extremes
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    thr = np.percentile(land(truth.sel(**sel["2011-2014"]), mask), 90)  # P90 of baseline truth
    yrs = pd.DatetimeIndex(truth.time.values).year
    exc_t = pd.Series((land(truth, mask) > thr).mean(1), index=yrs).groupby(level=0).mean() * 365
    exc_p = pd.Series((land(pred, mask) > thr).mean(1), index=yrs).groupby(level=0).mean() * 365
    xs = np.arange(len(exc_t))
    axes[0].bar(xs - 0.2, exc_t.values, 0.38, color=C_TRUTH, label="truth")
    axes[0].bar(xs + 0.2, exc_p.values, 0.38, color=C_PRED, label="ensemble")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(exc_t.index, rotation=60, fontsize=7)
    axes[0].set_ylabel("days/yr per pixel"); axes[0].grid(axis="x", visible=False)
    axes[0].set_title(f"Hot days > {thr:.1f} °C (P90 of 2011-2014 truth)"); axes[0].legend()
    pcs = [1, 5, 10, 50, 90, 95, 99, 99.9]
    for name, c in zip(PERIODS, C_PER):
        t = land(truth.sel(**sel[name]), mask); p = land(pred.sel(**sel[name]), mask)
        axes[1].plot(range(len(pcs)), np.percentile(p, pcs) - np.percentile(t, pcs), "-o", color=c, ms=5, label=name)
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_xticks(range(len(pcs))); axes[1].set_xticklabels([f"P{p}" for p in pcs])
    axes[1].set_ylabel("pred − truth percentile (°C)"); axes[1].set_title("Percentile error across the distribution")
    axes[1].legend()
    save(fig, out, "12_extremes")

    json.dump({"hot_day_threshold_C": float(thr)}, open(f"{out}/extra.json", "w"))
    print("done ->", out)


if __name__ == "__main__":
    main()
