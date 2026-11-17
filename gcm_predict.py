"""Apply a trained U-Net run (lighten-io_SD/UNET.py) to CNRM-ESM2-1 GCM predictors and analyse the outputs.

The networks were trained on ALADIN_CORRIGE predictors upscaled onto the GCM grid (mean of the 3 km land
pixels inside each of the 4 GCM cells). Here the same 24-number vector is built from the GCM itself.

Inputs actually available in Data/CMIP_CNRM_ESM21:
    tasmin, hursmax, hursmin, rsds   historical + ssp370
    pr                               historical only (no ssp370 file)
    sfcWind                          missing
Missing predictors (pr, sfcWind) are replaced, for every year, by the ALADIN-upscaled day-of-year
climatology of 1981-2010 (pr is replaced in historical too, so the inputs stay consistent across 2015).

Two GCM variants are predicted:
    gcm_raw : GCM cell values fed as they are
    gcm_adj : GCM values rescaled per calendar month (mean and std) onto the ALADIN-upscaled predictors
              of the 1981-2014 reference; the same transform is applied to every year, so the GCM trend is kept
plus a reference: aladin = the same networks fed with the ALADIN-upscaled predictors (all days, 1981-2100;
days outside the test periods were seen in training by some folds).

Usage:
    python gcm_predict.py /mnt/nas/Data/christian_data/run1 [--stage all|predict|plot]
Outputs: outputs/<run>_gcm/*.nc (predictions) and plots/<run>_gcm/*.png|csv
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

DATA = "/mnt/nas/Data"
GCM_DIR = f"{DATA}/CMIP_CNRM_ESM21"
ALADIN = f"{DATA}/METEO_FRANCE/DONNEES_SIMULATIONS/ALADIN/ALADIN_CORRIGE"
DEM = f"{DATA}/DEM/gebco_2025_n-20.82_s-21.45_w55.15_e55.9.nc"
GCM_GRID = f"{GCM_DIR}/tas_day_CNRM-ESM2-1_historical_r1i1p1f2_gr_18500101-20141231.nc"
UNET_REPO = os.path.expanduser("~/Git/lighten-io_SD")
GCM_VAR = {"tasminAdjust": "tasmin", "hursmaxAdjust": "hursmax", "hursminAdjust": "hursmin",
           "prtotAdjust": "pr", "rsdsAdjust": "rsds", "sfcWindAdjust": None}
FORCE_CLIM = {"prtotAdjust", "sfcWindAdjust"}      # not usable from the GCM over the whole period
Y0, Y1 = 1981, 2100
REF = ("1981", "2014")
K2C = 273.15


# ------------------------------------------------------------------ GCM extraction
def gcm_series(var, centres, scenario_files):
    """(T, n_cells) values of the GCM cells whose centres were used in training."""
    parts = []
    for f in scenario_files:
        with xr.open_dataset(f) as ds:
            da = ds[var]
            pts = [da.sel(lat=la, lon=lo, method="nearest") for la, lo in centres]
            arr = xr.concat(pts, "cell").transpose("time", "cell").load()
        arr = arr.assign_coords(time=arr.time.dt.floor("D"))
        parts.append(arr.sel(time=slice(str(Y0), str(Y1))))
    out = xr.concat(parts, "time")
    _, keep = np.unique(out.time.values, return_index=True)
    return out.isel(time=np.sort(keep))


def gcm_files(var, scenario):
    fs = sorted(glob.glob(f"{GCM_DIR}/{var}_day_CNRM-ESM2-1_{scenario}_r1i1p1f2_gr_*.nc"))
    return [f for f in fs if int(os.path.basename(f).split("_")[-1][9:13]) >= Y0]  # file end year >= 1981


def doy_climatology(times, values, ref_years=(1981, 2010), smooth=31):
    """Day-of-year climatology (smoothed) of values (T, C), evaluated on `times`."""
    t = pd.DatetimeIndex(times)
    sel = (t.year >= ref_years[0]) & (t.year <= ref_years[1])
    doy = np.minimum(t.dayofyear, 365)
    clim = pd.DataFrame(values[sel]).groupby(doy[sel]).mean().reindex(range(1, 366)).values
    pad = np.concatenate([clim[-smooth:], clim, clim[:smooth]])
    kern = np.ones(smooth) / smooth
    sm = np.stack([np.convolve(pad[:, c], kern, mode="same") for c in range(clim.shape[1])], 1)[smooth:-smooth]
    return sm[doy - 1]


def monthly_scaling(times, x, ref_x, ref_times, ref_period):
    """Rescale x (T,C) per calendar month so that its ref-period mean/std match ref_x."""
    t, tr = pd.DatetimeIndex(times), pd.DatetimeIndex(ref_times)
    in_ref = (t.year >= int(ref_period[0])) & (t.year <= int(ref_period[1]))
    in_ref_r = (tr.year >= int(ref_period[0])) & (tr.year <= int(ref_period[1]))
    out = np.empty_like(x)
    for m in range(1, 13):
        a, b = (t.month == m), (tr.month == m)
        mu_g, sd_g = x[a & in_ref].mean(0), x[a & in_ref].std(0)
        mu_r, sd_r = ref_x[b & in_ref_r].mean(0), ref_x[b & in_ref_r].std(0)
        out[a] = (x[a] - mu_g) / (sd_g + 1e-12) * sd_r + mu_r
    return out


# ------------------------------------------------------------------ prediction
def predict(run_dir, feats_by_variant, dem, land_mask, unet):
    import torch
    torch.set_num_threads(os.cpu_count())
    folds = sorted(glob.glob(f"{run_dir}/fold_*/best.pt"))
    out = {}
    for name, feats in feats_by_variant.items():
        acc = None
        for fpath in folds:
            fd = os.path.dirname(fpath)
            stats = json.load(open(f"{fd}/normalization_stats.json"))
            model = unet.build_model(stats, feats.shape[1], dem, torch.device("cpu"))
            model.load_state_dict(torch.load(fpath, map_location="cpu", weights_only=True))
            model.eval()
            x = torch.from_numpy(unet.normalize_features(feats, stats))
            preds = []
            t0 = time.time()
            with torch.no_grad():
                for i in range(0, len(x), 1024):
                    preds.append(model(x[i:i + 1024]).numpy()[:, 0])
            p = np.concatenate(preds) * stats["y_std"] + stats["y_mean"]
            acc = p if acc is None else acc + p
            print(f"  {name} | {os.path.basename(fd)} | {time.time() - t0:.0f} s", flush=True)
        out[name] = np.where(land_mask[None], acc / len(folds), np.nan).astype(np.float32)
    return out


def stage_predict(run_dir, out_nc):
    sys.path.insert(0, UNET_REPO)
    import UNET as unet

    print("ALADIN predictors (upscaled to the GCM cells)...", flush=True)
    feats_al, Y, times, lat, lon, names = unet.load_data(ALADIN, ["historical", "ssp370"],
                                                         list(GCM_VAR), "tasmaxAdjust", GCM_GRID)
    dem = unet.load_dem(DEM, lat, lon)
    land_mask = np.load(f"{run_dir}/land_mask.npy")
    labels, centres = unet.gcm_cells(lat, lon, GCM_GRID)
    n_cells = len(centres)
    assert feats_al.shape[1] == len(GCM_VAR) * n_cells, "cells with no land? feature layout mismatch"
    print("cell centres:", centres)

    print("GCM predictors...", flush=True)
    gtimes = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D").values
    raw = np.full((len(gtimes), feats_al.shape[1]), np.nan, np.float32)
    gcm_avail = {}
    for v, (pred_name, gv) in enumerate(GCM_VAR.items()):
        cols = slice(v * n_cells, (v + 1) * n_cells)
        if gv is None:
            continue
        files = gcm_files(gv, "historical") + gcm_files(gv, "ssp370")
        s = gcm_series(gv, centres, files)
        idx = np.searchsorted(gtimes, s.time.values)
        raw[idx, cols] = s.values
        gcm_avail[pred_name] = (str(s.time.values[0])[:10], str(s.time.values[-1])[:10])
        print(f"  {gv}: {gcm_avail[pred_name]}", flush=True)
    tx = gcm_series("tasmax", centres, gcm_files("tasmax", "historical") + gcm_files("tasmax", "ssp370"))

    # missing / forced -> ALADIN day-of-year climatology
    clim = doy_climatology(times, feats_al)
    clim_g = doy_climatology(times, feats_al)  # same climatology, evaluated on the GCM calendar below
    clim_g = pd.DataFrame(clim_g, index=pd.DatetimeIndex(times)).reindex(pd.DatetimeIndex(gtimes)).ffill().bfill().values
    adj = raw.copy()
    al_on_g = pd.DataFrame(feats_al, index=pd.DatetimeIndex(times)).reindex(pd.DatetimeIndex(gtimes)).values
    for v, pred_name in enumerate(GCM_VAR):
        cols = slice(v * n_cells, (v + 1) * n_cells)
        if pred_name in FORCE_CLIM:
            raw[:, cols] = adj[:, cols] = clim_g[:, cols]
        else:
            ok = np.isfinite(al_on_g[:, cols]).all(1)
            adj[:, cols] = monthly_scaling(gtimes, raw[:, cols], al_on_g[ok][:, cols], gtimes[ok], REF)
    assert np.isfinite(raw).all() and np.isfinite(adj).all()

    # sensitivity: ALADIN predictors with pr & sfcWind -> climatology (how much is lost by the missing inputs?)
    al_clim = feats_al.copy()
    for v, pred_name in enumerate(GCM_VAR):
        if pred_name in FORCE_CLIM:
            al_clim[:, v * n_cells:(v + 1) * n_cells] = clim[:, v * n_cells:(v + 1) * n_cells]

    print("Predicting (10 folds, CPU)...", flush=True)
    preds = predict(run_dir, {"aladin": feats_al, "aladin_climfill": al_clim, "gcm_raw": raw, "gcm_adj": adj},
                    dem, land_mask, unet)

    os.makedirs(out_nc, exist_ok=True)
    enc = {"tasmaxAdjust": {"zlib": True, "complevel": 4}}
    for name, p in preds.items():
        t = times if name.startswith("aladin") else gtimes
        xr.DataArray(p, dims=("time", "lat", "lon"), coords={"time": t, "lat": lat, "lon": lon},
                     name="tasmaxAdjust", attrs={"units": "K", "source": f"U-Net {run_dir}, 10-fold ensemble, input={name}"}
                     ).to_netcdf(f"{out_nc}/tasmaxAdjust_unet_{name}_{Y0}-{Y1}.nc", encoding=enc)
    xr.DataArray(Y[:, 0], dims=("time", "lat", "lon"), coords={"time": times, "lat": lat, "lon": lon},
                 name="tasmaxAdjust").to_netcdf(f"{out_nc}/tasmaxAdjust_aladin_truth_{Y0}-{Y1}.nc", encoding=enc)
    cn = [f"{la:.2f}N,{lo:.2f}E" for la, lo in centres]
    xr.Dataset({"raw": (("time", "feature"), raw), "adj": (("time", "feature"), adj),
                "tasmax_gcm": (("time_tx", "cell"), tx.values)},
               coords={"time": gtimes, "feature": names, "time_tx": tx.time.values, "cell": cn}
               ).to_netcdf(f"{out_nc}/gcm_predictors.nc")
    xr.Dataset({"aladin": (("time", "feature"), feats_al)}, coords={"time": times, "feature": names}
               ).to_netcdf(f"{out_nc}/aladin_upscaled_predictors.nc")
    json.dump({"gcm_available": gcm_avail, "replaced_by_climatology": sorted(FORCE_CLIM), "ref_period": REF,
               "cell_centres": centres}, open(f"{out_nc}/info.json", "w"), indent=2)


# ------------------------------------------------------------------ plots
def stage_plot(run_dir, out_nc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as st

    C_T, C_AL, C_ADJ, C_RAW, C_GCM = "#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#52514e"
    INK2 = "#52514e"
    C_PER = ["#2a78d6", "#1baf7a", "#e34948"]
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(out, exist_ok=True)
    mask = np.load(f"{run_dir}/land_mask.npy")
    o = lambda n: xr.open_dataarray(f"{out_nc}/tasmaxAdjust_{n}_{Y0}-{Y1}.nc").load() - K2C
    D = {"ALADIN truth": o("aladin_truth"), "U-Net (ALADIN inputs)": o("unet_aladin"),
         "U-Net (GCM adjusted)": o("unet_gcm_adj"), "U-Net (GCM raw)": o("unet_gcm_raw")}
    climfill = o("unet_aladin_climfill")
    COL = dict(zip(D, [C_T, C_AL, C_ADJ, C_RAW]))
    T = D["ALADIN truth"]
    for k in D:
        D[k] = D[k].sel(time=T.time)          # common calendar (ALADIN has no missing days, GCM is gregorian)
    climfill = climfill.sel(time=T.time)
    P = {"1985-2014": ("1985", "2014"), "2041-2070": ("2041", "2070"), "2071-2100": ("2071", "2100")}
    land = lambda da: da.values[:, mask]
    sv = lambda da, p: da.sel(time=slice(*P[p]))
    W = lambda a, b: st.wasserstein_distance(np.ravel(a), np.ravel(b))
    pg = xr.open_dataset(f"{out_nc}/gcm_predictors.nc")
    pal = xr.open_dataset(f"{out_nc}/aladin_upscaled_predictors.nc")["aladin"]
    lon, lat = T.lon.values, T.lat.values

    def save(fig, name):
        fig.savefig(f"{out}/{name}.png"); plt.close(fig); print("  ->", name)

    # ---------- metrics table
    rows = []
    test = np.isin(pd.DatetimeIndex(T.time.values).year, list(range(2011, 2015)) + list(range(2046, 2051)) + list(range(2096, 2101)))
    for p in list(P) + ["1981-2100"]:
        t = land(T) if p == "1981-2100" else land(sv(T, p))
        for k in list(D)[1:]:
            x = land(D[k]) if p == "1981-2100" else land(sv(D[k], p))
            e = x - t
            rows.append(dict(period=p, input=k, mean_bias=e.mean(), W1=W(x, t), ks=st.ks_2samp(x.ravel()[::5], t.ravel()[::5]).statistic,
                             std_ratio=x.std() / t.std(), p99_err=np.percentile(x, 99) - np.percentile(t, 99),
                             p01_err=np.percentile(x, 1) - np.percentile(t, 1),
                             daily_r_islandmean=np.corrcoef(x.mean(1), t.mean(1))[0, 1],
                             rmse_daily=np.sqrt((e ** 2).mean())))
    met = pd.DataFrame(rows); met.to_csv(f"{out}/metrics_gcm.csv", index=False)
    print(met.round(3).to_string(index=False))
    t_test, a_test, c_test = land(T)[test], land(D["U-Net (ALADIN inputs)"])[test], land(climfill)[test]
    sens = dict(rmse_all_inputs=float(np.sqrt(((a_test - t_test) ** 2).mean())),
                rmse_pr_sfcWind_climatology=float(np.sqrt(((c_test - t_test) ** 2).mean())),
                W1_all_inputs=W(a_test, t_test), W1_pr_sfcWind_climatology=W(c_test, t_test))
    json.dump(sens, open(f"{out}/sensitivity_missing_predictors.json", "w"), indent=2)
    print("sensitivity (test days, ALADIN inputs):", {k: round(v, 3) for k, v in sens.items()})

    # ---------- 01 predictors: GCM raw / adjusted vs ALADIN upscaled (4-cell mean)
    names = [str(f) for f in pg.feature.values]
    vars_ = list(GCM_VAR)
    units = {"tasminAdjust": "K", "hursmaxAdjust": "%", "hursminAdjust": "%", "prtotAdjust": "kg m⁻² s⁻¹",
             "rsdsAdjust": "W m⁻²", "sfcWindAdjust": "m s⁻¹"}
    fig, axes = plt.subplots(2, 6, figsize=(22, 6.5))
    for v, var in enumerate(vars_):
        cols = [i for i, n in enumerate(names) if n.startswith(var + "@")]
        series = {"ALADIN upscaled": (pal.time.values, pal.values[:, cols].mean(1), C_T),
                  "GCM raw": (pg.time.values, pg["raw"].values[:, cols].mean(1), C_RAW),
                  "GCM adjusted": (pg.time.values, pg["adj"].values[:, cols].mean(1), C_ADJ)}
        for lab, (tt, vv, c) in series.items():
            s = pd.Series(vv, index=pd.DatetimeIndex(tt))
            r = s[REF[0]:REF[1]]
            axes[0, v].plot(range(1, 13), r.groupby(r.index.month).mean(), "-o", ms=3, color=c, label=lab)
            a = s.groupby(s.index.year).mean()
            axes[1, v].plot(a.index, a.values, color=c, lw=1.3, label=lab)
        tag = " (climatology fill)" if var in FORCE_CLIM else ""
        axes[0, v].set_title(f"{var}{tag}"); axes[0, v].set_xticks(range(1, 13)); axes[0, v].set_xticklabels("JFMAMJJASOND")
        axes[0, v].set_ylabel(units[var]); axes[1, v].set_ylabel(f"annual mean ({units[var]})")
    axes[0, 0].legend(); axes[0, 0].text(0.0, 1.13, "Monthly climatology 1981-2014 (4-cell mean)", transform=axes[0, 0].transAxes, fontsize=10)
    axes[1, 0].text(0.0, 1.05, "Annual means 1981-2100", transform=axes[1, 0].transAxes, fontsize=10)
    save(fig, "01_predictors_gcm_vs_aladin")

    # ---------- 02 annual island-mean tasmax
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, da in D.items():
        a = da.mean(("lat", "lon")).groupby("time.year").mean()
        ax.plot(a.year, a, color=COL[k], lw=2 if k == "ALADIN truth" else 1.5, label=k)
    txg = pg["tasmax_gcm"].mean("cell") - K2C
    a = txg.groupby("time_tx.year").mean()
    ax.plot(a.year, a, color=C_GCM, lw=1.2, ls="--", label="GCM tasmax, raw (4 cells incl. ocean)")
    for y0, y1 in [(2011, 2014), (2046, 2050), (2096, 2100)]:
        ax.axvspan(y0 - 0.5, y1 + 0.5, color="#e4e3df", alpha=0.6, lw=0)
    ax.set_ylabel("annual mean tasmax over land (°C)"); ax.legend(ncol=2, loc="upper left")
    ax.set_title("Annual island-mean tasmax, 1981-2100 (grey bands: U-Net test periods)", loc="left")
    save(fig, "02_annual_timeseries")

    # ---------- 03 densities per 30-yr period
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, p in zip(axes, P):
        t = land(sv(T, p)).ravel()
        x = np.linspace(t.min() - 4, t.max() + 4, 400)
        for k, da in D.items():
            v = land(sv(da, p)).ravel()
            lab = k if k == "ALADIN truth" else f"{k}  W1={W(v, t):.2f}"
            ax.plot(x, st.gaussian_kde(v[::11])(x), color=COL[k], lw=2.2 if k == "ALADIN truth" else 1.6, label=lab)
        ax.set_title(p); ax.set_xlabel("tasmax (°C)"); ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("density (land pixels × days)")
    save(fig, "03_density_30yr_periods")

    # ---------- 04 QQ
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    q = np.linspace(0.001, 0.999, 300)
    for ax, p in zip(axes, P):
        qt = np.quantile(land(sv(T, p)), q)
        ax.plot([qt[0], qt[-1]], [qt[0], qt[-1]], color=INK2, ls="--", lw=1)
        for k in list(D)[1:]:
            ax.plot(qt, np.quantile(land(sv(D[k], p)), q), color=COL[k], lw=1.8, label=k)
        ax.set_title(p); ax.set_xlabel("ALADIN quantile (°C)"); ax.set_aspect("equal")
    axes[0].set_ylabel("U-Net quantile (°C)"); axes[0].legend(fontsize=8)
    save(fig, "04_qq_30yr_periods")

    # ---------- 05 maps: climatology, bias and per-pixel W1 (GCM adjusted)
    def pp(fn, a, b):
        A, B = a.values, b.values; r = np.full(mask.shape, np.nan)
        for i, j in zip(*np.where(mask)):
            r[i, j] = fn(A[:, i, j], B[:, i, j])
        return r
    fig, axes = plt.subplots(4, 3, figsize=(12, 13))
    k = "U-Net (GCM adjusted)"
    mt = {p: sv(T, p).mean("time") for p in P}; mp = {p: sv(D[k], p).mean("time") for p in P}
    vmin, vmax = min(float(m.min()) for m in mt.values()), max(float(m.max()) for m in mt.values())
    bias = {p: (mp[p] - mt[p]).values for p in P}; bb = max(np.nanmax(np.abs(b)) for b in bias.values())
    w1 = {p: pp(W, sv(D[k], p), sv(T, p)) for p in P}; wm = max(np.nanmax(w) for w in w1.values())
    rows_ = [("ALADIN mean", mt, "YlOrRd", vmin, vmax, "°C"), (f"{k} mean", mp, "YlOrRd", vmin, vmax, "°C"),
             ("bias (U-Net − ALADIN)", {p: bias[p] for p in P}, "RdBu_r", -bb, bb, "°C"),
             ("per-pixel W1", w1, "Blues", 0, wm, "°C")]
    for r, (title, dd, cmap, a, b, u) in enumerate(rows_):
        for c, p in enumerate(P):
            im = axes[r, c].pcolormesh(lon, lat, np.asarray(dd[p]), cmap=cmap, vmin=a, vmax=b, shading="nearest")
            axes[r, c].set_title(f"{title}\n{p} (land mean {np.nanmean(np.asarray(dd[p])):.2f})", fontsize=8.5)
            axes[r, c].set_aspect("equal"); axes[r, c].grid(False); axes[r, c].tick_params(labelsize=7)
        fig.colorbar(im, ax=axes[r, :], shrink=0.85, label=u)
    save(fig, "05_maps_gcm_adjusted")

    # ---------- 06 warming signal maps and table
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    sig = []
    for r, p in enumerate(["2041-2070", "2071-2100"]):
        dd = {kk: (sv(D[kk], p).mean("time") - sv(D[kk], "1985-2014").mean("time")) for kk in D}
        a, b = min(float(v.min()) for v in dd.values()), max(float(v.max()) for v in dd.values())
        for c, (kk, v) in enumerate(dd.items()):
            im = axes[r, c].pcolormesh(lon, lat, v, cmap="Reds", vmin=a, vmax=b, shading="nearest")
            axes[r, c].set_title(f"{kk}\nΔ {p} (mean {float(v.mean()):.2f})", fontsize=8.5)
            axes[r, c].set_aspect("equal"); axes[r, c].grid(False); axes[r, c].tick_params(labelsize=7)
            sig.append(dict(period=p, source=kk, delta_mean=float(v.mean())))
        fig.colorbar(im, ax=axes[r, :], shrink=0.85, label="warming vs 1985-2014 (°C)")
    dg = txg.sel(time_tx=slice("1985", "2014")).mean()
    for p in ["2041-2070", "2071-2100"]:
        sig.append(dict(period=p, source="GCM tasmax raw", delta_mean=float(txg.sel(time_tx=slice(*P[p])).mean() - dg)))
    save(fig, "06_warming_signal_maps")
    sig = pd.DataFrame(sig); sig.to_csv(f"{out}/warming_signal.csv", index=False); print(sig.round(3).to_string(index=False))

    # ---------- 07 seasonal cycle
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, p in zip(axes, P):
        for kk, da in D.items():
            m = sv(da, p).mean(("lat", "lon")).groupby("time.month").mean()
            ax.plot(m.month, m, "-o", ms=3, color=COL[kk], lw=2 if kk == "ALADIN truth" else 1.4, label=kk)
        ax.set_xticks(range(1, 13)); ax.set_xticklabels("JFMAMJJASOND"); ax.set_title(p)
    axes[0].set_ylabel("island-mean tasmax (°C)"); axes[0].legend(fontsize=8)
    save(fig, "07_seasonal_cycle")

    # ---------- 08 extremes
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    thr = np.percentile(land(sv(T, "1985-2014")), 90)
    yrs = pd.DatetimeIndex(T.time.values).year
    for kk, da in D.items():
        e = pd.Series((land(da) > thr).mean(1), index=yrs).groupby(level=0).mean() * 365.25
        axes[0].plot(e.index, e.values, color=COL[kk], lw=2 if kk == "ALADIN truth" else 1.4, label=kk)
    axes[0].set_ylabel("days/yr per land pixel"); axes[0].set_title(f"Hot days > {thr:.1f} °C (P90 of ALADIN 1985-2014)", loc="left")
    axes[0].legend(fontsize=8)
    pcs = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    for p, c in zip(P, C_PER):
        t = land(sv(T, p))
        for kk, ls in [("U-Net (GCM adjusted)", "-"), ("U-Net (ALADIN inputs)", ":")]:
            axes[1].plot(range(len(pcs)), np.percentile(land(sv(D[kk], p)), pcs) - np.percentile(t, pcs), ls, marker="o",
                         ms=4, color=c, label=f"{p} {kk.split('(')[1][:-1]}")
    axes[1].axhline(0, color="k", lw=0.8); axes[1].set_xticks(range(len(pcs))); axes[1].set_xticklabels([f"P{x}" for x in pcs])
    axes[1].set_ylabel("U-Net − ALADIN percentile (°C)"); axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_title("Percentile errors (solid: GCM adjusted, dotted: ALADIN inputs)", loc="left")
    save(fig, "08_extremes")

    # ---------- 09 example days: GCM coarse field vs downscaled vs ALADIN
    with xr.open_dataset(gcm_files("tasmax", "ssp370")[0]) as ds:
        reg = ds["tasmax"].sel(lat=slice(-24, -18), lon=slice(52.5, 59))
        days = ["2050-02-15", "2090-07-15"]
        g = reg.sel(time=[np.datetime64(d + "T12:00") for d in days], method="nearest").load() - K2C
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for r, d in enumerate(days):
        fields = [("GCM tasmax (raw, 1.4°)", g.isel(time=r))] + \
                 [(kk, D[kk].sel(time=d)) for kk in ["ALADIN truth", "U-Net (GCM adjusted)", "U-Net (ALADIN inputs)"]]
        vals = np.concatenate([np.ravel(f.values)[np.isfinite(np.ravel(f.values))] for _, f in fields[1:]])
        a, b = np.percentile(vals, 1), np.percentile(vals, 99)
        for c, (title, f) in enumerate(fields):
            im = axes[r, c].pcolormesh(f.lon, f.lat, f, cmap="YlOrRd", vmin=a, vmax=b, shading="nearest")
            if c == 0:
                axes[r, c].add_patch(plt.Rectangle((lon.min(), lat.min()), np.ptp(lon), np.ptp(lat), fill=False, ec="k", lw=1))
            axes[r, c].set_title(f"{title}\n{d}", fontsize=8.5); axes[r, c].set_aspect("equal"); axes[r, c].grid(False)
        fig.colorbar(im, ax=axes[r, :], shrink=0.85, label="tasmax (°C)")
    save(fig, "09_example_days")

    # ---------- 10 W1 over time (5-yr windows) per input
    fig, ax = plt.subplots(figsize=(12, 4))
    for kk in list(D)[1:]:
        ws = []
        for y in range(1981, 2100, 5):
            sl = slice(str(y), str(y + 4))
            ws.append((y + 2, W(land(D[kk].sel(time=sl)), land(T.sel(time=sl)))))
        ws = np.array(ws); ax.plot(ws[:, 0], ws[:, 1], "-o", ms=4, color=COL[kk], label=kk)
    ax.set_ylabel("Wasserstein-1 vs ALADIN (°C)"); ax.set_title("Distributional distance in 5-year windows", loc="left"); ax.legend()
    save(fig, "10_wasserstein_over_time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stage", default="all", choices=["all", "predict", "plot"])
    a = ap.parse_args()
    run = os.path.basename(os.path.normpath(a.run_dir))
    out_nc, out = os.path.join("outputs", f"{run}_gcm"), os.path.join("plots", f"{run}_gcm")
    if a.stage in ("all", "predict"):
        stage_predict(a.run_dir, out_nc)
    if a.stage in ("all", "plot"):
        stage_plot(a.run_dir, out_nc, out)


if __name__ == "__main__":
    main()
