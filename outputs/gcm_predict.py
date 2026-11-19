"""Apply a trained U-Net run (UNET.py) to CNRM-ESM2-1 GCM predictors and analyse the outputs.

The networks were trained on ALADIN_CORRIGE predictors upscaled onto the GCM grid: the mean of the 3 km
land pixels inside each of the 2 x 2 GCM cells over La Reunion. Here the same inputs are built from the
GCM itself, by joining the historical and ssp370 GCM files into one daily series (1981-2014 from historical,
2015-2100 from ssp370), exactly like the ALADIN training data (historical + ssp370 joined).

Everything is read from the run itself (fold_*/normalization_stats.json): predictors, target, layout,
GCM cells, test periods. Both layouts work:
    2D GCM grid + sin/cos day of year (run2 and later): inputs (T, V, 2, 2) + (T, 2)
    flat vector (run1): (T, V x 4 cells)

GCM data (--gcm-dir, default <data-dir>/CMIP_CNRM_ESM21):
    <gcm-dir>/historical/<var>_day_CNRM-ESM2-1_historical_*.nc
    <gcm-dir>/ssp370/<var>_day_CNRM-ESM2-1_ssp370_*.nc
    Files are looked for in the scenario folder first, then in every other sub-folder (fallback, e.g.
    to_check/), and only the files needed to cover the years are kept.
    The files used are listed in info.json. Predictor -> GCM variable: drop "Adjust", prtot -> pr.
    A predictor with no GCM file covering 1981-2100 (e.g. sfcWind: only sfcWindmax exists, ssp370 only) is
    replaced, every year, by the ALADIN-upscaled day-of-year climatology of 1981-2010.

Inputs predicted (each with the 10-fold ensemble = mean of the folds):
    gcm_raw         : GCM cell values fed as they are
    gcm_adj         : GCM values rescaled per calendar month and per cell (mean and std) onto the
                      ALADIN-upscaled predictors of 1981-2014; the same transform for every year, so the
                      GCM trend is kept. Needed because a GCM cell averages land and ocean, while training
                      used land pixels only.
    aladin          : reference, the same networks fed with the ALADIN-upscaled predictors (1981-2100;
                      days outside the test periods were seen in training by some folds)
    aladin_climfill : only if some predictor is filled with climatology: ALADIN inputs with those predictors
                      replaced the same way (what is lost by the missing GCM inputs)

Usage:
    python outputs/gcm_predict.py ~/nas/christian_data/run2               # predict + plots
    python outputs/gcm_predict.py RUN --stage predict --device cuda                 # on the GPU machine
    python outputs/gcm_predict.py RUN --stage plot
    python outputs/gcm_predict.py RUN --scenario ssp585                             # other projection
Outputs: outputs/<run>_gcm/*.nc (predictions, predictors, info.json) and plots/<run>_gcm/*.png|csv
(change with --out-nc / --out-plots).
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]                      # outputs/ sits inside the repository
sys.path.insert(0, str(REPO))
import UNET as unet  # noqa: E402

Y0, Y1 = 1981, 2100
REF = ("1981", "2014")                                           # reference period of the monthly rescaling
CLIM_YEARS = (1981, 2010)                                        # climatology used to fill missing predictors
K2C = 273.15


# ------------------------------------------------------------------ run description
def run_info(run_dir):
    stats_files = sorted(glob.glob(f"{run_dir}/fold_*/normalization_stats.json"))
    if not stats_files:
        raise SystemExit(f"no fold_*/normalization_stats.json in {run_dir}")
    s = json.load(open(stats_files[0]))
    cfg_file = Path(run_dir) / "run_config.json"
    cfg = json.load(open(cfg_file)) if cfg_file.exists() else {}
    return dict(predictors=s.get("predictors") or cfg["predictors"], target=s.get("target") or cfg.get("target", unet.TARGET),
                scenarios=s.get("scenarios") or cfg.get("scenarios", unet.SCENARIOS),
                test_periods=s.get("test_periods") or cfg.get("test_periods", unet.TEST_PERIODS),
                projection_start=cfg.get("projection_start", unet.PROJECTION_START),
                layout=s.get("input_layout", "flat_1d"), land_only_dem="land_frac_path" in s,
                gcm_grid=s.get("gcm_grid"))


def gcm_name(predictor):
    base = predictor[:-len("Adjust")] if predictor.endswith("Adjust") else predictor
    return {"prtot": "pr"}.get(base, base)


def legacy_dem(path, lat, lon):
    # DEM as built before the land-only version (2026-11-18), for runs trained with it (e.g. run1)
    with xr.open_dataset(path) as ds:
        dem = ds["elevation"].load()
    window = {d: max(1, int(round(abs(float(c[1] - c[0])) / abs(float(dem[d][1] - dem[d][0])))))
              for d, c in (("lat", lat), ("lon", lon))}
    dem = dem.rolling(window, center=True, min_periods=1).mean()
    return dem.interp(lat=lat, lon=lon).clip(min=0).fillna(0).values.astype(np.float32)


# ------------------------------------------------------------------ GCM extraction
def file_years(f):
    m = re.search(r"_(\d{4})\d{4}-(\d{4})\d{4}\.nc$", os.path.basename(f))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def find_files(gcm_dir, var, scenario, y0, y1):
    """Files of `var` for `scenario` overlapping y0..y1: the scenario folder first, then any other sub-folder.
    Returns (files, covered) where covered says whether every year y0..y1 is in at least one file."""
    pattern = f"{var}_day_*_{scenario}_*.nc"
    cands = sorted(glob.glob(f"{gcm_dir}/{scenario}/{pattern}")) + \
        sorted(f for f in glob.glob(f"{gcm_dir}/*/{pattern}") if Path(f).parent.name != scenario)
    files, years = [], set()
    for f in cands:
        a, b = file_years(f)
        if a is None or b < y0 or a > y1:
            continue
        new = set(range(max(a, y0), min(b, y1) + 1)) - years
        if new:                                                  # skip a file whose years are already covered
            files.append(f)
            years |= new
    return files, years >= set(range(y0, y1 + 1))


def gcm_cells_series(files, var, cell_lat, cell_lon, y0, y1):
    """(time, rows, cols) values of the GCM cells used in training, from the files joined in time."""
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            da = ds[var].sel(lat=list(cell_lat), lon=[v % 360 for v in cell_lon], method="nearest")
            if np.abs(da.lat.values - np.array(cell_lat)).max() > 0.1 or \
               np.abs(da.lon.values - np.array(cell_lon) % 360).max() > 0.1:
                raise ValueError(f"{f}: grid does not match the training GCM cells")
            da = da.sel(time=slice(str(y0), str(y1))).load()
        parts.append(da.assign_coords(time=da.time.dt.floor("D").values.astype("datetime64[ns]")))
    out = xr.concat(parts, "time").sortby("time")
    _, keep = np.unique(out.time.values, return_index=True)
    return out.isel(time=np.sort(keep)).transpose("time", "lat", "lon")


def gcm_variable(gcm_dir, var, segments, cell_lat, cell_lon):
    """Joins the scenario segments, e.g. [("historical", 1981, 2014), ("ssp370", 2015, 2100)].
    Returns (DataArray or None, files used, missing segments)."""
    pieces, used, missing = [], [], []
    for scen, y0, y1 in segments:
        files, covered = find_files(gcm_dir, var, scen, y0, y1)
        if not covered:
            missing.append(f"{scen} {y0}-{y1}")
            continue
        pieces.append(gcm_cells_series(files, var, cell_lat, cell_lon, y0, y1))
        used += files
    if missing:
        return None, used, missing
    return xr.concat(pieces, "time"), used, []


# ------------------------------------------------------------------ transforms
def doy_climatology(times, values, ref_years=CLIM_YEARS, smooth=31):
    """Smoothed day-of-year climatology of values (T, ...), evaluated on `times`."""
    t = pd.DatetimeIndex(times)
    flat = values.reshape(len(values), -1)
    sel = (t.year >= ref_years[0]) & (t.year <= ref_years[1])
    doy = np.minimum(t.dayofyear, 365)
    clim = pd.DataFrame(flat[sel]).groupby(doy[sel]).mean().reindex(range(1, 366)).values
    pad = np.concatenate([clim[-smooth:], clim, clim[:smooth]])
    kern = np.ones(smooth) / smooth
    sm = np.stack([np.convolve(pad[:, c], kern, mode="same") for c in range(clim.shape[1])], 1)[smooth:-smooth]
    return sm[doy - 1].reshape(values.shape).astype(np.float32)


def monthly_scaling(times, x, ref_x, ref_period=REF):
    """Rescale x (T, ...) per calendar month, independently for each cell, so that its ref-period mean and
    std match ref_x (same calendar). The same transform is applied to every year."""
    t = pd.DatetimeIndex(times)
    in_ref = (t.year >= int(ref_period[0])) & (t.year <= int(ref_period[1]))
    out = np.empty_like(x)
    for m in range(1, 13):
        a = t.month == m
        g, r = x[a & in_ref], ref_x[a & in_ref]
        out[a] = (x[a] - g.mean(0)) / (g.std(0) + 1e-12) * r.std(0) + r.mean(0)
    return out


# ------------------------------------------------------------------ prediction
def predict(run_dir, variants, season, grid, dem, land_mask, device, batch):
    import torch
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
    folds = sorted(glob.glob(f"{run_dir}/fold_*/best.pt"))
    out = {}
    for name, coarse in variants.items():
        acc = None
        for fpath in folds:
            fd = os.path.dirname(fpath)
            stats = json.load(open(f"{fd}/normalization_stats.json"))
            if stats.get("coarse_shape") is not None:            # 2D GCM grid (+ sin/cos)
                inputs = {"x_coarse": torch.from_numpy(unet.normalize_features(coarse, stats))}
                if stats.get("seasonal"):
                    inputs["x_1D"] = torch.from_numpy(season)
                n_features = stats.get("n_features", 0)
            else:                                                # older flat vector (run1)
                flat, _ = unet.flatten_grid(coarse, grid, stats["predictors"])
                inputs = {"x_1D": torch.from_numpy(unet.normalize_features(flat, stats))}
                n_features = flat.shape[1]
            model = unet.build_model(stats, n_features, dem, device)
            model.load_state_dict(torch.load(fpath, map_location=device, weights_only=True))
            model.eval()
            preds, t0 = [], time.time()
            with torch.no_grad():
                for i in range(0, len(coarse), batch):
                    x = {k: v[i:i + batch].to(device) for k, v in inputs.items()}
                    preds.append(model(**x).float().cpu().numpy()[:, 0])
            p = np.concatenate(preds) * stats["y_std"] + stats["y_mean"]
            acc = p if acc is None else acc + p
            print(f"  {name} | {os.path.basename(fd)} | {time.time() - t0:.0f} s", flush=True)
        out[name] = np.where(land_mask[None], acc / len(folds), np.nan).astype(np.float32)
    return out


def stage_predict(a, info, out_nc):
    import torch
    preds_, target = info["predictors"], info["target"]
    if info["layout"] == "flat_1d" and info["gcm_grid"] is None:
        print("older run: GCM cells taken from the GCM grid file")
    data_dir = Path(a.data_dir).expanduser()
    ns = argparse.Namespace(data_dir=data_dir, aladin_dir=None, dem_path=None, land_frac_path=None,
                            gcm_grid_file=a.gcm_grid_file, models_dir=out_nc, cache_dir=a.cache_dir or None,
                            scenarios=info["scenarios"], predictors=preds_, target=target)
    ns = unet.resolve_paths(ns)
    gcm_dir = Path(a.gcm_dir).expanduser() if a.gcm_dir else data_dir / "CMIP_CNRM_ESM21"

    print("ALADIN predictors upscaled to the GCM cells (as in training)...", flush=True)
    coarse_al, Y, times, lat, lon, grid, dem = unet.load_grid_cached(ns)
    if not info["land_only_dem"]:
        print("run trained with the older DEM (no land mask): rebuilding that DEM")
        dem = legacy_dem(ns.dem_path, lat, lon)
    land_mask = np.load(f"{a.run_dir}/land_mask.npy")
    keep = (times >= np.datetime64(f"{Y0}-01-01")) & (times <= np.datetime64(f"{Y1}-12-31"))
    coarse_al, Y, times = coarse_al[keep], Y[keep], times[keep].astype("datetime64[ns]")
    cell_lat, cell_lon = grid["cell_lat"], grid["cell_lon"]
    if info["gcm_grid"] and (not np.allclose(info["gcm_grid"]["cell_lat"], cell_lat) or
                             not np.allclose(info["gcm_grid"]["cell_lon"], cell_lon)):
        raise SystemExit("GCM cells differ from the ones recorded in the run")
    print(f"GCM cells: rows (lat) {np.round(cell_lat, 2).tolist()} | cols (lon) {np.round(cell_lon, 2).tolist()}")

    split = info["projection_start"]
    segments = [("historical", Y0, split - 1), (a.scenario, split, Y1)]
    print(f"GCM predictors from {gcm_dir}: " + " + ".join(f"{s} {y0}-{y1}" for s, y0, y1 in segments), flush=True)
    raw = np.full_like(coarse_al, np.nan)
    files_used, filled = {}, {}
    for v, p in enumerate(preds_ + [None]):
        var = gcm_name(target) if p is None else gcm_name(p)
        da, used, missing = gcm_variable(gcm_dir, var, segments, cell_lat, cell_lon)
        files_used[var] = [os.path.relpath(f, gcm_dir) for f in used]
        if p is None:                                            # raw GCM target at the cells, for the plots
            target_gcm = da
            continue
        if da is None:
            filled[p] = f"no GCM {var} for {', '.join(missing)}"
            print(f"  {p}: {filled[p]} -> ALADIN day-of-year climatology {CLIM_YEARS[0]}-{CLIM_YEARS[1]}")
            continue
        da = da.reindex(time=times)
        if np.isnan(da.values).any():
            raise SystemExit(f"{var}: {int(np.isnan(da.values).any(axis=(1, 2)).sum())} days missing after joining")
        raw[:, v] = da.values
        print(f"  {p} <- {var}: {', '.join(files_used[var])}", flush=True)

    clim = doy_climatology(times, coarse_al)
    adj = raw.copy()
    for v, p in enumerate(preds_):
        if p in filled:
            raw[:, v] = adj[:, v] = clim[:, v]
        else:
            adj[:, v] = monthly_scaling(times, raw[:, v], coarse_al[:, v])
    assert np.isfinite(raw).all() and np.isfinite(adj).all()

    variants = {"aladin": coarse_al, "gcm_raw": raw, "gcm_adj": adj}
    if filled:
        al_clim = coarse_al.copy()
        for v, p in enumerate(preds_):
            if p in filled:
                al_clim[:, v] = clim[:, v]
        variants["aladin_climfill"] = al_clim
    for name in a.skip or []:
        variants.pop(name, None)

    device = unet.get_device(a.device)
    print(f"Predicting {', '.join(variants)} with {len(glob.glob(f'{a.run_dir}/fold_*/best.pt'))} folds on {device}...", flush=True)
    preds = predict(a.run_dir, variants, unet.seasonal_features(times), grid, dem, land_mask, device, a.batch)

    os.makedirs(out_nc, exist_ok=True)
    enc = {target: {"zlib": True, "complevel": 4}}
    coords = {"time": times, "lat": lat, "lon": lon}
    for name, p in preds.items():
        xr.DataArray(p, dims=("time", "lat", "lon"), coords=coords, name=target,
                     attrs={"units": "K" if target.startswith("tas") else "",
                            "source": f"U-Net {a.run_dir}, {len(glob.glob(f'{a.run_dir}/fold_*/best.pt'))}-fold ensemble, "
                                      f"input={name}, historical+{a.scenario}"}
                     ).to_netcdf(f"{out_nc}/{target}_unet_{name}_{Y0}-{Y1}.nc", encoding=enc)
    xr.DataArray(Y[:, 0], dims=("time", "lat", "lon"), coords=coords, name=target
                 ).to_netcdf(f"{out_nc}/{target}_aladin_truth_{Y0}-{Y1}.nc", encoding=enc)
    gc = {"time": times, "variable": preds_, "row_lat": cell_lat, "col_lon": cell_lon}
    ds = xr.Dataset({"raw": (("time", "variable", "row_lat", "col_lon"), raw),
                     "adj": (("time", "variable", "row_lat", "col_lon"), adj),
                     "aladin": (("time", "variable", "row_lat", "col_lon"), coarse_al)}, coords=gc)
    if target_gcm is not None:
        ds["target_gcm"] = (("time_t", "row_lat", "col_lon"), target_gcm.values)
        ds = ds.assign_coords(time_t=target_gcm.time.values)
    ds.to_netcdf(f"{out_nc}/predictors_gcm_vs_aladin.nc")
    json.dump({"run": str(a.run_dir), "target": target, "predictors": preds_, "layout": info["layout"],
               "segments": segments, "gcm_dir": str(gcm_dir), "files_used": files_used,
               "replaced_by_climatology": filled, "ref_period_rescaling": REF, "climatology_years": CLIM_YEARS,
               "cell_lat": cell_lat, "cell_lon": cell_lon, "land_pixels": grid["land_pixels"],
               "test_periods": info["test_periods"], "device": str(device), "torch": torch.__version__},
              open(f"{out_nc}/info.json", "w"), indent=2)
    print(f"predictions -> {out_nc}")


# ------------------------------------------------------------------ plots
def stage_plot(a, info, out_nc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as st

    target, preds_ = info["target"], info["predictors"]
    short = gcm_name(target)
    is_temp = target.startswith("tas")
    conv, unit = (K2C, "°C") if is_temp else (0.0, "")
    C_T, C_AL, C_ADJ, C_RAW, C_GCM = "#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#52514e"
    INK2 = "#52514e"
    C_PER = ["#2a78d6", "#1baf7a", "#e34948"]
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(f"{out_nc}/info.json"))
    mask = np.load(f"{a.run_dir}/land_mask.npy")
    o = lambda n: xr.open_dataarray(f"{out_nc}/{target}_{n}_{Y0}-{Y1}.nc").load() - conv
    D = {"ALADIN truth": o("aladin_truth"), "U-Net (ALADIN inputs)": o("unet_aladin"),
         "U-Net (GCM adjusted)": o("unet_gcm_adj"), "U-Net (GCM raw)": o("unet_gcm_raw")}
    has_climfill = os.path.exists(f"{out_nc}/{target}_unet_aladin_climfill_{Y0}-{Y1}.nc")
    climfill = o("unet_aladin_climfill") if has_climfill else None
    COL = dict(zip(D, [C_T, C_AL, C_ADJ, C_RAW]))
    T = D["ALADIN truth"]
    P = {"1985-2014": ("1985", "2014"), "2041-2070": ("2041", "2070"), "2071-2100": ("2071", "2100")}
    land = lambda da: da.values[:, mask]
    sv = lambda da, p: da.sel(time=slice(*P[p]))
    W = lambda x, y: st.wasserstein_distance(np.ravel(x), np.ravel(y))
    pg = xr.open_dataset(f"{out_nc}/predictors_gcm_vs_aladin.nc")
    lon, lat = T.lon.values, T.lat.values
    scen = meta["segments"][-1][0]

    def save(fig, name):
        fig.savefig(f"{out}/{name}.png"); plt.close(fig); print("  ->", name)

    # ---------- metrics table
    rows = []
    yrs_all = pd.DatetimeIndex(T.time.values).year
    test = np.zeros(len(yrs_all), bool)
    for y0, y1 in unet.parse_periods(info["test_periods"]):
        test |= (yrs_all >= y0) & (yrs_all <= y1)
    for p in list(P) + [f"{Y0}-{Y1}"]:
        t = land(T) if p not in P else land(sv(T, p))
        for k in list(D)[1:]:
            x = land(D[k]) if p not in P else land(sv(D[k], p))
            e = x - t
            rows.append(dict(period=p, input=k, mean_bias=e.mean(), W1=W(x, t),
                             ks=st.ks_2samp(x.ravel()[::5], t.ravel()[::5]).statistic,
                             std_ratio=x.std() / t.std(), p99_err=np.percentile(x, 99) - np.percentile(t, 99),
                             p01_err=np.percentile(x, 1) - np.percentile(t, 1),
                             daily_r_islandmean=np.corrcoef(x.mean(1), t.mean(1))[0, 1],
                             rmse_daily=np.sqrt((e ** 2).mean())))
    met = pd.DataFrame(rows); met.to_csv(f"{out}/metrics_gcm.csv", index=False)
    print(met.round(3).to_string(index=False))
    if has_climfill:
        t_test, a_test, c_test = land(T)[test], land(D["U-Net (ALADIN inputs)"])[test], land(climfill)[test]
        sens = dict(rmse_all_inputs=float(np.sqrt(((a_test - t_test) ** 2).mean())),
                    rmse_with_climatology_fill=float(np.sqrt(((c_test - t_test) ** 2).mean())),
                    W1_all_inputs=W(a_test, t_test), W1_with_climatology_fill=W(c_test, t_test),
                    filled=meta["replaced_by_climatology"])
        json.dump(sens, open(f"{out}/sensitivity_missing_predictors.json", "w"), indent=2)
        print("sensitivity (test days, ALADIN inputs):", sens)

    # ---------- 01 predictors: GCM raw / adjusted vs ALADIN upscaled, per variable (mean of the cells)
    units = {"tasminAdjust": "K", "tasmaxAdjust": "K", "hursmaxAdjust": "%", "hursminAdjust": "%",
             "prtotAdjust": "kg m⁻² s⁻¹", "rsdsAdjust": "W m⁻²", "sfcWindAdjust": "m s⁻¹"}
    nv = len(preds_)
    fig, axes = plt.subplots(2, nv, figsize=(3.7 * nv, 6.5), squeeze=False)
    tt = pg.time.values
    for v, var in enumerate(preds_):
        series = {"ALADIN upscaled": (pg["aladin"].isel(variable=v).mean(("row_lat", "col_lon")).values, C_T),
                  "GCM raw": (pg["raw"].isel(variable=v).mean(("row_lat", "col_lon")).values, C_RAW),
                  "GCM adjusted": (pg["adj"].isel(variable=v).mean(("row_lat", "col_lon")).values, C_ADJ)}
        for lab, (vv, c) in series.items():
            s = pd.Series(vv, index=pd.DatetimeIndex(tt))
            r = s[REF[0]:REF[1]]
            axes[0, v].plot(range(1, 13), r.groupby(r.index.month).mean(), "-o", ms=3, color=c, label=lab)
            yr = s.groupby(s.index.year).mean()
            axes[1, v].plot(yr.index, yr.values, color=c, lw=1.3, label=lab)
        tag = " (climatology fill)" if var in meta["replaced_by_climatology"] else ""
        axes[0, v].set_title(f"{var}{tag}"); axes[0, v].set_xticks(range(1, 13)); axes[0, v].set_xticklabels("JFMAMJJASOND")
        axes[0, v].set_ylabel(units.get(var, "")); axes[1, v].set_ylabel(f"annual mean ({units.get(var, '')})")
        axes[1, v].axvline(meta["segments"][-1][1] - 0.5, color=INK2, lw=0.8, ls=":")
    axes[0, 0].legend()
    axes[0, 0].text(0.0, 1.13, f"Monthly climatology {REF[0]}-{REF[1]} (mean of the {pg.row_lat.size * pg.col_lon.size} GCM cells)",
                    transform=axes[0, 0].transAxes, fontsize=10)
    axes[1, 0].text(0.0, 1.05, f"Annual means {Y0}-{Y1} (dotted: historical | {scen})", transform=axes[1, 0].transAxes, fontsize=10)
    save(fig, "01_predictors_gcm_vs_aladin")

    # ---------- 02 annual island-mean target
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, da in D.items():
        yr = da.mean(("lat", "lon")).groupby("time.year").mean()
        ax.plot(yr.year, yr, color=COL[k], lw=2 if k == "ALADIN truth" else 1.5, label=k)
    txg = None
    if "target_gcm" in pg:
        txg = pg["target_gcm"].mean(("row_lat", "col_lon")) - conv
        yr = txg.groupby("time_t.year").mean()
        ax.plot(yr.year, yr, color=C_GCM, lw=1.2, ls="--", label=f"GCM {short}, raw (cells incl. ocean)")
    for y0, y1 in unet.parse_periods(info["test_periods"]):
        ax.axvspan(y0 - 0.5, y1 + 0.5, color="#e4e3df", alpha=0.6, lw=0)
    ax.set_ylabel(f"annual mean {short} over land ({unit})"); ax.legend(ncol=2, loc="upper left")
    ax.set_title(f"Annual island-mean {short}, {Y0}-{Y1}, historical + {scen} (grey bands: U-Net test periods)", loc="left")
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
        ax.set_title(p); ax.set_xlabel(f"{short} ({unit})"); ax.legend(fontsize=7.5, loc="upper left")
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
        ax.set_title(p); ax.set_xlabel(f"ALADIN quantile ({unit})"); ax.set_aspect("equal")
    axes[0].set_ylabel(f"U-Net quantile ({unit})"); axes[0].legend(fontsize=8)
    save(fig, "04_qq_30yr_periods")

    # ---------- 05 maps: climatology, bias and per-pixel W1 (GCM adjusted)
    def pp(fn, x, y):
        A, B = x.values, y.values; r = np.full(mask.shape, np.nan)
        for i, j in zip(*np.where(mask)):
            r[i, j] = fn(A[:, i, j], B[:, i, j])
        return r
    fig, axes = plt.subplots(4, 3, figsize=(12, 13))
    k = "U-Net (GCM adjusted)"
    mt = {p: sv(T, p).mean("time") for p in P}; mp = {p: sv(D[k], p).mean("time") for p in P}
    vmin, vmax = min(float(m.min()) for m in mt.values()), max(float(m.max()) for m in mt.values())
    bias = {p: (mp[p] - mt[p]).values for p in P}; bb = max(np.nanmax(np.abs(b)) for b in bias.values())
    w1 = {p: pp(W, sv(D[k], p), sv(T, p)) for p in P}; wm = max(np.nanmax(w) for w in w1.values())
    rows_ = [("ALADIN mean", mt, "YlOrRd", vmin, vmax, unit), (f"{k} mean", mp, "YlOrRd", vmin, vmax, unit),
             ("bias (U-Net − ALADIN)", bias, "RdBu_r", -bb, bb, unit), ("per-pixel W1", w1, "Blues", 0, wm, unit)]
    for r, (title, dd, cmap, lo_, hi_, u) in enumerate(rows_):
        for c, p in enumerate(P):
            im = axes[r, c].pcolormesh(lon, lat, np.asarray(dd[p]), cmap=cmap, vmin=lo_, vmax=hi_, shading="nearest")
            axes[r, c].set_title(f"{title}\n{p} (land mean {np.nanmean(np.asarray(dd[p])):.2f})", fontsize=8.5)
            axes[r, c].set_aspect("equal"); axes[r, c].grid(False); axes[r, c].tick_params(labelsize=7)
        fig.colorbar(im, ax=axes[r, :], shrink=0.85, label=u)
    save(fig, "05_maps_gcm_adjusted")

    # ---------- 06 climate change signal maps and table
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    sig = []
    for r, p in enumerate(["2041-2070", "2071-2100"]):
        dd = {kk: (sv(D[kk], p).mean("time") - sv(D[kk], "1985-2014").mean("time")) for kk in D}
        lo_, hi_ = min(float(v.min()) for v in dd.values()), max(float(v.max()) for v in dd.values())
        for c, (kk, v) in enumerate(dd.items()):
            im = axes[r, c].pcolormesh(lon, lat, v, cmap="Reds", vmin=lo_, vmax=hi_, shading="nearest")
            axes[r, c].set_title(f"{kk}\nΔ {p} (mean {float(v.mean()):.2f})", fontsize=8.5)
            axes[r, c].set_aspect("equal"); axes[r, c].grid(False); axes[r, c].tick_params(labelsize=7)
            sig.append(dict(period=p, source=kk, delta_mean=float(v.mean())))
        fig.colorbar(im, ax=axes[r, :], shrink=0.85, label=f"change vs 1985-2014 ({unit})")
    if txg is not None:
        dg = txg.sel(time_t=slice("1985", "2014")).mean()
        for p in ["2041-2070", "2071-2100"]:
            sig.append(dict(period=p, source=f"GCM {short} raw", delta_mean=float(txg.sel(time_t=slice(*P[p])).mean() - dg)))
    save(fig, "06_change_signal_maps")
    sig = pd.DataFrame(sig); sig.to_csv(f"{out}/change_signal.csv", index=False); print(sig.round(3).to_string(index=False))

    # ---------- 07 seasonal cycle
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, p in zip(axes, P):
        for kk, da in D.items():
            m = sv(da, p).mean(("lat", "lon")).groupby("time.month").mean()
            ax.plot(m.month, m, "-o", ms=3, color=COL[kk], lw=2 if kk == "ALADIN truth" else 1.4, label=kk)
        ax.set_xticks(range(1, 13)); ax.set_xticklabels("JFMAMJJASOND"); ax.set_title(p)
    axes[0].set_ylabel(f"island-mean {short} ({unit})"); axes[0].legend(fontsize=8)
    save(fig, "07_seasonal_cycle")

    # ---------- 08 extremes
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    thr = np.percentile(land(sv(T, "1985-2014")), 90)
    for kk, da in D.items():
        e = pd.Series((land(da) > thr).mean(1), index=yrs_all).groupby(level=0).mean() * 365.25
        axes[0].plot(e.index, e.values, color=COL[kk], lw=2 if kk == "ALADIN truth" else 1.4, label=kk)
    axes[0].set_ylabel("days/yr per land pixel")
    axes[0].set_title(f"Days > {thr:.1f} {unit} (P90 of ALADIN 1985-2014)", loc="left"); axes[0].legend(fontsize=8)
    pcs = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    for p, c in zip(P, C_PER):
        t = land(sv(T, p))
        for kk, ls in [("U-Net (GCM adjusted)", "-"), ("U-Net (ALADIN inputs)", ":")]:
            axes[1].plot(range(len(pcs)), np.percentile(land(sv(D[kk], p)), pcs) - np.percentile(t, pcs), ls, marker="o",
                         ms=4, color=c, label=f"{p} {kk.split('(')[1][:-1]}")
    axes[1].axhline(0, color="k", lw=0.8); axes[1].set_xticks(range(len(pcs))); axes[1].set_xticklabels([f"P{x}" for x in pcs])
    axes[1].set_ylabel(f"U-Net − ALADIN percentile ({unit})"); axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_title("Percentile errors (solid: GCM adjusted, dotted: ALADIN inputs)", loc="left")
    save(fig, "08_extremes")

    # ---------- 09 example days: GCM coarse field vs downscaled vs ALADIN
    gcm_dir = meta["gcm_dir"]
    files, _ = find_files(gcm_dir, short, scen, 2050, 2090)
    if files:
        days = ["2050-02-15", "2090-07-15"]
        gs = []
        for d in days:
            f = next(f for f in files if file_years(f)[0] <= int(d[:4]) <= file_years(f)[1])
            with xr.open_dataset(f) as ds:
                gs.append(ds[short].sel(lat=slice(-24, -18), lon=slice(52.5, 59))
                          .sel(time=np.datetime64(d + "T12:00"), method="nearest").load() - conv)
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
        for r, d in enumerate(days):
            fields = [(f"GCM {short} (raw, 1.4°)", gs[r])] + \
                     [(kk, D[kk].sel(time=d)) for kk in ["ALADIN truth", "U-Net (GCM adjusted)", "U-Net (ALADIN inputs)"]]
            vals = np.concatenate([np.ravel(f.values)[np.isfinite(np.ravel(f.values))] for _, f in fields[1:]])
            lo_, hi_ = np.percentile(vals, 1), np.percentile(vals, 99)
            for c, (title, f) in enumerate(fields):
                im = axes[r, c].pcolormesh(f.lon, f.lat, f, cmap="YlOrRd", vmin=lo_, vmax=hi_, shading="nearest")
                if c == 0:
                    axes[r, c].add_patch(plt.Rectangle((lon.min(), lat.min()), np.ptp(lon), np.ptp(lat), fill=False, ec="k", lw=1))
                axes[r, c].set_title(f"{title}\n{d}", fontsize=8.5); axes[r, c].set_aspect("equal"); axes[r, c].grid(False)
            fig.colorbar(im, ax=axes[r, :], shrink=0.85, label=f"{short} ({unit})")
        save(fig, "09_example_days")

    # ---------- 10 W1 over time (5-yr windows) per input
    fig, ax = plt.subplots(figsize=(12, 4))
    for kk in list(D)[1:]:
        ws = []
        for y in range(Y0, Y1, 5):
            sl = slice(str(y), str(y + 4))
            ws.append((y + 2, W(land(D[kk].sel(time=sl)), land(T.sel(time=sl)))))
        ws = np.array(ws); ax.plot(ws[:, 0], ws[:, 1], "-o", ms=4, color=COL[kk], label=kk)
    ax.axvline(meta["segments"][-1][1] - 0.5, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel(f"Wasserstein-1 vs ALADIN ({unit})")
    ax.set_title(f"Distributional distance in 5-year windows (dotted: historical | {scen})", loc="left"); ax.legend()
    save(fig, "10_wasserstein_over_time")


def main():
    ap = argparse.ArgumentParser(description="Apply a UNET.py run to CNRM-ESM2-1 GCM predictors (historical + scenario joined)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--stage", default="all", choices=["all", "predict", "plot"])
    ap.add_argument("--scenario", default="ssp370", help="GCM projection joined after historical")
    ap.add_argument("--data-dir", default=os.environ.get("UNET_DATA_DIR", unet.DEFAULT_DATA_DIR))
    ap.add_argument("--gcm-dir", default=None, help="default: <data-dir>/CMIP_CNRM_ESM21")
    ap.add_argument("--gcm-grid-file", default=None, help="GCM file giving the grid (default: as UNET.py)")
    ap.add_argument("--cache-dir", default=os.environ.get("UNET_CACHE_DIR", unet.DEFAULT_CACHE_DIR),
                    help="UNET.py cache of the ALADIN arrays, '' = off")
    ap.add_argument("--device", default=os.environ.get("UNET_DEVICE", "auto"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--skip", nargs="+", choices=["aladin", "gcm_raw", "gcm_adj", "aladin_climfill"],
                    help="inputs not to predict (saves time)")
    ap.add_argument("--out-nc", default=None, help="default: outputs/<run>_gcm")
    ap.add_argument("--out-plots", default=None, help="default: plots/<run>_gcm")
    a = ap.parse_args()
    run = os.path.basename(os.path.normpath(a.run_dir))
    suffix = "" if a.scenario == "ssp370" else f"_{a.scenario}"
    out_nc = a.out_nc or os.path.join("outputs", f"{run}_gcm{suffix}")
    out = a.out_plots or os.path.join("plots", f"{run}_gcm{suffix}")
    info = run_info(a.run_dir)
    print(f"run {run}: target {info['target']} | predictors {', '.join(info['predictors'])} | layout {info['layout']}")
    if a.stage in ("all", "predict"):
        stage_predict(a, info, out_nc)
    if a.stage in ("all", "plot"):
        stage_plot(a, info, out_nc, out)


if __name__ == "__main__":
    main()
