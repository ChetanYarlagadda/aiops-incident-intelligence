from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA_PROCESSED = PROJ / "data" / "processed"
OUT_DIR = PROJ / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY_TAG = "2020_04_11"
METRICS_PATH = DATA_PROCESSED / f"day_{DAY_TAG}_metrics.parquet"
TRACE_PATH   = DATA_PROCESSED / f"day_{DAY_TAG}_trace_kpis.parquet"
FAIL_PATH    = DATA_PROCESSED / "failures_clean.csv"


def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)


def points_to_windows(t: pd.Series, is_anom: pd.Series, z_abs: pd.Series, gap_min=2) -> pd.DataFrame:
    d = pd.DataFrame({
        "t": pd.to_datetime(pd.Series(t)).reset_index(drop=True),
        "a": pd.Series(is_anom).reset_index(drop=True).astype(int),
        "z": pd.Series(z_abs).reset_index(drop=True).astype(float),
    })
    d = d[d["a"] == 1].sort_values("t")
    if d.empty:
        return pd.DataFrame(columns=["win_start", "win_end", "n_points", "peak_abs_z"])

    windows = []
    start = d.iloc[0]["t"]
    prev = start
    count = 1
    peak = float(d.iloc[0]["z"])

    for _, row in d.iloc[1:].iterrows():
        tt = row["t"]
        zz = float(row["z"])
        if (tt - prev) <= pd.Timedelta(minutes=gap_min):
            prev = tt
            count += 1
            peak = max(peak, zz)
        else:
            windows.append((start, prev, count, peak))
            start = tt
            prev = tt
            count = 1
            peak = zz

    windows.append((start, prev, count, peak))
    out = pd.DataFrame(windows, columns=["win_start", "win_end", "n_points", "peak_abs_z"])
    out["win_start"] = out["win_start"] - pd.Timedelta(minutes=1)
    out["win_end"] = out["win_end"] + pd.Timedelta(minutes=1)
    return out


def merge_windows(w: pd.DataFrame, gap_min=3) -> pd.DataFrame:
    if w.empty:
        return pd.DataFrame(columns=["incident_start", "incident_end"])

    w = w.sort_values("win_start").reset_index(drop=True)
    merged = []
    cur_s = w.loc[0, "win_start"]
    cur_e = w.loc[0, "win_end"]

    for i in range(1, len(w)):
        s = w.loc[i, "win_start"]
        e = w.loc[i, "win_end"]
        if s <= cur_e + pd.Timedelta(minutes=gap_min):
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return pd.DataFrame(merged, columns=["incident_start", "incident_end"])


def overlap(a_start, a_end, b_start, b_end) -> bool:
    return (a_start <= b_end) and (b_start <= a_end)


def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{ts}" + path.suffix)
        df.to_csv(alt, index=False, encoding="utf-8")
        return alt


def main():
    metrics = pd.read_parquet(METRICS_PATH)
    traces  = pd.read_parquet(TRACE_PATH)
    fails   = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])

    # Clean + align timezone
    metrics["entity"] = metrics["entity"].astype(str).str.strip()
    metrics["kpi"] = metrics["kpi"].astype(str).str.strip()
    metrics["timestamp"] = metrics["timestamp"] + pd.Timedelta(hours=8)

    traces["entity"] = traces["entity"].astype(str).str.strip()
    traces["kpi"] = traces["kpi"].astype(str).str.strip()
    traces["timestamp"] = pd.to_datetime(traces["timestamp"])  # already +8 in build script

    allm = pd.concat([metrics[["timestamp", "entity", "kpi", "value", "source"]],
                      traces[["timestamp", "entity", "kpi", "value", "source"]]], ignore_index=True)

    day_min = allm["timestamp"].min().floor("min")
    day_max = allm["timestamp"].max().floor("min")
    full_index = pd.date_range(day_min, day_max, freq="1min")

    buf = pd.Timedelta(minutes=10)
    day_fails = fails[
        (fails["event_start"] >= day_min - buf) &
        (fails["event_start"] <= day_max + buf) &
        (fails["name"].notna())
    ].copy()
    day_fails["name"] = day_fails["name"].astype(str).str.strip()

    print("Day window:", day_min, "->", day_max)
    print("Failures in window:", len(day_fails))
    print("Total rows (metrics+traces):", len(allm))

    # aggregate to 1-min mean
    allm["t"] = allm["timestamp"].dt.floor("min")
    agg = allm.groupby(["entity", "kpi", "t"], as_index=False)["value"].mean()

    # params
    z_thresh = 3.5
    win = 15
    min_p = 5

    # anomaly windows for all entity-kpi
    windows_rows = []
    grouped = agg.groupby(["entity", "kpi"], sort=False)
    print("Total (entity,kpi) series:", len(grouped))

    for (entity, kpi), g in grouped:
        s = g.set_index("t")["value"].reindex(full_index).interpolate(limit=3)
        z = rolling_mad_zscore(s, window=win, min_p=min_p).abs()
        is_anom = (z >= z_thresh).astype(int)

        wdf = points_to_windows(full_index, is_anom, z, gap_min=2)
        if wdf.empty:
            continue
        wdf["entity"] = entity
        wdf["kpi"] = kpi
        windows_rows.append(wdf)

    if not windows_rows:
        print("No anomaly windows found.")
        return

    signal_windows = pd.concat(windows_rows, ignore_index=True)
    signal_windows = signal_windows[["entity", "kpi", "win_start", "win_end", "n_points", "peak_abs_z"]]
    safe_write_csv(signal_windows, OUT_DIR / f"signal_windows_metrics_traces_{DAY_TAG}.csv")

    # incidents per entity + top signals
    incidents = []
    for entity, ew in signal_windows.groupby("entity"):
        merged = merge_windows(ew[["win_start", "win_end"]], gap_min=3)
        for idx, inc in merged.iterrows():
            inc_s, inc_e = inc["incident_start"], inc["incident_end"]
            fired = ew[(ew["win_start"] <= inc_e) & (ew["win_end"] >= inc_s)].copy()
            top = (fired.groupby("kpi", as_index=False)["peak_abs_z"]
                        .max().sort_values("peak_abs_z", ascending=False).head(5))
            top_str = "; ".join([f"{r.kpi}:{r.peak_abs_z:.2f}" for r in top.itertuples(index=False)])
            incidents.append({
                "incident_id": f"{entity}_{idx+1}",
                "entity": entity,
                "incident_start": inc_s,
                "incident_end": inc_e,
                "duration_min": (inc_e - inc_s).total_seconds() / 60.0,
                "num_signals": int(fired["kpi"].nunique()),
                "top_signals_peakz": top_str,
            })

    incidents_df = pd.DataFrame(incidents).sort_values(["incident_start", "entity"])
    safe_write_csv(incidents_df, OUT_DIR / f"incidents_metrics_traces_{DAY_TAG}.csv")

    # map failures -> incidents for entity=name
    map_rows = []
    for fr in day_fails.itertuples(index=False):
        entity = fr.name
        ev_s, ev_e = fr.event_start, fr.event_end

        incs = incidents_df[incidents_df["entity"] == entity]
        hit = False
        first_inc = None
        for inc in incs.itertuples(index=False):
            if overlap(inc.incident_start, inc.incident_end, ev_s, ev_e):
                hit = True
                if first_inc is None or inc.incident_start < first_inc:
                    first_inc = inc.incident_start

        delay = None
        lead = None
        if hit and first_inc is not None:
            raw = (first_inc - ev_s).total_seconds() / 60.0
            if raw < 0:
                lead = abs(raw)
                delay = 0.0
            else:
                delay = raw
                lead = 0.0

        map_rows.append({
            "failure_index": getattr(fr, "index", None),
            "fault_description": getattr(fr, "fault_description", None),
            "name": entity,
            "event_start": ev_s,
            "event_end": ev_e,
            "incident_detected": hit,
            "incident_start": first_inc,
            "delay_min": delay,
            "lead_min": lead,
        })

    fmap = pd.DataFrame(map_rows)
    safe_write_csv(fmap, OUT_DIR / f"failure_vs_incidents_metrics_traces_{DAY_TAG}.csv")

    # summary
    total = len(fmap)
    detected = int(fmap["incident_detected"].sum())
    print("\n=== Summary (Metrics + Traces) ===")
    print("Failures:", total)
    print("Detected:", detected)
    print("Detection rate:", round(detected / max(total, 1), 3))

    if "fault_description" in fmap.columns:
        by_type = fmap.groupby("fault_description")["incident_detected"].agg(["count","sum"])
        by_type["rate"] = (by_type["sum"]/by_type["count"]).round(3)
        print("\n=== Detection by fault_description ===")
        print(by_type.rename(columns={"count":"failures","sum":"detected"}).sort_values("rate", ascending=False).to_string())


if __name__ == "__main__":
    main()
