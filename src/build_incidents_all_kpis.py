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
FAIL_PATH = DATA_PROCESSED / "failures_clean.csv"


# ----------------- utilities -----------------

def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    """Robust z-score using rolling median and MAD."""
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)


def points_to_windows(t: pd.Series, is_anom: pd.Series, z_abs: pd.Series, gap_min=2) -> pd.DataFrame:
    """
    Convert anomaly points to windows.
    Also compute peak_abs_z inside each window.
    """
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
    # widen a bit
    out["win_start"] = out["win_start"] - pd.Timedelta(minutes=1)
    out["win_end"] = out["win_end"] + pd.Timedelta(minutes=1)
    return out


def merge_windows(w: pd.DataFrame, gap_min=3) -> pd.DataFrame:
    """Merge overlapping/nearby windows into incidents."""
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


# ----------------- main -----------------

def main():
    metrics = pd.read_parquet(METRICS_PATH)
    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])

    metrics["entity"] = metrics["entity"].astype(str).str.strip()
    metrics["kpi"] = metrics["kpi"].astype(str).str.strip()

    # Align timezone (+8h, consistent with your pipeline)
    metrics["timestamp"] = metrics["timestamp"] + pd.Timedelta(hours=8)

    # Define day window from metrics
    day_min = metrics["timestamp"].min().floor("min")
    day_max = metrics["timestamp"].max().floor("min")
    full_index = pd.date_range(day_min, day_max, freq="1min")

    # Failures restricted to this day window (+buffer)
    buf = pd.Timedelta(minutes=10)
    day_fails = fails[
        (fails["event_start"] >= day_min - buf) &
        (fails["event_start"] <= day_max + buf) &
        (fails["name"].notna())
    ].copy()
    day_fails["name"] = day_fails["name"].astype(str).str.strip()

    print("Day window:", day_min, "->", day_max)
    print("Failures in window:", len(day_fails))

    # Aggregate metrics to 1-min mean to avoid repeated resampling cost
    metrics["t"] = metrics["timestamp"].dt.floor("min")
    agg = metrics.groupby(["entity", "kpi", "t"], as_index=False)["value"].mean()

    # Params
    z_thresh = 3.5
    win = 15
    min_p = 5

    # 1) Build anomaly windows for ALL entity-kpi series
    windows_rows = []

    grouped = agg.groupby(["entity", "kpi"], sort=False)
    total_series = len(grouped)
    print("Total (entity,kpi) series:", total_series)

    for (entity, kpi), g in grouped:
        # build aligned 1-min series across full day
        s = (
            g.set_index("t")["value"]
            .reindex(full_index)
            .interpolate(limit=3)
        )

        z = rolling_mad_zscore(s, window=win, min_p=min_p).abs()
        is_anom = (z >= z_thresh).astype(int)

        wdf = points_to_windows(full_index, is_anom, z, gap_min=2)
        if wdf.empty:
            continue

        wdf["entity"] = entity
        wdf["kpi"] = kpi
        windows_rows.append(wdf)

    if not windows_rows:
        print("No anomaly windows found in entire day.")
        return

    signal_windows = pd.concat(windows_rows, ignore_index=True)
    signal_windows = signal_windows[["entity", "kpi", "win_start", "win_end", "n_points", "peak_abs_z"]]
    signal_path = OUT_DIR / f"signal_windows_allkpis_{DAY_TAG}.csv"
    safe_write_csv(signal_windows, signal_path)

    # 2) Merge windows into incidents per entity + add top signals
    incidents = []
    contrib = []

    for entity, ew in signal_windows.groupby("entity"):
        merged = merge_windows(ew[["win_start", "win_end"]], gap_min=3)

        for idx, inc in merged.iterrows():
            inc_s, inc_e = inc["incident_start"], inc["incident_end"]

            # KPIs that fired in this incident and their max peak z
            fired = ew[(ew["win_start"] <= inc_e) & (ew["win_end"] >= inc_s)].copy()
            top = (
                fired.groupby("kpi", as_index=False)["peak_abs_z"]
                .max()
                .sort_values("peak_abs_z", ascending=False)
                .head(5)
            )
            top_str = "; ".join([f"{r.kpi}:{r.peak_abs_z:.2f}" for r in top.itertuples(index=False)])

            incident_id = f"{entity}_{idx+1}"
            incidents.append({
                "incident_id": incident_id,
                "entity": entity,
                "incident_start": inc_s,
                "incident_end": inc_e,
                "duration_min": (inc_e - inc_s).total_seconds() / 60.0,
                "num_signals": int(fired["kpi"].nunique()),
                "top_signals_peakz": top_str,
            })

            for r in top.itertuples(index=False):
                contrib.append({
                    "incident_id": incident_id,
                    "entity": entity,
                    "kpi": r.kpi,
                    "peak_abs_z": float(r.peak_abs_z),
                })

    incidents_df = pd.DataFrame(incidents).sort_values(["incident_start", "entity"])
    incidents_path = OUT_DIR / f"incidents_allkpis_{DAY_TAG}.csv"
    safe_write_csv(incidents_df, incidents_path)

    contrib_df = pd.DataFrame(contrib)
    contrib_path = OUT_DIR / f"incident_contrib_allkpis_{DAY_TAG}.csv"
    safe_write_csv(contrib_df, contrib_path)

    # 3) Map failures -> incidents (entity-wise)
    map_rows = []
    for fr in day_fails.itertuples(index=False):
        entity = fr.name
        ev_s = fr.event_start
        ev_e = fr.event_end

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
            # if incident starts before the failure window, that's "lead time"
            if raw < 0:
                lead = abs(raw)
                delay = 0.0
            else:
                delay = raw
                lead = 0.0

        map_rows.append({
            "failure_index": getattr(fr, "index", None),
            "entity": entity,
            "fault_description": getattr(fr, "fault_description", None),
            "event_start": ev_s,
            "event_end": ev_e,
            "incident_detected": hit,
            "incident_start": first_inc,
            "delay_min": delay,
            "lead_min": lead,
        })

    fmap = pd.DataFrame(map_rows)
    fmap_path = OUT_DIR / f"failure_vs_incidents_allkpis_{DAY_TAG}.csv"
    safe_write_csv(fmap, fmap_path)

    # Summary
    total = len(fmap)
    detected = int(fmap["incident_detected"].sum())
    delays = fmap.loc[fmap["incident_detected"] & fmap["delay_min"].notna(), "delay_min"]
    leads = fmap.loc[fmap["incident_detected"] & fmap["lead_min"].notna(), "lead_min"]

    print("\n=== Incident-level Summary (ALL KPIs) ===")
    print("Failures:", total)
    print("Detected:", detected)
    print("Detection rate:", round(detected / max(total, 1), 3))
    if len(delays):
        print("Avg delay (min):", round(float(delays.mean()), 2), " Median:", round(float(delays.median()), 2))
    if len(leads):
        print("Avg lead (min):", round(float(leads.mean()), 2), " Median:", round(float(leads.median()), 2))

    print("\nSaved:", signal_path)
    print("Saved:", incidents_path)
    print("Saved:", contrib_path)
    print("Saved:", fmap_path)


if __name__ == "__main__":
    main()
