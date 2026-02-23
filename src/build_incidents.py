from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA_PROCESSED = PROJ / "data" / "processed"
OUT_DIR = PROJ / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY_TAG = "2020_04_11"
METRICS_PATH = DATA_PROCESSED / f"day_{DAY_TAG}_metrics.parquet"
FAIL_PATH = DATA_PROCESSED / "failures_clean.csv"


def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)


def points_to_windows(t: pd.Series, is_anom: pd.Series, gap_min=2) -> pd.DataFrame:
    d = pd.DataFrame({
        "t": pd.to_datetime(pd.Series(t)).reset_index(drop=True),
        "a": pd.Series(is_anom).reset_index(drop=True).astype(int),
    })
    d = d[d["a"] == 1].sort_values("t")
    if d.empty:
        return pd.DataFrame(columns=["win_start", "win_end", "n_points"])

    windows = []
    start = d.iloc[0]["t"]
    prev = start
    count = 1

    for tt in d.iloc[1:]["t"]:
        if (tt - prev) <= pd.Timedelta(minutes=gap_min):
            prev = tt
            count += 1
        else:
            windows.append((start, prev, count))
            start = tt
            prev = tt
            count = 1

    windows.append((start, prev, count))
    out = pd.DataFrame(windows, columns=["win_start", "win_end", "n_points"])
    # widen slightly
    out["win_start"] = out["win_start"] - pd.Timedelta(minutes=1)
    out["win_end"] = out["win_end"] + pd.Timedelta(minutes=1)
    return out


def merge_windows(windows: pd.DataFrame, gap_min=3) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame(columns=["incident_start", "incident_end"])

    w = windows.sort_values("win_start").reset_index(drop=True)
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


def main():
    metrics = pd.read_parquet(METRICS_PATH)
    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])

    metrics["entity"] = metrics["entity"].astype(str).str.strip()
    metrics["kpi"] = metrics["kpi"].astype(str).str.strip()

    # Align timezone
    metrics["timestamp"] = metrics["timestamp"] + pd.Timedelta(hours=8)

    day_min = metrics["timestamp"].min()
    day_max = metrics["timestamp"].max()
    buf = pd.Timedelta(minutes=10)

    day_fails = fails[
        (fails["event_start"] >= day_min - buf) &
        (fails["event_start"] <= day_max + buf) &
        (fails["name"].notna()) &
        (fails["kpi_one"].notna())
    ].copy()

    print("Day window:", day_min, "->", day_max)
    print("Failures in window:", len(day_fails))

    # We'll compute anomaly windows for all (entity,kpi) pairs that appear in failures
    pairs = (
        day_fails[["name", "kpi_one"]]
        .drop_duplicates()
        .rename(columns={"name": "entity", "kpi_one": "kpi"})
    )
    pairs["entity"] = pairs["entity"].astype(str).str.strip()
    pairs["kpi"] = pairs["kpi"].astype(str).str.strip()

    # params
    z_thresh = 3.5
    win = 15
    min_p = 5

    # Store per-signal windows and peaks
    signal_windows = []  # (entity,kpi,win_start,win_end,n_points)
    signal_peaks = []    # (entity,kpi,incident_time_range,peak_abs_z)

    # Pre-group metric rows by (entity,kpi) for speed
    metrics_grp = metrics.groupby(["entity", "kpi"], sort=False)

    for _, row in pairs.iterrows():
        entity, kpi = row["entity"], row["kpi"]
        if (entity, kpi) not in metrics_grp.groups:
            continue

        m = metrics_grp.get_group((entity, kpi)).sort_values("timestamp")
        s = (
            m.set_index("timestamp")["value"]
            .resample("1min").mean()
            .interpolate(limit=3)
            .reset_index()
            .rename(columns={"timestamp": "t", "value": "v"})
        )

        z = rolling_mad_zscore(s["v"], window=win, min_p=min_p)
        is_anom = (z.abs() >= z_thresh).astype(int)

        wdf = points_to_windows(s["t"], is_anom, gap_min=2)
        if not wdf.empty:
            wdf2 = wdf.copy()
            wdf2["entity"] = entity
            wdf2["kpi"] = kpi
            signal_windows.append(wdf2)

    if not signal_windows:
        print("No anomaly windows found for any failure-linked signals.")
        return

    signal_windows_df = pd.concat(signal_windows, ignore_index=True)
    signal_windows_df.to_csv(OUT_DIR / f"signal_windows_{DAY_TAG}.csv", index=False, encoding="utf-8")

    # Build incidents per entity by merging all signal windows for that entity
    incidents_all = []
    contrib_all = []

    for entity, ew in signal_windows_df.groupby("entity"):
        merged = merge_windows(ew[["win_start", "win_end"]].drop_duplicates(), gap_min=3)

        # For each incident, compute top contributing KPIs by peak z within incident
        for idx, inc in merged.iterrows():
            inc_s, inc_e = inc["incident_start"], inc["incident_end"]

            # contributing windows (which KPIs fired)
            fired = ew[(ew["win_start"] <= inc_e) & (ew["win_end"] >= inc_s)][["kpi"]].drop_duplicates()
            fired_kpis = fired["kpi"].tolist()

            # compute peak z per kpi within incident
            top = []
            for kpi in fired_kpis:
                # re-load series (only for those kpis) to compute peak quickly
                # (fine for your scale; you can cache later)
                m = metrics_grp.get_group((entity, kpi)).sort_values("timestamp")
                s = (
                    m.set_index("timestamp")["value"]
                    .resample("1min").mean()
                    .interpolate(limit=3)
                    .reset_index()
                    .rename(columns={"timestamp": "t", "value": "v"})
                )
                z = rolling_mad_zscore(s["v"], window=win, min_p=min_p)
                mask = (s["t"] >= inc_s) & (s["t"] <= inc_e)
                peak = float(z[mask].abs().max()) if mask.any() else 0.0
                top.append((kpi, peak))

            top = sorted(top, key=lambda x: x[1], reverse=True)[:5]
            top_str = "; ".join([f"{k}:{p:.2f}" for k, p in top])

            incident_id = f"{entity}_{idx+1}"
            incidents_all.append({
                "incident_id": incident_id,
                "entity": entity,
                "incident_start": inc_s,
                "incident_end": inc_e,
                "duration_min": (inc_e - inc_s).total_seconds() / 60.0,
                "num_signals": len(fired_kpis),
                "top_signals_peakz": top_str
            })

            for k, p in top:
                contrib_all.append({
                    "incident_id": incident_id,
                    "entity": entity,
                    "kpi": k,
                    "peak_abs_z": p,
                })

    incidents_df = pd.DataFrame(incidents_all).sort_values(["incident_start", "entity"])
    incidents_path = OUT_DIR / f"incidents_{DAY_TAG}.csv"
    incidents_df.to_csv(incidents_path, index=False, encoding="utf-8")

    contrib_df = pd.DataFrame(contrib_all)
    contrib_df.to_csv(OUT_DIR / f"incident_contrib_{DAY_TAG}.csv", index=False, encoding="utf-8")

    # Map failures to incidents (detected at incident level)
    map_rows = []
    for _, fr in day_fails.iterrows():
        entity = str(fr["name"]).strip()
        ev_s, ev_e = fr["event_start"], fr["event_end"]

        incs = incidents_df[incidents_df["entity"] == entity]
        hit = False
        first_inc = None
        for _, inc in incs.iterrows():
            if overlap(inc["incident_start"], inc["incident_end"], ev_s, ev_e):
                hit = True
                if first_inc is None or inc["incident_start"] < first_inc:
                    first_inc = inc["incident_start"]

        delay = None
        if hit and first_inc is not None:
            delay = (first_inc - ev_s).total_seconds() / 60.0

        map_rows.append({
            "failure_index": fr.get("index"),
            "entity": entity,
            "kpi": fr.get("kpi_one"),
            "fault_description": fr.get("fault_description"),
            "event_start": ev_s,
            "event_end": ev_e,
            "incident_detected": hit,
            "incident_start": first_inc,
            "incident_delay_min": delay,
        })

    failure_map = pd.DataFrame(map_rows)
    fmap_path = OUT_DIR / f"failure_vs_incidents_{DAY_TAG}.csv"
    failure_map.to_csv(fmap_path, index=False, encoding="utf-8")

    # Print quick summary
    total = len(failure_map)
    detected = int(failure_map["incident_detected"].sum())
    delays = failure_map.loc[failure_map["incident_detected"] & failure_map["incident_delay_min"].notna(), "incident_delay_min"]

    print("\n=== Incident-level Summary ===")
    print("Failures:", total)
    print("Detected (incident-level):", detected)
    print("Detection rate:", round(detected / total, 3))
    if len(delays):
        print("Avg delay (min):", round(float(delays.mean()), 2))
        print("Median delay (min):", round(float(delays.median()), 2))

    print("\nSaved:", incidents_path)
    print("Saved:", fmap_path)
    print("Saved:", OUT_DIR / f"incident_contrib_{DAY_TAG}.csv")


if __name__ == "__main__":
    main()
