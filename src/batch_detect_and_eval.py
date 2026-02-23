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


def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)


def points_to_windows(ts: pd.Series, is_anom: pd.Series, gap_min=2) -> pd.DataFrame:
    """
    Convert anomaly points into continuous windows.
    IMPORTANT: reset_index() so ts and is_anom align by row (not by index labels).
    """
    d = pd.DataFrame({
        "t": pd.to_datetime(pd.Series(ts)).reset_index(drop=True),
        "a": pd.Series(is_anom).reset_index(drop=True).astype(int),
    })
    d = d[d["a"] == 1].sort_values("t")
    if d.empty:
        return pd.DataFrame(columns=["win_start", "win_end", "n_points"])

    windows = []
    start = d.iloc[0]["t"]
    prev = start
    count = 1

    for t in d.iloc[1:]["t"]:
        if (t - prev) <= pd.Timedelta(minutes=gap_min):
            prev = t
            count += 1
        else:
            windows.append((start, prev, count))
            start = t
            prev = t
            count = 1

    windows.append((start, prev, count))

    out = pd.DataFrame(windows, columns=["win_start", "win_end", "n_points"])
    # widen slightly (+1 min each side)
    out["win_start"] = out["win_start"] - pd.Timedelta(minutes=1)
    out["win_end"] = out["win_end"] + pd.Timedelta(minutes=1)
    return out


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
    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])

    # Clean strings
    metrics["entity"] = metrics["entity"].astype(str).str.strip()
    metrics["kpi"] = metrics["kpi"].astype(str).str.strip()

    # Align timezone (+8h like your other scripts)
    metrics["timestamp"] = metrics["timestamp"] + pd.Timedelta(hours=8)

    # Restrict failures to the day window (with buffer)
    day_min = metrics["timestamp"].min()
    day_max = metrics["timestamp"].max()
    buf = pd.Timedelta(minutes=10)

    day_fails = fails[
        (fails["event_start"] >= day_min - buf) &
        (fails["event_start"] <= day_max + buf) &
        (fails["kpi_one"].notna()) &
        (fails["name"].notna())
    ].copy()

    print("Day window:", day_min, "->", day_max)
    print("Failures in this window:", len(day_fails))

    # Detection params (same as your robust detector)
    z_thresh = 3.5
    win = 15
    min_p = 5

    results = []

    for _, fr in day_fails.iterrows():
        entity = str(fr["name"]).strip()
        kpi = str(fr["kpi_one"]).strip()
        ev_start = fr["event_start"]
        ev_end = fr["event_end"]

        m = metrics[(metrics["entity"] == entity) & (metrics["kpi"] == kpi)].copy()

        if m.empty:
            results.append({
                "failure_index": fr.get("index"),
                "entity": entity,
                "kpi": kpi,
                "fault_description": fr.get("fault_description"),
                "event_start": ev_start,
                "event_end": ev_end,
                "has_metric_series": False,
                "detected": False,
                "first_detect_time": None,
                "detection_delay_min": None,
                "anom_points": 0,
                "anom_windows": 0,
            })
            continue

        m = m.sort_values("timestamp")

        # Build a clean aligned 1-min series table
        s = (
            m.set_index("timestamp")["value"]
            .resample("1min")
            .mean()
            .interpolate(limit=3)
            .reset_index()
            .rename(columns={"timestamp": "t", "value": "v"})
        )

        z = rolling_mad_zscore(s["v"], window=win, min_p=min_p)
        is_anom = (z.abs() >= z_thresh).astype(int)

        wdf = points_to_windows(s["t"], is_anom, gap_min=2)

        hit = False
        first_detect = None
        for _, w in wdf.iterrows():
            if overlap(w["win_start"], w["win_end"], ev_start, ev_end):
                hit = True
                if first_detect is None or w["win_start"] < first_detect:
                    first_detect = w["win_start"]

        delay_min = None
        if hit and first_detect is not None:
            delay_min = (first_detect - ev_start).total_seconds() / 60.0

        results.append({
            "failure_index": fr.get("index"),
            "entity": entity,
            "kpi": kpi,
            "fault_description": fr.get("fault_description"),
            "event_start": ev_start,
            "event_end": ev_end,
            "has_metric_series": True,
            "detected": hit,
            "first_detect_time": first_detect,
            "detection_delay_min": delay_min,
            "anom_points": int(is_anom.sum()),
            "anom_windows": int(len(wdf)),
        })

    out = pd.DataFrame(results)
    out = out.sort_values(["detected", "detection_delay_min"], ascending=[False, True], na_position="last")

    out_csv = OUT_DIR / f"batch_eval_{DAY_TAG}.csv"
    written = safe_write_csv(out, out_csv)

    # Summary
    total = len(out)
    has_series = int(out["has_metric_series"].sum())
    detected = int(out["detected"].sum())
    delays = out.loc[out["detected"] & out["detection_delay_min"].notna(), "detection_delay_min"]

    summary_lines = [
        f"DAY: {DAY_TAG}",
        f"Failures in day window: {total}",
        f"Failures with available metric series: {has_series}",
        f"Detected failures: {detected}",
        f"Detection rate (over failures with series): {round(detected / max(has_series, 1), 3)}",
    ]
    if len(delays):
        summary_lines += [
            f"Avg delay (min): {round(float(delays.mean()), 2)}",
            f"Median delay (min): {round(float(delays.median()), 2)}",
            f"Min/Max delay (min): {round(float(delays.min()), 2)} / {round(float(delays.max()), 2)}",
        ]

    summary_txt = OUT_DIR / f"batch_summary_{DAY_TAG}.txt"
    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print("\nSaved:", written)
    print("Saved:", summary_txt)


if __name__ == "__main__":
    main()
