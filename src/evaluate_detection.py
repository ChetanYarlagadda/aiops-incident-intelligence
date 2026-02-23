from __future__ import annotations
from pathlib import Path
import pandas as pd
from datetime import datetime


PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")

FAIL_PATH = PROJ / "data" / "processed" / "failures_clean.csv"

# <-- IMPORTANT: point to ROBUST scores file
SCORES_PATH = PROJ / "outputs" / "reports" / "2020_04_11_docker_003_container_cpu_used_robust.csv"

OUT_DIR = PROJ / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def points_to_windows(df: pd.DataFrame, time_col="timestamp", flag_col="is_anom", gap_min=2) -> pd.DataFrame:
    d = df[df[flag_col] == 1].copy()
    if d.empty:
        return pd.DataFrame(columns=["win_start", "win_end", "n_points"])

    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col)

    windows = []
    start = d.iloc[0][time_col]
    prev = start
    count = 1

    for t in d.iloc[1:][time_col]:
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
    out["win_start"] = out["win_start"] - pd.Timedelta(minutes=1)
    out["win_end"] = out["win_end"] + pd.Timedelta(minutes=1)
    return out


def overlap(a_start, a_end, b_start, b_end) -> bool:
    return (a_start <= b_end) and (b_start <= a_end)


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Avoid PermissionError if CSV is open in Excel.
    If blocked, write to a timestamped filename.
    """
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{ts}" + path.suffix)
        df.to_csv(alt, index=False, encoding="utf-8")
        return alt


def main():
    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])
    scores = pd.read_csv(SCORES_PATH)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])

    target_entity = "docker_003"
    target_kpi = "container_cpu_used"

    # ---- FIX 1: restrict failures to the score time span ----
    scores_min = scores["timestamp"].min()
    scores_max = scores["timestamp"].max()
    buf = pd.Timedelta(minutes=10)

    f = fails[
        (fails["name"] == target_entity) &
        (fails["kpi_one"] == target_kpi) &
        (fails["event_start"] >= scores_min - buf) &
        (fails["event_start"] <= scores_max + buf)
    ].copy()

    print("\n=== Score Time Window ===")
    print(scores_min, "->", scores_max)

    if f.empty:
        print("No failures found for", target_entity, target_kpi, "within score time window.")
        return

    print("\n=== Filtered Failures (Fix 1 applied) ===")
    show_cols = [c for c in ["index", "fault_description", "event_start", "event_end"] if c in f.columns]
    print(f[show_cols].to_string(index=False))

    wins = points_to_windows(scores, gap_min=2)

    print("\n=== Anomaly Windows ===")
    print(wins.to_string(index=False) if len(wins) else "None")

    rows = []
    for _, r in f.iterrows():
        ev_start, ev_end = r["event_start"], r["event_end"]
        hit = False
        first_detect_time = None

        for _, w in wins.iterrows():
            if overlap(w["win_start"], w["win_end"], ev_start, ev_end):
                hit = True
                if first_detect_time is None or w["win_start"] < first_detect_time:
                    first_detect_time = w["win_start"]

        delay_min = None
        if hit and pd.notna(first_detect_time):
            delay_min = (first_detect_time - ev_start).total_seconds() / 60.0

        rows.append({
            "entity": target_entity,
            "kpi": target_kpi,
            "failure_index": r.get("index"),
            "fault_description": r.get("fault_description"),
            "event_start": ev_start,
            "event_end": ev_end,
            "detected": hit,
            "first_detect_time": first_detect_time,
            "detection_delay_min": delay_min,
        })

    report = pd.DataFrame(rows)

    out_path = OUT_DIR / f"eval_{target_entity}_{target_kpi}.csv"
    written_path = safe_to_csv(report, out_path)

    detected_count = int(report["detected"].sum())
    total = len(report)
    delays = report.loc[report["detected"] & report["detection_delay_min"].notna(), "detection_delay_min"]

    print("\n=== Evaluation Summary ===")
    print("Failures:", total)
    print("Detected:", detected_count)
    print("Detection rate:", round(detected_count / total, 3))
    if len(delays):
        print("Avg delay (min):", round(float(delays.mean()), 2))
        print("Median delay (min):", round(float(delays.median()), 2))
        print("Min/Max delay (min):", round(float(delays.min()), 2), "/", round(float(delays.max()), 2))

    print("\nSaved evaluation report:", written_path)


if __name__ == "__main__":
    main()
