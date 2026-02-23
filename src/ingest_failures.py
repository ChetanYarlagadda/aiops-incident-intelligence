from __future__ import annotations
from pathlib import Path
import re
import pandas as pd


DATA_ROOT = Path(r"C:\Users\ychet\Desktop\Project\AIOps\AIOps挑战赛数据")
FAILURE_CSV = DATA_ROOT / "故障整理（预赛）.csv"

OUT_DIR = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence\data\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="ignore")


def parse_duration_to_minutes(x) -> float | None:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    # common patterns: 5min, 10mins, 1h, 2hour, 30m
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(min|mins|m|hour|hours|h)\s*$", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit in ["hour", "hours", "h"]:
        return val * 60.0
    return val


def main():
    df = read_csv_robust(FAILURE_CSV)

    # Normalize column names a bit
    # (keep originals too, but avoid typos pain)
    rename_map = {
        "fault_desrcibtion": "fault_description",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse times
    # Your sample shows: 2020/4/11 0:05
    for col in ["log_time", "start_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Duration to minutes
    df["duration_min"] = df["duration"].apply(parse_duration_to_minutes) if "duration" in df.columns else None

    # Choose start time for event
    df["event_start"] = df["log_time"]
    if "start_time" in df.columns:
        df["event_start"] = df["event_start"].fillna(df["start_time"])

    df["event_end"] = df["event_start"] + pd.to_timedelta(df["duration_min"].fillna(0), unit="m")

    # Explode KPI list (some rows have NaN KPI; keep them too)
    if "kpi" in df.columns:
        df["kpi_list"] = df["kpi"].fillna("").astype(str).apply(lambda s: [k.strip() for k in s.split(";") if k.strip()])
    else:
        df["kpi_list"] = [[] for _ in range(len(df))]

    # Make a “long” version: one row per KPI (or single row with kpi=None if empty)
    rows = []
    for _, r in df.iterrows():
        kpis = r["kpi_list"]
        if not kpis:
            rr = r.copy()
            rr["kpi_one"] = None
            rows.append(rr)
        else:
            for k in kpis:
                rr = r.copy()
                rr["kpi_one"] = k
                rows.append(rr)

    out = pd.DataFrame(rows)

    # Keep only useful columns first (you can keep more later)
    keep_cols = [c for c in [
        "index",
        "object",
        "fault_description",
        "kpi_one",
        "name",
        "container",
        "log_time",
        "event_start",
        "event_end",
        "duration",
        "duration_min",
        "block",
        "log_block",
    ] if c in out.columns]

    out = out[keep_cols].sort_values(["event_start", "index"], na_position="last")

    out_path = OUT_DIR / "failures_clean.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved:", out_path)
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
