from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA_PROCESSED = PROJ / "data" / "processed"

DAY_TAG = "2020_04_11"
METRICS_PATH = DATA_PROCESSED / f"day_{DAY_TAG}_metrics.parquet"

OUT_DIR = PROJ / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    """
    Robust z-score using rolling median and MAD.
    z = 0.6745 * (x - median) / MAD
    Works better for noisy telemetry and doesn’t require big warmup.
    """
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median()
    mad = mad.replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)


def main():
    metrics = pd.read_parquet(METRICS_PATH)
    metrics["entity"] = metrics["entity"].astype(str).str.strip()
    metrics["kpi"] = metrics["kpi"].astype(str).str.strip()

    # align timezone like before
    metrics["timestamp"] = metrics["timestamp"] + pd.Timedelta(hours=8)

    target_entity = "docker_003"
    target_kpi = "container_cpu_used"

    s = metrics[(metrics["entity"] == target_entity) & (metrics["kpi"] == target_kpi)].copy()
    s = s.sort_values("timestamp")

    # 1-min grid
    s = s.set_index("timestamp")["value"].resample("1min").mean().interpolate(limit=3).reset_index()

    s["z_robust"] = rolling_mad_zscore(s["value"], window=15, min_p=5)

    # Threshold: robust z usually uses ~3.5
    thresh = 3.5
    s["is_anom"] = (s["z_robust"].abs() >= thresh).astype(int)

    out_path = OUT_DIR / f"{DAY_TAG}_{target_entity}_{target_kpi}_robust.csv"
    s.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved robust anomaly scores:", out_path)
    print("Anomaly points:", int(s["is_anom"].sum()))
    print("Time range:", s["timestamp"].min(), "->", s["timestamp"].max())


if __name__ == "__main__":
    main()
