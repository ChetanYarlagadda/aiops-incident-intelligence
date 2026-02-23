from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
METRICS_PATH = PROJ / "data" / "processed" / "day_2020_04_11_metrics.parquet"
FAIL_PATH = PROJ / "data" / "processed" / "failures_clean.csv"

def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)

def main():
    metrics = pd.read_parquet(METRICS_PATH)
    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])

    metrics["entity"] = metrics["entity"].astype(str).str.strip()
    metrics["kpi"] = metrics["kpi"].astype(str).str.strip()
    metrics["timestamp"] = metrics["timestamp"] + pd.Timedelta(hours=8)

    entity = "docker_003"

    # restrict failures to this day window
    day_min = metrics["timestamp"].min()
    day_max = metrics["timestamp"].max()
    f = fails[(fails["name"] == entity) & (fails["event_start"].between(day_min, day_max))].copy()
    if f.empty:
        print("No failures for", entity, "in this day window.")
        return

    # pick first failure window for scoring
    ev_start = f.iloc[0]["event_start"]
    ev_end = f.iloc[0]["event_end"]
    print("Scoring KPIs for failure window:", ev_start, "->", ev_end)

    results = []
    kpis = metrics[metrics["entity"] == entity]["kpi"].unique()

    for kpi in kpis:
        s = metrics[(metrics["entity"] == entity) & (metrics["kpi"] == kpi)].copy()
        s = s.sort_values("timestamp")
        s = s.set_index("timestamp")["value"].resample("1min").mean().interpolate(limit=3).reset_index()

        z = rolling_mad_zscore(s["value"], window=15, min_p=5)
        is_anom = (z.abs() >= 3.5)

        # any anomaly inside the failure window?
        mask = (s["timestamp"] >= ev_start) & (s["timestamp"] <= ev_end)
        hit = bool(is_anom[mask].any())

        # peak z in window as a score
        peak = float(z[mask].abs().max()) if mask.any() else 0.0

        results.append((kpi, hit, peak, int(is_anom.sum())))

    out = pd.DataFrame(results, columns=["kpi", "hit_in_failure_window", "peak_abs_z_in_window", "anom_points_total"])
    out = out.sort_values(["hit_in_failure_window", "peak_abs_z_in_window"], ascending=[False, False])

    print("\nTop 15 KPIs by ability to detect the fault window:")
    print(out.head(15).to_string(index=False))

    out_path = PROJ / "outputs" / "reports" / f"best_kpi_search_{entity}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
