from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA = PROJ / "data" / "processed"
REPORTS = PROJ / "outputs" / "reports"

DAY_TAG = "2020_04_11"
METRICS_PATH = DATA / f"day_{DAY_TAG}_metrics.parquet"

# Missed failure details (from your output)
ENTITY = "docker_008"
START = pd.Timestamp("2020-04-11 04:40:00")
END   = pd.Timestamp("2020-04-11 04:45:00")

def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)

def main():
    m = pd.read_parquet(METRICS_PATH)
    m["timestamp"] = m["timestamp"] + pd.Timedelta(hours=8)
    m["entity"] = m["entity"].astype(str).str.strip()
    m["kpi"] = m["kpi"].astype(str).str.strip()

    em = m[m["entity"] == ENTITY].copy()
    if em.empty:
        print("No metrics found for entity:", ENTITY)
        return

    # aggregate to 1-min
    em["t"] = em["timestamp"].dt.floor("min")
    agg = em.groupby(["kpi", "t"], as_index=False)["value"].mean()

    full_index = pd.date_range(agg["t"].min().floor("min"), agg["t"].max().floor("min"), freq="1min")

    results = []
    for kpi, g in agg.groupby("kpi"):
        s = g.set_index("t")["value"].reindex(full_index).interpolate(limit=3)
        z = rolling_mad_zscore(s, window=15, min_p=5).abs()

        mask = (full_index >= START - pd.Timedelta(minutes=2)) & (full_index <= END + pd.Timedelta(minutes=2))
        peak = float(z[mask].max()) if mask.any() else 0.0
        results.append((kpi, peak, int((z >= 3.5).sum())))

    out = pd.DataFrame(results, columns=["kpi", "peak_abs_z_in_window", "anom_points_total"])
    out = out.sort_values("peak_abs_z_in_window", ascending=False)

    print("\nTop KPIs for", ENTITY, "around", START, "->", END)
    print(out.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
