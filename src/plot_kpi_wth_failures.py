from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA_PROCESSED = PROJ / "data" / "processed"

DAY_TAG = "2020_04_11"
METRICS_PATH = DATA_PROCESSED / f"day_{DAY_TAG}_metrics.parquet"
FAIL_PATH = DATA_PROCESSED / "failures_clean.csv"

OUT_FIG = PROJ / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)


def main():
    metrics = pd.read_parquet(METRICS_PATH)
    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])

    # Choose one failure KPI & entity from your table (example shown in your output)
    target_kpi = "container_cpu_used"
    # Your failure row shows docker_003 (name) and container_001 (container)
    # Platform container metrics use cmdb_id like container_001, container_002, etc.
    # We'll try entity = container_001 first.
    target_entity = "container_001"

    # Filter series
    s = metrics[(metrics["kpi"] == target_kpi) & (metrics["entity"] == target_entity)].copy()
    if s.empty:
        print("No rows found for", target_kpi, target_entity)
        # print some suggestions
        print("Example KPIs that contain 'cpu':", metrics[metrics["kpi"].str.contains("cpu", case=False, na=False)]["kpi"].unique()[:20])
        print("Example entities that contain 'container':", metrics[metrics["entity"].str.contains("container", case=False, na=False)]["entity"].unique()[:20])
        return

    s = s.sort_values("timestamp")
    # resample to 1-min grid (optional)
    s = s.set_index("timestamp")["value"].resample("1min").mean().interpolate(limit=3).reset_index()

    # Failures for same object family (docker/container)
    f_sub = fails[(fails["object"].astype(str).str.contains("docker|container", case=False, na=False))].copy()

    plt.figure()
    plt.plot(s["timestamp"], s["value"])
    for _, r in f_sub.iterrows():
        if pd.notna(r["event_start"]) and pd.notna(r["event_end"]):
            plt.axvspan(r["event_start"], r["event_end"], alpha=0.2)

    plt.title(f"{target_entity} | {target_kpi} with failure windows")
    plt.xlabel("Time")
    plt.ylabel("Value")
    out_path = OUT_FIG / f"{DAY_TAG}_{target_entity}_{target_kpi}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
