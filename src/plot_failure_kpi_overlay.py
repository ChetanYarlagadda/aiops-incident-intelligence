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

    # Clean strings
    metrics["entity"] = metrics["entity"].astype(str).str.strip()
    metrics["kpi"] = metrics["kpi"].astype(str).str.strip()

    # IMPORTANT: align timezone (metrics appear ~8 hours behind failures)
    metrics["timestamp"] = metrics["timestamp"] + pd.Timedelta(hours=8)

    # Pick a specific failure row to start with:
    # Example from your failure CSV head: docker_003 has KPI container_cpu_used at 2020/4/11 0:05
    target_entity = "docker_003"
    target_kpi = "container_cpu_used"

    # Filter failures for the same "name" (docker_003) and KPI (container_cpu_used)
    f = fails[(fails["name"] == target_entity) & (fails["kpi_one"] == target_kpi)].copy()
    if f.empty:
        print("No matching failures found for", target_entity, target_kpi)
        print("Try checking failures_clean.csv for available names/kpis.")
        return

    # Filter metric series
    s = metrics[(metrics["entity"] == target_entity) & (metrics["kpi"] == target_kpi)].copy()
    if s.empty:
        print("No metric rows found for", target_entity, target_kpi)
        # Show what KPIs exist for that entity
        print("KPIs for entity:", metrics[metrics["entity"] == target_entity]["kpi"].value_counts().head(20))
        return

    s = s.sort_values("timestamp")
    # resample to 1-min grid to smooth and align
    s = s.set_index("timestamp")["value"].resample("1min").mean().interpolate(limit=3).reset_index()

    plt.figure()
    plt.plot(s["timestamp"], s["value"])

    # Shade failure windows
    for _, r in f.iterrows():
        if pd.notna(r["event_start"]) and pd.notna(r["event_end"]):
            plt.axvspan(r["event_start"], r["event_end"], alpha=0.25)

    plt.title(f"{target_entity} | {target_kpi} with matching failure windows")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()

    out_path = OUT_FIG / f"{DAY_TAG}_{target_entity}_{target_kpi}_failure_overlay.png"
    plt.savefig(out_path, dpi=150)
    print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
