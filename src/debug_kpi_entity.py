from pathlib import Path
import pandas as pd

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
METRICS_PATH = PROJ / "data" / "processed" / "day_2020_04_11_metrics.parquet"

metrics = pd.read_parquet(METRICS_PATH)

# strip to remove hidden spaces
metrics["entity"] = metrics["entity"].astype(str).str.strip()
metrics["kpi"] = metrics["kpi"].astype(str).str.strip()

target_kpi = "container_cpu_used"
target_entity = "container_001"

print("\n--- Entities that have container_cpu_used (top 20) ---")
s = metrics[metrics["kpi"] == target_kpi]
print(s["entity"].value_counts().head(20))

print("\n--- KPIs available for container_001 (top 30) ---")
e = metrics[metrics["entity"] == target_entity]
print(e["kpi"].value_counts().head(30))

print("\n--- Quick check: rows where entity=container_001 & kpi=container_cpu_used ---")
both = metrics[(metrics["entity"] == target_entity) & (metrics["kpi"] == target_kpi)]
print("Rows found:", len(both))
if len(both) > 0:
    print(both.head(5).to_string(index=False))
