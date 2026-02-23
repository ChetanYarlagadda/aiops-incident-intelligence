from __future__ import annotations
from pathlib import Path
import pandas as pd

from build_day_metrics_from_zip import build_day_metrics_from_zip
from build_incidents_allkpis_day import build_incidents_for_day

# Reuse your existing batch runner script outputs:
# - trace KPI builder is already inside batch_run_all_days.py, but we’ll just call it by importing that file
#   If you prefer not to import, re-run batch_run_all_days.py after this script.

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA_ROOT = Path(r"C:\Users\ychet\Desktop\Project\AIOps\AIOps挑战赛数据")

PROC = PROJ / "data" / "processed"
REPORTS = PROJ / "outputs" / "reports"

def main():
    zips = sorted(DATA_ROOT.glob("2020_*.zip"))
    days = [z.stem for z in zips]

    print("Found ZIP days:", len(days))

    # 1) build metrics parquet for all days
    for day in days:
        out_parquet = PROC / f"day_{day}_metrics.parquet"
        if out_parquet.exists():
            continue
        zip_path = DATA_ROOT / f"{day}.zip"
        print(f"[metrics] building {day} ...")
        build_day_metrics_from_zip(zip_path, day, out_parquet)
        print(f"[metrics] saved {out_parquet.name}")

    # 2) build incidents for all days
    for day in days:
        inc_csv = REPORTS / f"incidents_allkpis_{day}.csv"
        if inc_csv.exists():
            continue
        print(f"[incidents] building {day} ...")
        r = build_incidents_for_day(day)
        print(f"[incidents] {day} -> {r['status']}")

    print("\nDone. Now run:")
    print("  python src/batch_run_all_days.py")
    print("and then:")
    print("  streamlit run app.py")

if __name__ == "__main__":
    main()
