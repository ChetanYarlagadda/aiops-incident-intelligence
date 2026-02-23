from __future__ import annotations
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np


PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA_ROOT = Path(r"C:\Users\ychet\Desktop\Project\AIOps\AIOps挑战赛数据")  # your dataset folder
OUT_DIR = PROJ / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY_TAG = "2020_04_11"
DAY_ZIP = DATA_ROOT / f"{DAY_TAG}.zip"

# inside zip: 2020_04_11/调用链指标/trace_*.csv
TRACE_DIR = f"{DAY_TAG}/调用链指标/"


def read_trace_csv_from_zip(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as f:
        df = pd.read_csv(f)
    return df


def main():
    if not DAY_ZIP.exists():
        raise FileNotFoundError(f"Missing zip: {DAY_ZIP}")

    rows = []
    with zipfile.ZipFile(DAY_ZIP, "r") as zf:
        members = [m for m in zf.namelist() if m.startswith(TRACE_DIR) and m.endswith(".csv") and "trace_" in m]
        if not members:
            print("No trace CSVs found inside zip under:", TRACE_DIR)
            return

        print("Trace files:", len(members))
        for m in members:
            df = read_trace_csv_from_zip(zf, m)

            # expected columns: callType,startTime,elapsedTime,success,traceId,id,pid,cmdb_id,serviceName
            needed = {"startTime", "elapsedTime", "success", "serviceName"}
            missing = needed - set(df.columns)
            if missing:
                print("Skipping", m, "missing:", missing)
                continue

            # startTime looks like ms epoch
            t = pd.to_datetime(df["startTime"], unit="ms", errors="coerce") + pd.Timedelta(hours=8)  # align timezone
            df = df.assign(t=t.dt.floor("min"))

            # coerce types
            df["elapsedTime"] = pd.to_numeric(df["elapsedTime"], errors="coerce")
            # success can be True/False or 1/0
            if df["success"].dtype != bool:
                df["success"] = df["success"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})

            df = df.dropna(subset=["t", "serviceName", "elapsedTime", "success"])

            # aggregate per serviceName per minute
            g = df.groupby(["serviceName", "t"], as_index=False).agg(
                trace_count=("elapsedTime", "size"),
                err_rate=("success", lambda x: float((~x).mean())),
                p95_latency=("elapsedTime", lambda x: float(np.nanpercentile(x, 95))),
                avg_latency=("elapsedTime", "mean"),
            )

            # keep a tag of which trace file contributed (optional)
            g["trace_file"] = Path(m).name.replace(".csv", "")
            rows.append(g)

    out = pd.concat(rows, ignore_index=True)

    # If same serviceName+t appears across different trace files, combine again
    out2 = out.groupby(["serviceName", "t"], as_index=False).agg(
        trace_count=("trace_count", "sum"),
        err_rate=("err_rate", "mean"),          # average err rate across sources
        p95_latency=("p95_latency", "max"),     # max p95 across sources is more sensitive
        avg_latency=("avg_latency", "mean"),
    )

    # convert to long "metrics-like" format: timestamp/entity/kpi/value/source
    long_rows = []
    for col in ["trace_count", "err_rate", "p95_latency", "avg_latency"]:
        tmp = out2[["serviceName", "t", col]].copy()
        tmp = tmp.rename(columns={"serviceName": "entity", "t": "timestamp", col: "value"})
        tmp["kpi"] = f"trace_{col}"
        tmp["source"] = "trace"
        long_rows.append(tmp)

    trace_metrics = pd.concat(long_rows, ignore_index=True)

    out_path = OUT_DIR / f"day_{DAY_TAG}_trace_kpis.parquet"
    trace_metrics.to_parquet(out_path, index=False)
    print("Saved:", out_path)
    print("Rows:", len(trace_metrics))
    print("Unique services:", trace_metrics["entity"].nunique())
    print("KPIs:", trace_metrics["kpi"].unique())


if __name__ == "__main__":
    main()
