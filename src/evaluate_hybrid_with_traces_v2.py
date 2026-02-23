from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA_PROCESSED = PROJ / "data" / "processed"
REPORTS = PROJ / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DAY_TAG = "2020_04_11"

FAIL_PATH = DATA_PROCESSED / "failures_clean.csv"
INC_PATH  = REPORTS / f"incidents_allkpis_{DAY_TAG}.csv"
TRACE_PATH = DATA_PROCESSED / f"day_{DAY_TAG}_trace_kpis.parquet"

OUT_CSV = REPORTS / f"hybrid_eval_v2_{DAY_TAG}.csv"


def clean_str(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def overlap(a_start, a_end, b_start, b_end) -> bool:
    return (a_start <= b_end) and (b_start <= a_end)


def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{ts}" + path.suffix)
        df.to_csv(alt, index=False, encoding="utf-8")
        return alt


def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)


def build_trace_z_table(trace_long: pd.DataFrame, day_min: pd.Timestamp, day_max: pd.Timestamp,
                        z_win: int = 15, min_p: int = 5) -> pd.DataFrame:
    full_index = pd.date_range(day_min.floor("min"), day_max.floor("min"), freq="1min")

    trace_long = trace_long.copy()
    trace_long["entity"] = trace_long["entity"].astype(str).str.strip()
    trace_long["kpi"] = trace_long["kpi"].astype(str).str.strip()
    trace_long["timestamp"] = pd.to_datetime(trace_long["timestamp"])
    trace_long["t"] = trace_long["timestamp"].dt.floor("min")

    keep_kpis = {"trace_err_rate", "trace_p95_latency", "trace_trace_count"}
    trace_long = trace_long[trace_long["kpi"].isin(keep_kpis)].copy()

    agg = trace_long.groupby(["entity", "kpi", "t"], as_index=False)["value"].mean()

    z_rows = []
    for (svc, kpi), g in agg.groupby(["entity", "kpi"], sort=False):
        s = g.set_index("t")["value"].reindex(full_index).interpolate(limit=3)
        z = rolling_mad_zscore(s, window=z_win, min_p=min_p).abs()

        z_rows.append(pd.DataFrame({
            "timestamp": full_index,
            "service": svc,
            "kpi": kpi,
            "z_abs": z.values
        }))

    if not z_rows:
        return pd.DataFrame(columns=["timestamp", "service", "kpi", "z_abs"])

    return pd.concat(z_rows, ignore_index=True)


def main():
    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])
    inc = pd.read_csv(INC_PATH, parse_dates=["incident_start", "incident_end"])
    traces = pd.read_parquet(TRACE_PATH)

    inc["entity"] = inc["entity"].astype(str).str.strip()

    day_min = inc["incident_start"].min()
    day_max = inc["incident_end"].max()
    buf = pd.Timedelta(minutes=10)

    day_fails = fails[
        (fails["event_start"] >= day_min - buf) &
        (fails["event_start"] <= day_max + buf)
    ].copy()

    if "name" in day_fails.columns:
        day_fails["name"] = day_fails["name"].astype(str).str.strip()
    if "container" in day_fails.columns:
        day_fails["container"] = day_fails["container"].astype(str).str.strip()
    if "fault_description" in day_fails.columns:
        day_fails["fault_description"] = day_fails["fault_description"].astype(str).str.strip()

    print("Day window:", day_min, "->", day_max)
    print("Failures in window:", len(day_fails))

    print("Building trace anomaly table (z-scores)...")
    ztab = build_trace_z_table(traces, day_min, day_max, z_win=15, min_p=5)
    print("Trace z-table rows:", len(ztab))

    # trace fallback for these failure types
    TRACE_FALLBACK_TYPES = {"network delay", "network loss", "CPU fault"}

    # network/cpu symptom KPIs (use OR for recall)
    SYMPTOM_KPIS = {"trace_p95_latency", "trace_err_rate"}
    Z_THRESH_TRACE = 3.5

    rows = []

    for _, fr in day_fails.iterrows():
        ev_s, ev_e = fr["event_start"], fr["event_end"]
        fdesc = clean_str(fr.get("fault_description")) or ""

        # ---- metric incident match: name + container ----
        candidates = []
        nm = clean_str(fr.get("name"))
        ct = clean_str(fr.get("container"))
        if nm: candidates.append(nm)
        if ct and ct not in candidates: candidates.append(ct)

        metric_hit = False
        metric_entity = None
        metric_inc_start = None

        for ent in candidates:
            incs = inc[inc["entity"] == ent]
            for _, ir in incs.iterrows():
                if overlap(ir["incident_start"], ir["incident_end"], ev_s, ev_e):
                    metric_hit = True
                    if metric_inc_start is None or ir["incident_start"] < metric_inc_start:
                        metric_inc_start = ir["incident_start"]
                        metric_entity = ent

        metric_delay = None
        metric_lead = None
        if metric_hit and metric_inc_start is not None:
            raw = (metric_inc_start - ev_s).total_seconds() / 60.0
            if raw < 0:
                metric_lead = abs(raw)
                metric_delay = 0.0
            else:
                metric_delay = raw
                metric_lead = 0.0

        # ---- trace fallback (now includes CPU fault) ----
        trace_hit = False
        top_services = ""
        top_peak = None

        if (not metric_hit) and (fdesc in TRACE_FALLBACK_TYPES):
            w0 = ev_s - pd.Timedelta(minutes=1)
            w1 = ev_e + pd.Timedelta(minutes=1)

            w = ztab[(ztab["timestamp"] >= w0) & (ztab["timestamp"] <= w1)].copy()
            w = w[w["kpi"].isin(SYMPTOM_KPIS)]

            if not w.empty:
                svc_kpi = w.groupby(["service", "kpi"], as_index=False)["z_abs"].max()
                svc_best = (svc_kpi.groupby("service", as_index=False)["z_abs"]
                            .max().sort_values("z_abs", ascending=False))

                if not svc_best.empty and float(svc_best.iloc[0]["z_abs"]) >= Z_THRESH_TRACE:
                    trace_hit = True
                    top_peak = float(svc_best.iloc[0]["z_abs"])
                    top5 = svc_best.head(5)
                    top_services = "; ".join([f"{r.service}:{float(r.z_abs):.2f}" for r in top5.itertuples(index=False)])

        hybrid_hit = metric_hit or trace_hit

        rows.append({
            "failure_index": fr.get("index"),
            "fault_description": fdesc,
            "name": nm,
            "container": ct,
            "event_start": ev_s,
            "event_end": ev_e,

            "metric_detected": metric_hit,
            "metric_matched_entity": metric_entity,
            "metric_incident_start": metric_inc_start,
            "metric_delay_min": metric_delay,
            "metric_lead_min": metric_lead,

            "trace_detected": trace_hit,
            "trace_top_services_peakz": top_services,
            "trace_top_peak_z": top_peak,

            "hybrid_detected": hybrid_hit,
        })

    out = pd.DataFrame(rows)
    out_path = safe_write_csv(out, OUT_CSV)

    total = len(out)
    det = int(out["hybrid_detected"].sum())

    print("\n=== HYBRID Summary (v2) ===")
    print("Failures:", total)
    print("Detected:", det)
    print("Detection rate:", round(det / max(total, 1), 3))

    by_type = out.groupby("fault_description")["hybrid_detected"].agg(["count", "sum"])
    by_type["rate"] = (by_type["sum"] / by_type["count"]).round(3)
    by_type = by_type.rename(columns={"count": "failures", "sum": "detected"}).sort_values("rate", ascending=False)
    print("\n=== HYBRID detection by fault_description (v2) ===")
    print(by_type.to_string())

    missed = out[out["hybrid_detected"] == False]
    print("\nMissed failures:", len(missed))
    if len(missed):
        cols = ["failure_index", "fault_description", "name", "container", "event_start", "event_end"]
        print(missed[cols].head(20).to_string(index=False))

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
