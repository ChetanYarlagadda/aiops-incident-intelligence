from __future__ import annotations
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# CONFIG
# =========================
PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DATA_ROOT = Path(r"C:\Users\ychet\Desktop\Project\AIOps\AIOps挑战赛数据")

PROC = PROJ / "data" / "processed"
REPORTS = PROJ / "outputs" / "reports"
FIGS = PROJ / "outputs" / "figures"
for p in [PROC, REPORTS, FIGS]:
    p.mkdir(parents=True, exist_ok=True)

FAIL_PATH = PROC / "failures_clean.csv"   # must exist

# robust z params
Z_THRESH_TRACE = 3.5
Z_WIN = 15
MIN_P = 5

TRACE_DIR_TMPL = "{day}/调用链指标/"

# =========================
# UTILS
# =========================
def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(path.stem + f"_{ts}" + path.suffix)
        df.to_csv(alt, index=False, encoding="utf-8")
        return alt

def clean_str(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    return s

def overlap(a_start, a_end, b_start, b_end) -> bool:
    return (a_start <= b_end) and (b_start <= a_end)

def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)

# =========================
# TRACE KPI BUILD
# =========================
def build_trace_kpis_for_day(day_tag: str) -> Path | None:
    zip_path = DATA_ROOT / f"{day_tag}.zip"
    if not zip_path.exists():
        return None

    trace_dir = TRACE_DIR_TMPL.format(day=day_tag)
    rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.startswith(trace_dir) and m.endswith(".csv") and "trace_" in m]
        if not members:
            return None

        for m in members:
            with zf.open(m) as f:
                df = pd.read_csv(f)

            needed = {"startTime", "elapsedTime", "success"}
            if not needed.issubset(df.columns):
                continue

            # serviceName preferred; if missing, fall back to cmdb_id (helps trace_jdbc.csv)
            if "serviceName" in df.columns:
                entity_col = "serviceName"
            elif "cmdb_id" in df.columns:
                entity_col = "cmdb_id"
            else:
                continue

            t = pd.to_datetime(df["startTime"], unit="ms", errors="coerce") + pd.Timedelta(hours=8)
            df = df.assign(t=t.dt.floor("min"))
            df["elapsedTime"] = pd.to_numeric(df["elapsedTime"], errors="coerce")

            if df["success"].dtype != bool:
                df["success"] = df["success"].astype(str).str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )

            df[entity_col] = df[entity_col].astype(str).str.strip()
            df = df.dropna(subset=["t", entity_col, "elapsedTime", "success"])

            g = df.groupby([entity_col, "t"], as_index=False).agg(
                trace_count=("elapsedTime", "size"),
                err_rate=("success", lambda x: float((~x).mean())),
                p95_latency=("elapsedTime", lambda x: float(np.nanpercentile(x, 95))),
                avg_latency=("elapsedTime", "mean"),
            )
            g = g.rename(columns={entity_col: "entity"})
            rows.append(g)

    if not rows:
        return None

    out = pd.concat(rows, ignore_index=True)
    # combine duplicates
    out2 = out.groupby(["entity", "t"], as_index=False).agg(
        trace_count=("trace_count", "sum"),
        err_rate=("err_rate", "mean"),
        p95_latency=("p95_latency", "max"),
        avg_latency=("avg_latency", "mean"),
    )

    long_rows = []
    for col in ["trace_count", "err_rate", "p95_latency", "avg_latency"]:
        tmp = out2[["entity", "t", col]].copy()
        tmp = tmp.rename(columns={"t": "timestamp", col: "value"})
        tmp["kpi"] = f"trace_{col}"
        tmp["source"] = "trace"
        long_rows.append(tmp)

    trace_metrics = pd.concat(long_rows, ignore_index=True)
    out_path = PROC / f"day_{day_tag}_trace_kpis.parquet"
    trace_metrics.to_parquet(out_path, index=False)
    return out_path

def build_trace_z_table(trace_long: pd.DataFrame, day_min: pd.Timestamp, day_max: pd.Timestamp) -> pd.DataFrame:
    full_index = pd.date_range(day_min.floor("min"), day_max.floor("min"), freq="1min")

    trace_long = trace_long.copy()
    trace_long["entity"] = trace_long["entity"].astype(str).str.strip()
    trace_long["kpi"] = trace_long["kpi"].astype(str).str.strip()
    trace_long["timestamp"] = pd.to_datetime(trace_long["timestamp"])
    trace_long["t"] = trace_long["timestamp"].dt.floor("min")

    keep = {"trace_err_rate", "trace_p95_latency", "trace_trace_count"}
    trace_long = trace_long[trace_long["kpi"].isin(keep)].copy()

    agg = trace_long.groupby(["entity", "kpi", "t"], as_index=False)["value"].mean()

    z_rows = []
    for (ent, kpi), g in agg.groupby(["entity", "kpi"], sort=False):
        s = g.set_index("t")["value"].reindex(full_index).interpolate(limit=3)
        z = rolling_mad_zscore(s, window=Z_WIN, min_p=MIN_P).abs()
        z_rows.append(pd.DataFrame({"timestamp": full_index, "service": ent, "kpi": kpi, "z_abs": z.values}))

    if not z_rows:
        return pd.DataFrame(columns=["timestamp", "service", "kpi", "z_abs"])
    return pd.concat(z_rows, ignore_index=True)

# =========================
# HYBRID EVAL (v2)
# =========================
def run_hybrid_v2_for_day(day_tag: str) -> dict | None:
    metric_path = PROC / f"day_{day_tag}_metrics.parquet"
    inc_path = REPORTS / f"incidents_allkpis_{day_tag}.csv"  # must exist from your earlier incident builder
    trace_path = PROC / f"day_{day_tag}_trace_kpis.parquet"

    if not metric_path.exists():
        return {"day": day_tag, "status": "skipped_missing_metrics"}
    if not inc_path.exists():
        return {"day": day_tag, "status": "skipped_missing_metric_incidents"}
    if not FAIL_PATH.exists():
        return {"day": day_tag, "status": "skipped_missing_failures_clean"}

    # ensure traces exist
    if not trace_path.exists():
        build_trace_kpis_for_day(day_tag)
    if not trace_path.exists():
        return {"day": day_tag, "status": "skipped_missing_traces"}

    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])
    inc = pd.read_csv(inc_path, parse_dates=["incident_start", "incident_end"])
    traces = pd.read_parquet(trace_path)

    inc["entity"] = inc["entity"].astype(str).str.strip()

    day_min = inc["incident_start"].min()
    day_max = inc["incident_end"].max()
    buf = pd.Timedelta(minutes=10)

    day_fails = fails[(fails["event_start"] >= day_min - buf) & (fails["event_start"] <= day_max + buf)].copy()
    if len(day_fails) == 0:
        return {"day": day_tag, "status": "skipped_no_failures_in_window"}

    if "name" in day_fails.columns:
        day_fails["name"] = day_fails["name"].astype(str).str.strip()
    if "container" in day_fails.columns:
        day_fails["container"] = day_fails["container"].astype(str).str.strip()
    if "fault_description" in day_fails.columns:
        day_fails["fault_description"] = day_fails["fault_description"].astype(str).str.strip()

    ztab = build_trace_z_table(traces, day_min, day_max)

    TRACE_FALLBACK_TYPES = {"network delay", "network loss", "CPU fault"}
    SYMPTOM_KPIS = {"trace_p95_latency", "trace_err_rate"}

    rows = []
    for _, fr in day_fails.iterrows():
        ev_s, ev_e = fr["event_start"], fr["event_end"]
        fdesc = clean_str(fr.get("fault_description")) or ""

        # metric incident match: name + container
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
                metric_lead = abs(raw); metric_delay = 0.0
            else:
                metric_delay = raw; metric_lead = 0.0

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
    out_csv = REPORTS / f"hybrid_eval_v2_{day_tag}.csv"
    safe_write_csv(out, out_csv)

    total = len(out)
    detected = int(out["hybrid_detected"].sum())
    rate = detected / max(total, 1)

    by_type = out.groupby("fault_description")["hybrid_detected"].agg(["count","sum"])
    by_type["rate"] = (by_type["sum"] / by_type["count"]).round(3)

    return {
        "day": day_tag,
        "status": "ok",
        "failures": total,
        "detected": detected,
        "detection_rate": round(rate, 3),
        "by_type": by_type.reset_index()
    }

# =========================
# REPORT GENERATION
# =========================
def write_day_md(day_tag: str) -> Path | None:
    csv_path = REPORTS / f"hybrid_eval_v2_{day_tag}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["event_start","event_end","metric_incident_start"])

    total = len(df)
    detected = int(df["hybrid_detected"].sum())
    rate = detected / max(total, 1)

    by_type = (
        df.groupby("fault_description")["hybrid_detected"]
        .agg(["count","sum"])
        .rename(columns={"count":"failures","sum":"detected"})
    )
    by_type["rate"] = (by_type["detected"] / by_type["failures"]).round(3)

    # pick 1 example each type
    examples = []
    for t in ["network delay","network loss","CPU fault","db connection limit","db  close"]:
        sub = df[df["fault_description"] == t]
        if len(sub): examples.append(sub.iloc[0])

    lines = []
    lines.append(f"# Day Report: {day_tag}\n")
    lines.append("## Summary\n")
    lines.append(f"- Failures evaluated: **{total}**")
    lines.append(f"- Detected (hybrid): **{detected}**")
    lines.append(f"- Detection rate: **{rate:.3f}**\n")
    lines.append("## Detection by Fault Type\n")
    lines.append(by_type.sort_values("rate", ascending=False).to_markdown())

    lines.append("\n\n## Example Evidence\n")
    for r in examples:
        lines.append(f"### {r['fault_description']} — {r['name']} ({r['event_start']} → {r['event_end']})")
        lines.append(f"- Metric detected: **{bool(r.get('metric_detected', False))}**")
        if bool(r.get("metric_detected", False)):
            lines.append(f"  - Matched entity: `{r.get('metric_matched_entity')}`")
            lines.append(f"  - Metric incident start: `{r.get('metric_incident_start')}`")
        lines.append(f"- Trace detected: **{bool(r.get('trace_detected', False))}**")
        if bool(r.get("trace_detected", False)):
            lines.append(f"  - Top services/CMDB entities (peak z): `{r.get('trace_top_services_peakz')}`")
        lines.append("")

    out_md = REPORTS / f"day_report_{day_tag}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md

def write_overall_summary(results: list[dict]) -> None:
    rows = []
    for r in results:
        if r.get("status") == "ok":
            rows.append({
                "day": r["day"],
                "failures": r["failures"],
                "detected": r["detected"],
                "detection_rate": r["detection_rate"],
            })
        else:
            rows.append({"day": r["day"], "failures": None, "detected": None, "detection_rate": None, "status": r["status"]})

    overall = pd.DataFrame(rows).sort_values("day")
    safe_write_csv(overall, REPORTS / "overall_summary.csv")

    ok = overall[overall.get("status").isna() if "status" in overall.columns else overall["detection_rate"].notna()]
    avg_rate = ok["detection_rate"].mean() if len(ok) else None

    md = []
    md.append("# Overall Summary\n")
    md.append(overall.to_markdown(index=False))
    md.append("")
    if avg_rate is not None:
        md.append(f"**Average detection rate over processed days:** {avg_rate:.3f}")
    (REPORTS / "overall_summary.md").write_text("\n".join(md), encoding="utf-8")

# =========================
# MAIN
# =========================
def main():
    zips = sorted(DATA_ROOT.glob("2020_*.zip"))
    days = [z.stem for z in zips]

    print("Found ZIP days:", len(days))
    results = []

    for day in days:
        # build trace KPIs (fast)
        tp = PROC / f"day_{day}_trace_kpis.parquet"
        if not tp.exists():
            out = build_trace_kpis_for_day(day)
            print(day, "trace_kpis:", "ok" if out else "none")

        # hybrid eval v2
        r = run_hybrid_v2_for_day(day)
        results.append(r)
        print(day, "status:", r["status"])

        if r["status"] == "ok":
            md = write_day_md(day)
            if md:
                print(day, "report:", md.name)

    write_overall_summary(results)
    print("Saved overall summary in outputs/reports/overall_summary.*")

if __name__ == "__main__":
    main()
