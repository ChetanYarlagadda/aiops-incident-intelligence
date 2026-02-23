from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
REPORTS = PROJ / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def parse_services(s: str, topn=5):
    if not isinstance(s, str) or not s.strip():
        return []
    parts = [p.strip() for p in s.split(";") if ":" in p]
    svcs = [p.split(":")[0].strip() for p in parts]
    out = []
    for x in svcs:
        if x and x not in out:
            out.append(x)
    return out[:topn]

def main(day: str = "2020_04_11"):
    hybrid = REPORTS / f"hybrid_eval_v2_{day}.csv"
    incidents = REPORTS / f"incidents_allkpis_{day}.csv"
    contrib = REPORTS / f"incident_contrib_allkpis_{day}.csv"

    if not hybrid.exists() or not incidents.exists() or not contrib.exists():
        print("Missing required files for day:", day)
        return

    h = pd.read_csv(hybrid, parse_dates=["event_start","event_end","metric_incident_start"])
    inc = pd.read_csv(incidents, parse_dates=["incident_start","incident_end"])
    con = pd.read_csv(contrib)

    inc["entity"] = inc["entity"].astype(str).str.strip()
    con["entity"] = con["entity"].astype(str).str.strip()
    con["kpi"] = con["kpi"].astype(str).str.strip()

    # Make it easy: for each failure, build an evidence string
    rows = []
    for r in h.itertuples(index=False):
        entity = getattr(r, "metric_matched_entity", None)
        if pd.isna(entity) or not str(entity).strip():
            entity = getattr(r, "name", "")
        entity = str(entity).strip()

        # metric evidence: top KPIs from contributor table for that entity (across all incidents)
        metric_top = (
            con[con["entity"] == entity]
            .groupby("kpi")["peak_abs_z"].max()
            .sort_values(ascending=False)
            .head(5)
        )
        metric_evidence = "; ".join([f"{k}:{v:.2f}" for k, v in metric_top.items()]) if len(metric_top) else ""

        # trace evidence (already computed in hybrid file)
        trace_evidence = getattr(r, "trace_top_services_peakz", "")

        rows.append({
            "failure_index": getattr(r, "failure_index", None),
            "fault_description": getattr(r, "fault_description", None),
            "event_start": getattr(r, "event_start", None),
            "event_end": getattr(r, "event_end", None),
            "matched_metric_entity": entity,
            "metric_top_kpis_peakz": metric_evidence,
            "trace_top_services_peakz": trace_evidence,
            "hybrid_detected": getattr(r, "hybrid_detected", None),
        })

    out = pd.DataFrame(rows)
    out_path = REPORTS / f"root_cause_hints_{day}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved:", out_path)

if __name__ == "__main__":
    # default day; change or pass argument later if you want
    main("2020_04_11")
