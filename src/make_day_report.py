from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
REPORTS = PROJ / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DAY_TAG = "2020_04_11"
HYBRID = REPORTS / f"hybrid_eval_v2_{DAY_TAG}.csv"
OUT_MD = REPORTS / f"day_report_{DAY_TAG}.md"

def main():
    df = pd.read_csv(HYBRID, parse_dates=["event_start","event_end","metric_incident_start"])

    total = len(df)
    detected = int(df["hybrid_detected"].sum()) if "hybrid_detected" in df.columns else total
    rate = detected / max(total, 1)

    by_type = (
        df.groupby("fault_description")["hybrid_detected"]
        .agg(["count","sum"])
        .rename(columns={"count":"failures","sum":"detected"})
    )
    by_type["rate"] = (by_type["detected"] / by_type["failures"]).round(3)

    # Examples: pick one from each type
    examples = []
    for ftype in ["network delay","network loss","CPU fault","db connection limit","db  close"]:
        sub = df[df["fault_description"] == ftype]
        if len(sub):
            examples.append(sub.iloc[0])

    lines = []
    lines.append(f"# AIOps Incident Intelligence — Day Report ({DAY_TAG})\n")
    lines.append("## Summary\n")
    lines.append(f"- Failures evaluated: **{total}**\n")
    lines.append(f"- Detected (hybrid): **{detected}**\n")
    lines.append(f"- Detection rate: **{rate:.3f}**\n")
    lines.append("\n## Detection by Fault Type\n")
    lines.append(by_type.sort_values("rate", ascending=False).to_markdown())

    lines.append("\n\n## How Detection Works\n")
    lines.append(
        "- **Metrics path:** anomaly windows across all KPIs per entity → merged into entity-level incidents.\n"
        "- **Trace path:** per-service KPIs (**p95 latency**, **error rate**, **call volume**) → robust z-score anomalies.\n"
        "- **Hybrid rule:** if metric incident overlaps failure window → detected; else trace symptom spike (for network + CPU) → detected.\n"
    )

    lines.append("\n## Example Detections (Evidence)\n")
    for r in examples:
        lines.append(f"### {r['fault_description']} — {r['name']} ({r['event_start']} → {r['event_end']})\n")
        lines.append(f"- Metric detected: **{bool(r.get('metric_detected', False))}**")
        if bool(r.get("metric_detected", False)):
            lines.append(f"  - Matched entity: `{r.get('metric_matched_entity')}`")
            lines.append(f"  - Metric incident start: `{r.get('metric_incident_start')}`")
        lines.append(f"- Trace detected: **{bool(r.get('trace_detected', False))}**")
        if bool(r.get("trace_detected", False)):
            lines.append(f"  - Top services (peak z): `{r.get('trace_top_services_peakz')}`")
            lines.append(f"  - Top peak z: `{r.get('trace_top_peak_z')}`")
        lines.append("")

    lines.append("\n## Output Artifacts\n")
    lines.append(f"- `outputs/reports/hybrid_eval_v2_{DAY_TAG}.csv` — per-failure detection + evidence\n")
    lines.append(f"- `outputs/reports/incidents_allkpis_{DAY_TAG}.csv` — metric incidents per entity\n")
    lines.append(f"- `data/processed/day_{DAY_TAG}_trace_kpis.parquet` — trace-derived KPIs\n")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", OUT_MD)

if __name__ == "__main__":
    main()
