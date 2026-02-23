from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd


PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
OUT_DIR = PROJ / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY_TAG = "2020_04_11"

FAIL_PATH = PROJ / "data" / "processed" / "failures_clean.csv"
INC_PATH  = OUT_DIR / f"incidents_allkpis_{DAY_TAG}.csv"   # produced by build_incidents_all_kpis.py

OUT_CSV = OUT_DIR / f"failure_vs_incidents_related_{DAY_TAG}.csv"


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


def clean_str(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s and s.lower() != "nan" else None


def main():
    fails = pd.read_csv(FAIL_PATH, parse_dates=["event_start", "event_end"])
    inc = pd.read_csv(INC_PATH, parse_dates=["incident_start", "incident_end"])

    # Normalize entity strings
    if "name" in fails.columns:
        fails["name"] = fails["name"].astype(str).str.strip()
    if "container" in fails.columns:
        fails["container"] = fails["container"].astype(str).str.strip()
    inc["entity"] = inc["entity"].astype(str).str.strip()

    # Define day window from incidents file (safer than metrics here)
    day_min = inc["incident_start"].min()
    day_max = inc["incident_end"].max()
    buf = pd.Timedelta(minutes=10)

    day_fails = fails[
        (fails["event_start"] >= day_min - buf) &
        (fails["event_start"] <= day_max + buf)
    ].copy()

    print("Incident window:", day_min, "->", day_max)
    print("Failures in this window:", len(day_fails))

    rows = []

    for _, fr in day_fails.iterrows():
        ev_s, ev_e = fr["event_start"], fr["event_end"]

        # related entities: name + container (+ optional cmdb_id if you ever add it)
        candidates = []
        nm = clean_str(fr.get("name"))
        ct = clean_str(fr.get("container"))

        if nm: candidates.append(nm)
        if ct and ct not in candidates: candidates.append(ct)

        # (optional) if your cleaned file has something like "cmdb_id"
        cm = clean_str(fr.get("cmdb_id"))
        if cm and cm not in candidates: candidates.append(cm)

        best_hit = False
        best_entity = None
        best_inc_start = None

        for ent in candidates:
            incs = inc[inc["entity"] == ent]
            for _, ir in incs.iterrows():
                if overlap(ir["incident_start"], ir["incident_end"], ev_s, ev_e):
                    if (best_inc_start is None) or (ir["incident_start"] < best_inc_start):
                        best_hit = True
                        best_entity = ent
                        best_inc_start = ir["incident_start"]

        delay_min = None
        lead_min = None
        if best_hit and best_inc_start is not None:
            raw = (best_inc_start - ev_s).total_seconds() / 60.0
            if raw < 0:
                lead_min = abs(raw)
                delay_min = 0.0
            else:
                delay_min = raw
                lead_min = 0.0

        rows.append({
            "failure_index": fr.get("index"),
            "fault_description": fr.get("fault_description"),
            "name": nm,
            "container": ct,
            "event_start": ev_s,
            "event_end": ev_e,
            "related_entities_checked": ";".join(candidates),
            "incident_detected": best_hit,
            "matched_entity": best_entity,
            "incident_start": best_inc_start,
            "delay_min": delay_min,
            "lead_min": lead_min,
        })

    out = pd.DataFrame(rows)
    out_path = safe_write_csv(out, OUT_CSV)

    # Summary
    total = len(out)
    detected = int(out["incident_detected"].sum())
    delays = out.loc[out["incident_detected"] & out["delay_min"].notna(), "delay_min"]
    leads  = out.loc[out["incident_detected"] & out["lead_min"].notna(), "lead_min"]

    print("\n=== Related-Entity Incident Summary ===")
    print("Failures:", total)
    print("Detected:", detected)
    print("Detection rate:", round(detected / max(total, 1), 3))
    if len(delays):
        print("Avg delay (min):", round(float(delays.mean()), 2), " Median:", round(float(delays.median()), 2))
    if len(leads):
        print("Avg lead (min):", round(float(leads.mean()), 2), " Median:", round(float(leads.median()), 2))

    # Detection by type
    if "fault_description" in out.columns:
        by_type = (
            out.groupby("fault_description", dropna=False)["incident_detected"]
            .agg(["count", "sum"])
            .rename(columns={"count": "failures", "sum": "detected"})
        )
        by_type["detection_rate"] = (by_type["detected"] / by_type["failures"]).round(3)
        by_type = by_type.sort_values(["detection_rate", "failures"], ascending=[False, False])
        print("\n=== Detection by fault_description (related entities) ===")
        print(by_type.to_string())

    # show missed
    missed = out[out["incident_detected"] == False].copy()
    show_cols = ["failure_index", "fault_description", "name", "container", "event_start", "event_end"]
    show_cols = [c for c in show_cols if c in missed.columns]
    print("\n=== Missed failures (first 20) ===")
    print(missed[show_cols].head(20).to_string(index=False) if len(missed) else "None")

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
