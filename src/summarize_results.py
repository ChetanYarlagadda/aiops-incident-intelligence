from pathlib import Path
import pandas as pd

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
DAY_TAG = "2020_04_11"

FAIL_MAP = PROJ / "outputs" / "reports" / f"failure_vs_incidents_allkpis_{DAY_TAG}.csv"
INCIDENTS = PROJ / "outputs" / "reports" / f"incidents_allkpis_{DAY_TAG}.csv"

def main():
    fmap = pd.read_csv(FAIL_MAP, parse_dates=["event_start", "event_end", "incident_start"])
    inc = pd.read_csv(INCIDENTS, parse_dates=["incident_start", "incident_end"])

    print("\n=== Overall Failure Detection ===")
    total = len(fmap)
    detected = int(fmap["incident_detected"].sum())
    print("Failures:", total)
    print("Detected:", detected)
    print("Detection rate:", round(detected / max(total, 1), 3))

    if "delay_min" in fmap.columns:
        delays = fmap.loc[fmap["incident_detected"] & fmap["delay_min"].notna(), "delay_min"]
        if len(delays):
            print("Avg delay (min):", round(float(delays.mean()), 2), "Median:", round(float(delays.median()), 2))

    if "lead_min" in fmap.columns:
        leads = fmap.loc[fmap["incident_detected"] & fmap["lead_min"].notna(), "lead_min"]
        if len(leads):
            print("Avg lead (min):", round(float(leads.mean()), 2), "Median:", round(float(leads.median()), 2))

    # Detection by fault type
    if "fault_description" in fmap.columns:
        by_type = (
            fmap.groupby("fault_description", dropna=False)["incident_detected"]
            .agg(["count", "sum"])
            .rename(columns={"count": "failures", "sum": "detected"})
        )
        by_type["detection_rate"] = (by_type["detected"] / by_type["failures"]).round(3)
        by_type = by_type.sort_values(["detection_rate", "failures"], ascending=[False, False])

        print("\n=== Detection by fault_description ===")
        print(by_type.to_string())

    # Missed failures
    missed = fmap[fmap["incident_detected"] == False].copy()
    print("\n=== Missed Failures (showing up to 20) ===")
    show_cols = [c for c in ["failure_index", "entity", "fault_description", "event_start", "event_end"] if c in missed.columns]
    print(missed[show_cols].head(20).to_string(index=False) if len(missed) else "None")

    # Noisiest entities (most incidents)
    noisy = inc.groupby("entity").size().sort_values(ascending=False).head(15)
    print("\n=== Top 15 entities by #incidents (potential noise) ===")
    print(noisy.to_string())

if __name__ == "__main__":
    main()
