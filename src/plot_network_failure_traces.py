from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
REPORTS = PROJ / "outputs" / "reports"
FIGS = PROJ / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

DAY_TAG = "2020_04_11"
HYBRID = REPORTS / f"hybrid_eval_v2_{DAY_TAG}.csv"
TRACE = PROJ / "data" / "processed" / f"day_{DAY_TAG}_trace_kpis.parquet"

def main():
    df = pd.read_csv(HYBRID, parse_dates=["event_start","event_end"])
    traces = pd.read_parquet(TRACE)
    traces["timestamp"] = pd.to_datetime(traces["timestamp"])

    # pick first network delay failure
    f = df[df["fault_description"] == "network delay"].iloc[0]
    ev_s, ev_e = f["event_start"], f["event_end"]

    # parse top services list
    top = str(f.get("trace_top_services_peakz", "")).split(";")
    top_services = [t.split(":")[0].strip() for t in top if ":" in t][:3]

    w0 = ev_s - pd.Timedelta(minutes=30)
    w1 = ev_e + pd.Timedelta(minutes=30)

    # plot p95 latency
    plt.figure()
    for svc in top_services:
        s = traces[(traces["entity"] == svc) & (traces["kpi"] == "trace_p95_latency")]
        s = s[(s["timestamp"] >= w0) & (s["timestamp"] <= w1)].sort_values("timestamp")
        plt.plot(s["timestamp"], s["value"], label=f"{svc} p95")

    plt.axvspan(ev_s, ev_e, alpha=0.2)
    plt.title(f"Network Delay — Trace p95 latency (top services)")
    plt.legend()
    out1 = FIGS / f"{DAY_TAG}_network_delay_trace_p95.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close()

    # plot error rate
    plt.figure()
    for svc in top_services:
        s = traces[(traces["entity"] == svc) & (traces["kpi"] == "trace_err_rate")]
        s = s[(s["timestamp"] >= w0) & (s["timestamp"] <= w1)].sort_values("timestamp")
        plt.plot(s["timestamp"], s["value"], label=f"{svc} err_rate")

    plt.axvspan(ev_s, ev_e, alpha=0.2)
    plt.title(f"Network Delay — Trace error rate (top services)")
    plt.legend()
    out2 = FIGS / f"{DAY_TAG}_network_delay_trace_err.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close()

    print("Saved:", out1)
    print("Saved:", out2)

if __name__ == "__main__":
    main()
