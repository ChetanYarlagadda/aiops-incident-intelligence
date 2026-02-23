# app.py (Demo-mode enabled)
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# PATHS (demo-aware)
# -------------------------
PROJ = Path(__file__).resolve().parent

REPORTS = PROJ / "outputs" / "reports"
PROC = PROJ / "data" / "processed"

DEMO = PROJ / "demo_data"
USE_DEMO = DEMO.exists() and any(DEMO.glob("hybrid_eval_v2_.csv"))



BASE_REPORTS = DEMO if USE_DEMO else REPORTS
BASE_DATA = DEMO if USE_DEMO else PROC

# -------------------------
# Streamlit page settings
# -------------------------
st.set_page_config(page_title="AIOps Incident Intelligence", layout="wide")
st.title("AIOps Incident Intelligence Dashboard")

if USE_DEMO:
    st.info("Running in DEMO mode (reading artifacts from demo_data/).")
else:
    st.caption("Running in LOCAL mode (reading artifacts from outputs/ and data/).")


# -------------------------
# Helpers
# -------------------------
def list_days() -> list[str]:
    files = sorted(BASE_REPORTS.glob("hybrid_eval_v2_*.csv"))
    return [f.stem.replace("hybrid_eval_v2_", "") for f in files]


def load_hybrid(day: str) -> pd.DataFrame:
    p = BASE_REPORTS / f"hybrid_eval_v2_{day}.csv"
    df = pd.read_csv(p, parse_dates=["event_start", "event_end", "metric_incident_start"])
    # normalize strings
    for c in ["fault_description", "name", "container", "metric_matched_entity", "trace_top_services_peakz"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].str.lower().isin(["nan", "none"]), c] = ""
    return df


def load_incidents(day: str) -> pd.DataFrame | None:
    p = BASE_REPORTS / f"incidents_allkpis_{day}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["incident_start", "incident_end"])
    df["entity"] = df["entity"].astype(str).str.strip()
    return df


def load_incident_contrib(day: str) -> pd.DataFrame | None:
    p = BASE_REPORTS / f"incident_contrib_allkpis_{day}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "incident_id" in df.columns:
        df["incident_id"] = pd.to_numeric(df["incident_id"], errors="coerce")
    df["entity"] = df["entity"].astype(str).str.strip()
    df["kpi"] = df["kpi"].astype(str).str.strip()
    df["peak_abs_z"] = pd.to_numeric(df["peak_abs_z"], errors="coerce")
    return df.dropna(subset=["entity", "kpi", "peak_abs_z"])


def load_trace(day: str) -> pd.DataFrame | None:
    p = BASE_DATA / f"day_{day}_trace_kpis.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["entity"] = df["entity"].astype(str).str.strip()
    df["kpi"] = df["kpi"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["timestamp", "entity", "kpi", "value"])


def load_metrics(day: str) -> pd.DataFrame | None:
    p = BASE_DATA / f"day_{day}_metrics.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["entity"] = df["entity"].astype(str).str.strip()
    df["kpi"] = df["kpi"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "entity", "kpi", "value"])

    # Align timezone if metrics start on previous date (common: UTC → local +8)
    day_date = pd.Timestamp(day.replace("_", "-")).date()
    if df["timestamp"].min().date() < day_date:
        df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=8)

    return df


def parse_top_services(s: str, topn: int = 3) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    parts = [p.strip() for p in s.split(";") if ":" in p]
    svcs = [p.split(":")[0].strip() for p in parts]
    out: list[str] = []
    for x in svcs:
        if x and x not in out:
            out.append(x)
    return out[:topn]


def plot_trace_overlay(traces: pd.DataFrame, services: list[str], kpi: str, ev_s, ev_e, title: str):
    plt.figure()
    w0 = ev_s - pd.Timedelta(minutes=30)
    w1 = ev_e + pd.Timedelta(minutes=30)

    plotted = 0
    for svc in services:
        s = traces[(traces["entity"] == svc) & (traces["kpi"] == kpi)]
        s = s[(s["timestamp"] >= w0) & (s["timestamp"] <= w1)].sort_values("timestamp")
        if len(s):
            plt.plot(s["timestamp"], s["value"], label=f"{svc}")
            plotted += 1

    plt.axvspan(ev_s, ev_e, alpha=0.2)
    plt.title(title)
    if plotted:
        plt.legend()
    st.pyplot(plt.gcf())
    plt.close()


def plot_metric_overlay(metrics: pd.DataFrame, entity: str, kpi: str, ev_s, ev_e, title: str):
    plt.figure()
    w0 = ev_s - pd.Timedelta(minutes=30)
    w1 = ev_e + pd.Timedelta(minutes=30)

    s = metrics[(metrics["entity"] == entity) & (metrics["kpi"] == kpi)]
    s = s[(s["timestamp"] >= w0) & (s["timestamp"] <= w1)].sort_values("timestamp")

    if len(s):
        plt.plot(s["timestamp"], s["value"])
    plt.axvspan(ev_s, ev_e, alpha=0.2)
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.close()


def find_overlapping_incident_id(inc: pd.DataFrame, entity: str, ev_s, ev_e):
    if inc is None or len(inc) == 0:
        return None
    sub = inc[inc["entity"] == entity].copy()
    if len(sub) == 0:
        return None
    ov = sub[(sub["incident_start"] <= ev_e) & (sub["incident_end"] >= ev_s)].sort_values("incident_start")
    if len(ov) == 0:
        return None
    return ov.iloc[0]["incident_id"]


def get_metric_root_cause_hints(contrib: pd.DataFrame, entity: str, incident_id=None, topn: int = 5):
    if contrib is None or len(contrib) == 0:
        return []
    if incident_id is not None and "incident_id" in contrib.columns:
        sub = contrib[(contrib["entity"] == entity) & (contrib["incident_id"] == incident_id)].copy()
    else:
        sub = contrib[contrib["entity"] == entity].copy()
    if len(sub) == 0:
        return []
    top = (sub.groupby("kpi")["peak_abs_z"].max()
           .sort_values(ascending=False)
           .head(topn))
    return [(k, float(v)) for k, v in top.items()]


# -------------------------
# Sidebar
# -------------------------
days = list_days()
if not days:
    st.error("No hybrid_eval_v2_*.csv found. For Streamlit Cloud, add demo_data/ with hybrid_eval_v2_*.csv.")
    st.stop()

day = st.sidebar.selectbox("Select day", days, index=0)
df = load_hybrid(day)

faults = sorted([x for x in df["fault_description"].dropna().unique().tolist() if x])
fault_filter = st.sidebar.multiselect("Filter fault types", faults, default=faults)

show_only_trace = st.sidebar.checkbox("Show only trace-detected failures", value=False)
show_only_metric = st.sidebar.checkbox("Show only metric-detected failures", value=False)
show_metric_plots = st.sidebar.checkbox("Show metric plots", value=True)
show_trace_plots = st.sidebar.checkbox("Show trace plots", value=True)

view = df[df["fault_description"].isin(fault_filter)].copy()
if show_only_trace:
    view = view[view["trace_detected"] == True]
if show_only_metric:
    view = view[view["metric_detected"] == True]

# -------------------------
# Summary cards
# -------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Failures", len(view))
with c2:
    st.metric("Detected", int(view["hybrid_detected"].sum()) if len(view) else 0)
with c3:
    st.metric("Detection rate", round(float(view["hybrid_detected"].mean()), 3) if len(view) else 0.0)

# -------------------------
# Table + row selection
# -------------------------
st.subheader("Failures (select row index below)")
st.dataframe(view.reset_index(drop=True), use_container_width=True, height=320)

if len(view) == 0:
    st.stop()

sel = st.number_input("Row index to inspect (0-based)", min_value=0, max_value=len(view) - 1, value=0, step=1)
row = view.reset_index(drop=True).iloc[int(sel)]

ev_s = row["event_start"]
ev_e = row["event_end"]

st.subheader("Selected Failure Details")
st.write(
    {
        "failure_index": row.get("failure_index"),
        "fault_description": row.get("fault_description"),
        "name": row.get("name"),
        "container": row.get("container"),
        "event_start": str(ev_s),
        "event_end": str(ev_e),
        "metric_detected": bool(row.get("metric_detected")),
        "metric_matched_entity": row.get("metric_matched_entity"),
        "metric_incident_start": str(row.get("metric_incident_start")),
        "trace_detected": bool(row.get("trace_detected")),
        "trace_top_services_peakz": row.get("trace_top_services_peakz"),
        "trace_top_peak_z": row.get("trace_top_peak_z"),
        "hybrid_detected": bool(row.get("hybrid_detected")),
    }
)

# -------------------------
# Root Cause Hints Panel
# -------------------------
st.subheader("Root Cause Hints")
inc = load_incidents(day)
contrib = load_incident_contrib(day)

entity_for_rca = row.get("metric_matched_entity")
if not isinstance(entity_for_rca, str) or not entity_for_rca.strip():
    entity_for_rca = row.get("name", "")
entity_for_rca = str(entity_for_rca).strip()

incident_id = None
if inc is not None and len(inc):
    incident_id = find_overlapping_incident_id(inc, entity_for_rca, ev_s, ev_e)

metric_hints = get_metric_root_cause_hints(contrib, entity_for_rca, incident_id=incident_id, topn=5)

trace_hint_raw = str(row.get("trace_top_services_peakz", "")).strip()
trace_services = parse_top_services(trace_hint_raw, topn=5)

cA, cB = st.columns(2)
with cA:
    st.markdown("**Metric suspects (top KPIs by peak anomaly score)**")
    if metric_hints:
        st.table(pd.DataFrame(metric_hints, columns=["KPI", "Peak |z|"]))
        if incident_id is not None:
            st.caption(f"Based on incident_id = {incident_id} for entity `{entity_for_rca}`")
        else:
            st.caption(f"Based on entity `{entity_for_rca}` (no overlapping incident found)")
    else:
        st.info("No metric KPI suspects available for this entity/incident (missing incident_contrib file).")

with cB:
    st.markdown("**Trace suspects (top services by peak anomaly score)**")
    if trace_hint_raw and trace_services:
        st.write(trace_hint_raw)
    else:
        st.info("No trace suspects for this failure (likely metric-detected).")

# -------------------------
# Load data needed for plots
# -------------------------
traces = load_trace(day) if show_trace_plots else None
metrics = load_metrics(day) if show_metric_plots else None

# -------------------------
# TRACE PLOTS (always show: fallback to top-volume services)
# -------------------------
if show_trace_plots:
    if traces is None:
        st.info("Trace KPI parquet not found. In demo mode, include day_<DAY>_trace_kpis.parquet in demo_data/.")
    else:
        services = parse_top_services(str(row.get("trace_top_services_peakz", "")), topn=3)

        # Fallback: if no evidence string, pick top 3 services by volume near failure window
        if not services:
            w0 = ev_s - pd.Timedelta(minutes=30)
            w1 = ev_e + pd.Timedelta(minutes=30)
            tmp = traces[(traces["timestamp"] >= w0) & (traces["timestamp"] <= w1) & (traces["kpi"] == "trace_trace_count")]
            if len(tmp):
                services = (
                    tmp.groupby("entity")["value"].sum()
                    .sort_values(ascending=False).head(3).index.tolist()
                )
                st.info(f"No trace evidence listed (metric-detected). Showing top services by trace volume: {services}")
            else:
                st.info("No trace data available in this window to plot.")
                services = []

        if services:
            st.subheader("Trace overlays (services)")
            plot_trace_overlay(traces, services, "trace_p95_latency", ev_s, ev_e, "Trace p95 latency")
            plot_trace_overlay(traces, services, "trace_err_rate", ev_s, ev_e, "Trace error rate")
            plot_trace_overlay(traces, services, "trace_trace_count", ev_s, ev_e, "Trace call volume")

# -------------------------
# METRIC PLOTS (top 3 KPIs by variance in window)
# -------------------------
if show_metric_plots:
    if metrics is None:
        st.info("Metrics parquet not found. In demo mode, include day_<DAY>_metrics.parquet in demo_data/.")
    else:
        st.subheader("Metric overlays (entity KPIs)")
        ent = row.get("metric_matched_entity")
        if not isinstance(ent, str) or not ent.strip():
            ent = row.get("name", "")
        ent = str(ent).strip()

        w0 = ev_s - pd.Timedelta(minutes=30)
        w1 = ev_e + pd.Timedelta(minutes=30)

        ent_df = metrics[(metrics["entity"] == ent) & (metrics["timestamp"] >= w0) & (metrics["timestamp"] <= w1)].copy()
        if len(ent_df) == 0:
            st.info(f"No metrics found for entity `{ent}` in this window.")
        else:
            top_kpis = (
                ent_df.groupby("kpi")["value"].var()
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )
            if not top_kpis:
                st.info(f"No KPI variance found to plot for `{ent}`.")
            else:
                for k in top_kpis:
                    plot_metric_overlay(metrics, ent, k, ev_s, ev_e, f"{ent} — {k}")

st.caption("Tip: For Streamlit Cloud deployment, commit a small demo_data/ bundle (one day) with the required CSV/Parquet artifacts.")
