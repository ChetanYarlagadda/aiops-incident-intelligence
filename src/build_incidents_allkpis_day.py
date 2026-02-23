from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
PROC = PROJ / "data" / "processed"
REPORTS = PROJ / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

Z_THRESH = 3.5          # robust z threshold
ROLL_WIN = 15
MIN_P = 5
POINT_GAP_MIN = 1       # anomaly points within this gap → same KPI window
INCIDENT_GAP_MIN = 2    # KPI windows within this gap → same entity incident
DAY_HOURS = 6           # this dataset's day window is typically first ~6h

def rolling_mad_zscore(x: pd.Series, window: int = 15, min_p: int = 5) -> pd.Series:
    med = x.rolling(window, min_periods=min_p).median()
    mad = (x - med).abs().rolling(window, min_periods=min_p).median().replace(0, np.nan)
    z = 0.6745 * (x - med) / mad
    return z.fillna(0)

def _maybe_shift_to_local(df: pd.DataFrame, day_tag: str) -> pd.DataFrame:
    """
    Auto-fix the UTC vs local mismatch:
    If metrics begin on the PREVIOUS date (typical when timestamps are UTC), shift +8 hours.
    """
    day_date = pd.Timestamp(day_tag.replace("_", "-")).date()
    min_date = df["timestamp"].min().date()
    if min_date < day_date:
        df = df.copy()
        df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=8)
    return df

def _extract_windows(times: pd.DatetimeIndex, mask: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out = []
    start_i = idx[0]
    prev_i = idx[0]
    for i in idx[1:]:
        if (times[i] - times[prev_i]) <= pd.Timedelta(minutes=POINT_GAP_MIN):
            prev_i = i
        else:
            out.append((times[start_i], times[prev_i], int(prev_i - start_i + 1)))
            start_i = i
            prev_i = i
    out.append((times[start_i], times[prev_i], int(prev_i - start_i + 1)))
    return out

def build_incidents_for_day(day_tag: str) -> dict:
    metric_path = PROC / f"day_{day_tag}_metrics.parquet"
    if not metric_path.exists():
        return {"day": day_tag, "status": "missing_metrics_parquet"}

    df = pd.read_parquet(metric_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["entity"] = df["entity"].astype(str).str.strip()
    df["kpi"] = df["kpi"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "entity", "kpi", "value"])

    # auto-align to local if needed
    df = _maybe_shift_to_local(df, day_tag)

    # define day window (first 6 hours)
    day_start = pd.Timestamp(day_tag.replace("_", "-") + " 00:00:00")
    day_end = day_start + pd.Timedelta(hours=DAY_HOURS)
    df = df[(df["timestamp"] >= day_start) & (df["timestamp"] <= day_end)].copy()

    if df.empty:
        return {"day": day_tag, "status": "no_metrics_in_day_window"}

    # compress to 1-min mean (already mostly 1-min, but safe)
    df["t"] = df["timestamp"].dt.floor("min")
    df = df.groupby(["entity", "kpi", "t"], as_index=False)["value"].mean()
    df = df.rename(columns={"t": "timestamp"})

    full_index = pd.date_range(day_start, day_end, freq="1min")

    # build KPI anomaly windows
    signal_windows = []
    contrib_rows = []
    incidents = []

    # group loop is faster than groupby.apply for 2k series
    for (ent, kpi), g in df.groupby(["entity", "kpi"], sort=False):
        s = g.set_index("timestamp")["value"].reindex(full_index)
        s = s.interpolate(limit=3)

        z = rolling_mad_zscore(s, window=ROLL_WIN, min_p=MIN_P).abs()
        mask = (z >= Z_THRESH).to_numpy()

        wins = _extract_windows(full_index, mask)
        for (ws, we, npts) in wins:
            peak = float(z[(full_index >= ws) & (full_index <= we)].max())
            signal_windows.append({
                "entity": ent,
                "kpi": kpi,
                "win_start": ws,
                "win_end": we,
                "n_points": npts,
                "peak_abs_z": peak,
            })

    if not signal_windows:
        # still save empty artifacts so pipeline doesn't crash
        sw = pd.DataFrame(columns=["entity","kpi","win_start","win_end","n_points","peak_abs_z"])
        sw.to_csv(REPORTS / f"signal_windows_allkpis_{day_tag}.csv", index=False, encoding="utf-8")
        inc_df = pd.DataFrame(columns=["incident_id","entity","incident_start","incident_end"])
        inc_df.to_csv(REPORTS / f"incidents_allkpis_{day_tag}.csv", index=False, encoding="utf-8")
        contrib = pd.DataFrame(columns=["incident_id","entity","kpi","peak_abs_z"])
        contrib.to_csv(REPORTS / f"incident_contrib_allkpis_{day_tag}.csv", index=False, encoding="utf-8")
        return {"day": day_tag, "status": "ok_but_no_anomalies", "series": int(df.groupby(["entity","kpi"]).ngroups)}

    sw = pd.DataFrame(signal_windows).sort_values(["entity", "win_start", "win_end"]).reset_index(drop=True)
    sw.to_csv(REPORTS / f"signal_windows_allkpis_{day_tag}.csv", index=False, encoding="utf-8")

    # merge KPI windows → entity incidents
    incident_id = 0
    for ent, g in sw.groupby("entity", sort=False):
        g = g.sort_values("win_start")
        cur_s = None
        cur_e = None
        members = []  # (kpi, peak)
        for r in g.itertuples(index=False):
            ws, we = r.win_start, r.win_end
            if cur_s is None:
                cur_s, cur_e = ws, we
                members = [(r.kpi, float(r.peak_abs_z))]
            else:
                if ws <= (cur_e + pd.Timedelta(minutes=INCIDENT_GAP_MIN)):
                    cur_e = max(cur_e, we)
                    members.append((r.kpi, float(r.peak_abs_z)))
                else:
                    incident_id += 1
                    incidents.append({"incident_id": incident_id, "entity": ent, "incident_start": cur_s, "incident_end": cur_e})
                    # top KPI evidence
                    top = sorted(members, key=lambda x: x[1], reverse=True)[:10]
                    for k, p in top:
                        contrib_rows.append({"incident_id": incident_id, "entity": ent, "kpi": k, "peak_abs_z": p})

                    cur_s, cur_e = ws, we
                    members = [(r.kpi, float(r.peak_abs_z))]

        # flush last
        if cur_s is not None:
            incident_id += 1
            incidents.append({"incident_id": incident_id, "entity": ent, "incident_start": cur_s, "incident_end": cur_e})
            top = sorted(members, key=lambda x: x[1], reverse=True)[:10]
            for k, p in top:
                contrib_rows.append({"incident_id": incident_id, "entity": ent, "kpi": k, "peak_abs_z": p})

    inc_df = pd.DataFrame(incidents).sort_values(["incident_start","entity"]).reset_index(drop=True)
    inc_df.to_csv(REPORTS / f"incidents_allkpis_{day_tag}.csv", index=False, encoding="utf-8")

    contrib = pd.DataFrame(contrib_rows).sort_values(["incident_id","peak_abs_z"], ascending=[True, False])
    contrib.to_csv(REPORTS / f"incident_contrib_allkpis_{day_tag}.csv", index=False, encoding="utf-8")

    return {
        "day": day_tag,
        "status": "ok",
        "series": int(df.groupby(["entity","kpi"]).ngroups),
        "incidents": len(inc_df),
        "signal_windows": len(sw),
        "day_window": f"{day_start} -> {day_end}",
    }

if __name__ == "__main__":
    day = "2020_04_20"
    print(build_incidents_for_day(day))
