from __future__ import annotations
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

def _read_csv_auto(file_obj):
    """Try utf-8, fallback to gbk (common for CN datasets)."""
    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="gbk", errors="ignore")

def _agg_chunk_to_min(df: pd.DataFrame, entity_col: str, time_col: str, kpi_col: str, value_col: str,
                     source: str) -> pd.DataFrame:
    df = df[[entity_col, time_col, kpi_col, value_col]].copy()

    df[entity_col] = df[entity_col].astype(str).str.strip()
    df[kpi_col] = df[kpi_col].astype(str).str.strip()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # timestamps are ms in this dataset
    ts = pd.to_datetime(df[time_col], unit="ms", errors="coerce")
    df["timestamp"] = ts.dt.floor("min")

    df = df.dropna(subset=["timestamp", entity_col, kpi_col, value_col])

    # 1-min aggregation (mean)
    out = (
        df.groupby([entity_col, kpi_col, "timestamp"], as_index=False)[value_col]
          .mean()
          .rename(columns={entity_col: "entity", kpi_col: "kpi", value_col: "value"})
    )
    out["source"] = source
    return out

def build_day_metrics_from_zip(
    zip_path: Path,
    day_tag: str,
    out_parquet: Path,
    chunksize: int = 250_000
) -> Path:
    """
    Produces a LONG metrics table:
      columns = [timestamp, entity, kpi, value, source]
    where timestamp is UTC-ish (raw ms → datetime). Downstream incident builder will auto-align to local.
    """
    business_dir = f"{day_tag}/业务指标/"
    platform_dir = f"{day_tag}/平台指标/"

    parts = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]

        business_files = [m for m in members if m.startswith(business_dir)]
        platform_files = [m for m in members if m.startswith(platform_dir)]

        # -------- business metrics (esb.csv etc.) --------
        for m in business_files:
            with zf.open(m) as f:
                df = _read_csv_auto(f)

            # handle known business schema (serviceName/startTime + metrics)
            if {"serviceName", "startTime"}.issubset(df.columns):
                metric_cols = [c for c in df.columns if c not in {"serviceName", "startTime"}]
                for col in metric_cols:
                    tmp = df[["serviceName", "startTime", col]].copy()
                    tmp = tmp.rename(columns={"serviceName": "entity", "startTime": "startTime", col: "value"})
                    tmp["kpi"] = col
                    tmp = _agg_chunk_to_min(tmp, "entity", "startTime", "kpi", "value", source="business")
                    parts.append(tmp)

        # -------- platform metrics (os_linux/db_oracle/mw_redis/dcos_container...) --------
        for m in platform_files:
            # platform files are big → read in chunks
            with zf.open(m) as f:
                try:
                    it = pd.read_csv(f, chunksize=chunksize)
                except UnicodeDecodeError:
                    f.seek(0)
                    it = pd.read_csv(f, chunksize=chunksize, encoding="gbk", errors="ignore")

                for chunk in it:
                    needed = {"timestamp", "value", "cmdb_id", "name"}
                    if not needed.issubset(chunk.columns):
                        continue
                    tmp = chunk[["cmdb_id", "timestamp", "name", "value"]].copy()
                    tmp = tmp.rename(columns={"cmdb_id": "entity", "name": "kpi"})
                    tmp = _agg_chunk_to_min(tmp, "entity", "timestamp", "kpi", "value", source="platform")
                    parts.append(tmp)

    if not parts:
        raise RuntimeError(f"No metrics parsed from {zip_path.name}. Check ZIP paths and CSV schemas.")

    df_all = pd.concat(parts, ignore_index=True)

    # final merge in case of overlapping chunks/files
    df_all = (
        df_all.groupby(["timestamp", "entity", "kpi", "source"], as_index=False)["value"]
              .mean()
              .sort_values(["timestamp", "entity", "kpi"])
              .reset_index(drop=True)
    )

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_parquet, index=False)
    return out_parquet

if __name__ == "__main__":
    # quick manual test
    PROJ = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence")
    DATA_ROOT = Path(r"C:\Users\ychet\Desktop\Project\AIOps\AIOps挑战赛数据")
    day = "2020_04_20"
    zip_path = DATA_ROOT / f"{day}.zip"
    out = PROJ / "data" / "processed" / f"day_{day}_metrics.parquet"
    p = build_day_metrics_from_zip(zip_path, day, out)
    print("Saved:", p)
