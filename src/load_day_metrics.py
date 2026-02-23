from __future__ import annotations
from pathlib import Path
from io import BytesIO
import zipfile
import pandas as pd


DATA_ROOT = Path(r"C:\Users\ychet\Desktop\Project\AIOps\AIOps挑战赛数据")
ZIP_PATH = DATA_ROOT / "2020_04_11.zip"

OUT_DIR = Path(r"C:\Users\ychet\Desktop\Project\aiops-incident-intelligence\data\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_from_zip(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    raw = zf.read(member)
    bio = BytesIO(raw)
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(bio, encoding=enc)
        except Exception:
            bio.seek(0)
    bio.seek(0)
    return pd.read_csv(bio, encoding="utf-8", errors="ignore")


def to_datetime_ms(series: pd.Series) -> pd.Series:
    # your timestamps are epoch ms
    return pd.to_datetime(series, unit="ms", errors="coerce")


def load_platform_metrics(zf: zipfile.ZipFile, path: str) -> pd.DataFrame:
    df = read_csv_from_zip(zf, path)
    # expected: itemid, name, bomc_id, timestamp, value, cmdb_id
    out = df[["timestamp", "cmdb_id", "name", "value"]].copy()
    out.rename(columns={"cmdb_id": "entity", "name": "kpi"}, inplace=True)
    out["timestamp"] = to_datetime_ms(out["timestamp"])
    out["source"] = "platform"
    return out


def load_business_metrics_esb(zf: zipfile.ZipFile, path: str) -> pd.DataFrame:
    df = read_csv_from_zip(zf, path)
    # columns: serviceName, startTime, avg_time, num, succee_num, succee_rate
    base = df[["serviceName", "startTime"]].copy()
    base.rename(columns={"serviceName": "entity", "startTime": "timestamp"}, inplace=True)
    base["timestamp"] = to_datetime_ms(base["timestamp"])

    value_cols = [c for c in df.columns if c not in ["serviceName", "startTime"]]
    long = df.melt(
        id_vars=["serviceName", "startTime"],
        value_vars=value_cols,
        var_name="kpi",
        value_name="value",
    )
    long.rename(columns={"serviceName": "entity", "startTime": "timestamp"}, inplace=True)
    long["timestamp"] = to_datetime_ms(long["timestamp"])
    long["source"] = "business"
    return long[["timestamp", "entity", "kpi", "value", "source"]]


def main():
    frames = []

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()

        # Business KPI: esb.csv
        esb_paths = [n for n in names if n.endswith("/业务指标/esb.csv")]
        for p in esb_paths:
            frames.append(load_business_metrics_esb(zf, p))

        # Platform metrics: all csv under /平台指标/
        platform_paths = [n for n in names if "/平台指标/" in n and n.endswith(".csv")]
        for p in platform_paths:
            frames.append(load_platform_metrics(zf, p))

    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics.dropna(subset=["timestamp", "entity", "kpi"])
    # ensure numeric where possible
    metrics["value"] = pd.to_numeric(metrics["value"], errors="coerce")

    # save
    day_tag = ZIP_PATH.stem  # 2020_04_11
    out_parquet = OUT_DIR / f"day_{day_tag}_metrics.parquet"
    out_csv = OUT_DIR / f"day_{day_tag}_metrics.csv"

    metrics.to_parquet(out_parquet, index=False)
    # CSV can be huge; keep if you want:
    # metrics.to_csv(out_csv, index=False, encoding="utf-8")

    print("Saved:", out_parquet)
    print("Rows:", len(metrics))
    print("Unique entities:", metrics["entity"].nunique())
    print("Unique KPIs:", metrics["kpi"].nunique())
    print(metrics.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
