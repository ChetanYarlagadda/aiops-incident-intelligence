from __future__ import annotations
from pathlib import Path
import zipfile
from io import BytesIO
import pandas as pd


DATA_ROOT = Path(r"C:\Users\ychet\Desktop\Project\AIOps\AIOps挑战赛数据")
ZIP_PATH = DATA_ROOT / "2020_04_11.zip"


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


def main():
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()

        # grab key files
        targets = []

        # Business KPI
        targets += [n for n in names if n.endswith("/业务指标/esb.csv")]

        # Platform metrics (pick a few)
        platform_files = [n for n in names if "/平台指标/" in n and n.endswith(".csv")]
        targets += platform_files[:3]

        # Trace metrics (pick a few)
        trace_files = [n for n in names if "/调用链指标/" in n and n.endswith(".csv")]
        targets += trace_files[:2]

        print("ZIP:", ZIP_PATH.name)
        print("Targets:")
        for t in targets:
            print(" -", t)

        for t in targets:
            print("\n=== FILE ===", t)
            df = read_csv_from_zip(zf, t)
            print("Shape:", df.shape)
            print("Columns:", list(df.columns))
            print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
