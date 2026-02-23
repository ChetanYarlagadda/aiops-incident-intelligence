# AIOps Incident Intelligence (Metrics + Traces)

End-to-end AIOps pipeline that detects production incidents using platform metrics + distributed traces and provides explainable root-cause hints.

## What it does
- Ingests KPI metrics + trace logs
- Builds 1-minute entity–KPI time series
- Detects anomalies using robust MAD z-score
- Merges anomalies into incident windows
- Hybrid detection using metrics + trace symptoms
- Streamlit dashboard to explore failures and evidence

## How to run (local)
1) Install deps  
   `pip install -r requirements.txt`

2) Run pipeline (optional)  
   `python src/batch_full_pipeline_all_days.py`  
   `python src/batch_run_all_days.py`

3) Run dashboard  
   `streamlit run app.py`

## Notes
- Dataset files are not included in the repo.
- Outputs are generated locally under `outputs/`.
