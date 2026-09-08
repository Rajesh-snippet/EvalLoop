import time
from typing import Any
import requests
import streamlit as st
API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 30

def _request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        r = requests.request(method, f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the EvalLoop API at {API_BASE}. Start it with `uvicorn src.api.main:app --reload`.") from exc

def api_get(path: str, params: dict | None = None) -> Any:
    return _request("GET", path, params=params or {})

def api_post(path: str, json: dict | None = None) -> Any:
    return _request("POST", path, json=json or {})

def run_pipeline_job_with_live_log(start_path: str, status_path_template: str, payload: dict | None = None, poll_seconds: float = 2.0) -> dict:
    try:
        start = api_post(start_path, payload)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    job_id = start["job_id"]
    status_box = st.empty()
    log_box = st.empty()
    while True:
        job = api_get(status_path_template.format(job_id=job_id))
        status = job.get("status", "unknown")
        status_box.info(f"Job status: {status}")
        lines = job.get("log") or []
        if lines:
            log_box.code("\n".join(lines[-50:]), language="text")
        if status in {"completed", "failed"}:
            return job
        time.sleep(poll_seconds)
