"""Shared helpers for the frontend pages — thin wrapper around the FastAPI backend."""

import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def api_post(path: str, json: dict | None = None) -> dict:
    response = requests.post(f"{API_BASE}{path}", json=json or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def api_get(path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{API_BASE}{path}", params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def run_pipeline_job_with_live_log(start_path: str, status_path_template: str, payload: dict | None = None) -> dict:
    """
    POST to start_path to kick off a background job, then poll
    status_path_template.format(job_id=...) until it finishes, updating a
    placeholder with the live captured log. Blocks the current Streamlit
    script run for the job's duration — acceptable for this local/demo tool.

    Returns the final job dict.
    """
    try:
        start_result = api_post(start_path, payload)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE} — is `uvicorn src.api.main:app --reload` running? ({exc})")
        st.stop()

    job_id = start_result["job_id"]
    st.write(f"Job started: `{job_id}`")

    status_placeholder = st.empty()
    log_placeholder = st.empty()

    while True:
        job = api_get(status_path_template.format(job_id=job_id))
        status_placeholder.write(f"**Status:** {job['status']}")

        log_lines = job.get("log")
        if log_lines:
            log_placeholder.code("\n".join(log_lines[-40:]), language="text")

        if job["status"] in ("completed", "failed"):
            return job

        time.sleep(2)
