import duckdb

con = duckdb.connect("data/eval_runs.duckdb")

latest_run = con.execute(
    "SELECT run_id FROM eval_runs ORDER BY timestamp DESC LIMIT 1"
).fetchone()[0]

rows = con.execute(
    """
    SELECT case_id, passed, LENGTH(model_response) AS resp_len, judge_reasoning
    FROM eval_runs
    WHERE run_id = ? AND passed = false
    ORDER BY case_id
    LIMIT 10
    """,
    [latest_run],
).fetchall()

for case_id, passed, resp_len, reasoning in rows:
    print(f"{case_id} | response_length={resp_len}")
    print(f"   {reasoning}")
    print()

con.close()