import duckdb

con = duckdb.connect("data/eval_runs.duckdb")
rows = con.execute(
    "SELECT case_id, passed, judge_reasoning FROM eval_runs ORDER BY timestamp DESC LIMIT 5"
).fetchall()
for case_id, passed, reasoning in rows:
    print(f"{case_id} -> passed={passed}")
    print(f"   {reasoning}")
    print()
con.close()