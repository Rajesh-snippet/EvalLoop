import duckdb

con = duckdb.connect("data/eval_runs.duckdb")
rows = con.execute(
    "SELECT case_id, judge_reasoning FROM eval_runs ORDER BY timestamp DESC LIMIT 5"
).fetchall()
for case_id, reasoning in rows:
    print(case_id, "->", reasoning)
con.close()