import duckdb

con = duckdb.connect("data/eval_runs.duckdb")

row = con.execute(
    """
    SELECT model_response
    FROM eval_runs
    WHERE case_id = 'case_1be5c7cf1466'
    ORDER BY timestamp DESC
    LIMIT 1
    """
).fetchone()

print(row[0])
print()
print(f"--- END (length={len(row[0])}) ---")
con.close()