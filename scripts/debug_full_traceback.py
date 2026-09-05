import traceback

from src.eval_builder.eval_case_db import DEFAULT_EVAL_DB_PATH, load_eval_cases_by_status
from src.eval_runner.runner import _generate_target_response, _judge_response

cases = load_eval_cases_by_status("approved", DEFAULT_EVAL_DB_PATH)
case = cases[0]
print(f"Testing case: {case.case_id}")

try:
    response = _generate_target_response(case)
    print("TARGET RESPONSE:")
    print(response)
    print(f"--- length={len(response)} ---")
except Exception:
    print("FAILED in _generate_target_response:")
    traceback.print_exc()
    raise SystemExit(1)

try:
    passed, reasoning = _judge_response(case, response)
    print(f"\nJUDGE RESULT: passed={passed}")
    print(f"reasoning: {reasoning}")
except Exception:
    print("FAILED in _judge_response:")
    traceback.print_exc()