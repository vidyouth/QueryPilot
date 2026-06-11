import json
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.agent import QueryAgent
from app.db import DB_PATH, Database
from app.generator import DEFAULT_MODEL, SQLGenerator

TEST_CASES_PATH = Path("eval/test_cases.json")
RESULTS_PATH = Path("eval/last_run.json")


def normalize_rows(rows: list[dict]) -> list:
    """Put rows in a canonical form so the same answer matches even if the
    column names or ordering differ."""
    canonical = [tuple(sorted(str(value) for value in row.values())) for row in rows]
    return sorted(canonical)


def results_match(expected: list[dict], actual: list[dict]) -> bool:
    return normalize_rows(expected) == normalize_rows(actual)


def run_evaluation() -> None:
    load_dotenv()
    database = Database(DB_PATH)
    client = OpenAI()
    generator = SQLGenerator(client=client, model=DEFAULT_MODEL)
    agent = QueryAgent(generator=generator, database=database)

    test_cases = json.loads(TEST_CASES_PATH.read_text())
    total = len(test_cases)

    correct = 0
    first_try_correct = 0
    recovered = 0
    total_attempts = 0
    total_seconds = 0.0
    by_difficulty: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    records: list[dict] = []

    print(f"Running {total} test cases...\n")

    for index, case in enumerate(test_cases, start=1):
        question = case["question"]
        gold_sql = case["sql"]
        difficulty = case.get("difficulty", "unknown")

        expected = database.run_query(gold_sql)["rows"]

        start = time.perf_counter()
        result = agent.answer(question)
        elapsed = time.perf_counter() - start

        attempts = len(result["attempts"])
        is_correct = result["success"] and results_match(expected, result["rows"])

        total_attempts += attempts
        total_seconds += elapsed
        by_difficulty[difficulty][1] += 1
        if is_correct:
            correct += 1
            by_difficulty[difficulty][0] += 1
            if attempts == 1:
                first_try_correct += 1
            else:
                recovered += 1

        status = "PASS" if is_correct else "FAIL"
        print(
            f"[{index:>2}/{total}] {status}  "
            f"({attempts} attempt{'s' if attempts != 1 else ''}, {elapsed:.1f}s)  {question}"
        )

        records.append(
            {
                "question": question,
                "difficulty": difficulty,
                "correct": is_correct,
                "attempts": attempts,
                "seconds": round(elapsed, 2),
                "generated_sql": result["sql"],
                "gold_sql": gold_sql,
            }
        )

    accuracy = correct / total * 100

    print("\n" + "=" * 52)
    print(f"Accuracy:            {correct}/{total}  ({accuracy:.0f}%)")
    print(f"Correct first try:   {first_try_correct}")
    print(f"Recovered by retry:  {recovered}")
    print(f"Avg attempts:        {total_attempts / total:.2f}")
    print(f"Avg time:            {total_seconds / total:.1f}s")
    print("\nBy difficulty:")
    for level in ("easy", "medium", "hard"):
        if level in by_difficulty:
            c, t = by_difficulty[level]
            print(f"  {level:<8} {c}/{t}")
    print("=" * 52)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy_percent": round(accuracy, 1),
            "first_try_correct": first_try_correct,
            "recovered_by_retry": recovered,
            "avg_attempts": round(total_attempts / total, 2),
            "avg_seconds": round(total_seconds / total, 2),
        },
        "results": records,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved detailed results to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()