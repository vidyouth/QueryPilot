import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI

from app.agent import QueryAgent
from app.db import Database
from app.generator import DEFAULT_MODEL, SQLGenerator

SPIDER_REPO = "prem-research/spider"
SAMPLE_SIZE = 50        # how many dev questions to test (raise later if you want)
RANDOM_SEED = 42        # makes the random sample the same every run
RESULTS_PATH = Path("eval/spider_last_run.json")


def find_dev_questions(root: Path) -> list[dict]:
    """Find and load the Spider dev questions JSON inside the downloaded repo."""
    json_files = sorted(root.rglob("*.json"))
    for path in json_files:
        if path.name.lower() in {"dev.json", "validation.json", "dev_spider.json"}:
            return json.loads(path.read_text())
    # Fallback: a list of question records that isn't the training split.
    for path in json_files:
        if "train" in path.name.lower() or "table" in path.name.lower():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict) and "question" in data[0]:
            return data
    raise FileNotFoundError(
        f"No dev questions JSON found. JSON files present: {[p.name for p in json_files]}"
    )


def normalize_rows(rows: list[dict]) -> list:
    """Canonical form of a result set. These choices ARE our definition of 'correct':
    - column order within a row is kept (we do NOT sort inside a row)
    - values compared as strings, so 5 and 5.0 don't falsely differ
    - row order ignored (we sort the rows), matching Spider's default
    - extra or missing columns make results differ (strict, like Spider)
    """
    return sorted(tuple(str(value) for value in row.values()) for row in rows)


def results_match(expected: list[dict], actual: list[dict]) -> bool:
    return normalize_rows(expected) == normalize_rows(actual)


def run_spider_eval() -> None:
    load_dotenv()

    print("Downloading Spider (one-time, a few hundred MB)...")
    spider_root = Path(snapshot_download(repo_id=SPIDER_REPO, repo_type="dataset"))
    database_dir = spider_root / "database"

    cases = find_dev_questions(spider_root)
    sample = random.Random(RANDOM_SEED).sample(cases, min(SAMPLE_SIZE, len(cases)))
    total = len(sample)

    client = OpenAI()
    generator = SQLGenerator(client=client, model=DEFAULT_MODEL)

    correct = 0
    recovered = 0
    skipped = 0
    total_attempts = 0
    total_seconds = 0.0
    records: list[dict] = []

    print(f"Evaluating {total} Spider questions...\n")

    for index, case in enumerate(sample, start=1):
        db_id = case["db_id"]
        question = case["question"]
        gold_sql = case.get("query") or case.get("sql")
        db_path = database_dir / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            skipped += 1
            print(f"[{index:>2}/{total}] SKIP  (no database file for '{db_id}')")
            continue

        database = Database(db_path)              # your class, unchanged
        agent = QueryAgent(generator=generator, database=database)

        try:
            expected = database.run_query(gold_sql)["rows"]   # gold answer, run locally
        except Exception:
            skipped += 1
            print(f"[{index:>2}/{total}] SKIP  (gold query failed on '{db_id}')")
            continue

        start = time.perf_counter()
        result = agent.answer(question)
        elapsed = time.perf_counter() - start

        attempts = len(result["attempts"])
        is_correct = result["success"] and results_match(expected, result["rows"])

        total_attempts += attempts
        total_seconds += elapsed
        if is_correct:
            correct += 1
            if attempts > 1:
                recovered += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"[{index:>2}/{total}] {status}  ({db_id}, {attempts} try(s), {elapsed:.1f}s)  {question}")

        records.append({
            "db_id": db_id,
            "question": question,
            "correct": is_correct,
            "attempts": attempts,
            "generated_sql": result["sql"],
            "gold_sql": gold_sql,
        })

    scored = total - skipped
    accuracy = (correct / scored * 100) if scored else 0.0

    print("\n" + "=" * 56)
    print(f"Execution accuracy:  {correct}/{scored}  ({accuracy:.1f}%)")
    print(f"Rescued by retry:    {recovered}")
    print(f"Skipped:             {skipped}")
    if scored:
        print(f"Avg tries:           {total_attempts / scored:.2f}")
        print(f"Avg time:            {total_seconds / scored:.1f}s")
    print("=" * 56)

    RESULTS_PATH.write_text(json.dumps({
        "benchmark": "Spider dev (subset)",
        "sample_size": total,
        "scored": scored,
        "correct": correct,
        "accuracy_percent": round(accuracy, 1),
        "rescued_by_retry": recovered,
        "results": records,
    }, indent=2))
    print(f"\nSaved details to {RESULTS_PATH}")


if __name__ == "__main__":
    run_spider_eval()