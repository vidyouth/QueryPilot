from app.db import Database
from app.generator import SQLGenerator


class QueryAgent:
    """Generates SQL, runs it, retries on errors, and explains the result."""

    def __init__(
        self, generator: SQLGenerator, database: Database, max_retries: int = 3
    ) -> None:
        self.generator = generator
        self.database = database
        self.max_retries = max_retries

    def answer(self, question: str) -> dict:
        schema = self.database.get_schema()
        sql = self.generator.generate(question, schema)
        attempts: list[dict] = []

        for attempt_number in range(self.max_retries + 1):
            try:
                result = self.database.run_query(sql)
            except Exception as error:
                attempts.append({"sql": sql, "error": str(error)})
                if attempt_number < self.max_retries:
                    sql = self.generator.fix(question, schema, attempts)
                continue

            # The query ran successfully.
            attempts.append({"sql": sql, "error": None})
            explanation = self.generator.explain(question, sql)
            return {
                "question": question,
                "success": True,
                "sql": sql,
                "explanation": explanation,
                "columns": result["columns"],
                "rows": result["rows"],
                "attempts": attempts,
                "error": None,
            }

        # Every attempt failed.
        return {
            "question": question,
            "success": False,
            "sql": sql,
            "explanation": None,
            "columns": [],
            "rows": [],
            "attempts": attempts,
            "error": attempts[-1]["error"] if attempts else "Unknown error",
        }