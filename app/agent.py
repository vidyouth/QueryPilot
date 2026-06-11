from app.db import Database
from app.generator import SQLGenerator


class QueryAgent:
    """Generates SQL, runs it, and retries by feeding errors back to the model."""

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
        last_error = ""

        for attempt_number in range(self.max_retries + 1):
            try:
                result = self.database.run_query(sql)
            except Exception as error:
                last_error = str(error)
                attempts.append({"sql": sql, "error": last_error})
                if attempt_number < self.max_retries:
                    sql = self.generator.fix(question, schema, sql, last_error)
                continue

            # The query ran successfully.
            attempts.append({"sql": sql, "error": None})
            return {
                "question": question,
                "success": True,
                "sql": sql,
                "columns": result["columns"],
                "rows": result["rows"],
                "attempts": attempts,
                "error": None,
            }

        # Every attempt failed — we give up and report the last error.
        return {
            "question": question,
            "success": False,
            "sql": sql,
            "columns": [],
            "rows": [],
            "attempts": attempts,
            "error": last_error,
        }