from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """You are an expert data analyst who writes SQLite SQL queries.

You will be given a database schema and a question in plain English. Write a single SQLite query that answers the question.

Rules:
- Use only the tables and columns that appear in the schema.
- Write valid SQLite syntax.
- Use only SELECT statements. Never modify data.
- Return only the SQL query — no explanation, no comments, no markdown code fences."""

EXPLAIN_PROMPT = """You explain SQL queries in plain English for someone who does not know SQL.

Given a question and the SQL query that answers it, write 1-3 short sentences describing what the query does. Avoid technical jargon. Do not repeat the SQL."""


def schema_to_text(schema: list[dict]) -> str:
    """Turn the structured schema into readable text for the prompt."""
    blocks: list[str] = []
    for table in schema:
        lines = [f"Table {table['name']}:"]
        for col in table["columns"]:
            pk = " PRIMARY KEY" if col["primary_key"] else ""
            lines.append(f"  - {col['name']} ({col['type']}){pk}")
        for fk in table["foreign_keys"]:
            lines.append(
                f"  - FOREIGN KEY {fk['column']} -> "
                f"{fk['references_table']}.{fk['references_column']}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def strip_sql_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped its answer in them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop the opening ``` or ```sql line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop the closing ```
        text = "\n".join(lines)
    return text.strip()


class SQLGenerator:
    """Generates, fixes, and explains SQL queries via an LLM."""

    def __init__(self, client: OpenAI, model: str = DEFAULT_MODEL) -> None:
        self.client = client
        self.model = model

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send one system+user prompt to the model and return its raw text."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def generate(self, question: str, schema: list[dict]) -> str:
        schema_text = schema_to_text(schema)
        user_prompt = (
            f"Database schema:\n{schema_text}\n\n"
            f"Question: {question}\n\n"
            f"SQL:"
        )
        return strip_sql_fences(self._chat(SYSTEM_PROMPT, user_prompt))

    def fix(self, question: str, schema: list[dict], attempts: list[dict]) -> str:
        schema_text = schema_to_text(schema)
        history = "\n\n".join(
            f"Tried this query:\n{past['sql']}\nIt failed with: {past['error']}"
            for past in attempts
        )
        user_prompt = (
            f"Database schema:\n{schema_text}\n\n"
            f"Question: {question}\n\n"
            f"The queries below were already tried and ALL failed. "
            f"Do not repeat any of them:\n\n{history}\n\n"
            f"Write a DIFFERENT SQLite query that avoids these errors. "
            f"Return only the query.\n\n"
            f"SQL:"
        )
        return strip_sql_fences(self._chat(SYSTEM_PROMPT, user_prompt))

    def explain(self, question: str, sql: str) -> str:
        user_prompt = f"Question: {question}\n\nSQL query:\n{sql}\n\nExplanation:"
        return self._chat(EXPLAIN_PROMPT, user_prompt).strip()