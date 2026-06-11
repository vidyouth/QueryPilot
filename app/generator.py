from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """You are an expert data analyst who writes SQLite SQL queries.

You will be given a database schema and a question in plain English. Write a single SQLite query that answers the question.

Rules:
- Use only the tables and columns that appear in the schema.
- Write valid SQLite syntax.
- Use only SELECT statements. Never modify data.
- Return only the SQL query — no explanation, no comments, no markdown code fences."""


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
    """Turns a natural-language question + schema into a SQL query via an LLM."""

    def __init__(self, client: OpenAI, model: str = DEFAULT_MODEL) -> None:
        self.client = client
        self.model = model

    def generate(self, question: str, schema: list[dict]) -> str:
        schema_text = schema_to_text(schema)
        user_prompt = (
            f"Database schema:\n{schema_text}\n\n"
            f"Question: {question}\n\n"
            f"SQL:"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        return strip_sql_fences(raw)