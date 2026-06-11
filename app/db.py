import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

def _is_read_only(sql: str) -> bool:
    """Allow only a single read-only SELECT (or WITH ... SELECT) statement."""
    cleaned = sql.strip().rstrip(";").strip()
    if ";" in cleaned:
        return False  # blocks stacked statements like "SELECT ...; DROP ..."
    lowered = cleaned.lower()
    return lowered.startswith("select") or lowered.startswith("with")

DB_PATH = Path("data/chinook.db")


class Database:
    """All database access goes through here. Nothing else touches sqlite3."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_schema(self) -> list[dict]:
        with self.connect() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()

            result: list[dict] = []
            for table in tables:
                table_name = table["name"]
                columns = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
                fkeys = conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()

                result.append(
                    {
                        "name": table_name,
                        "columns": [
                            {
                                "name": col["name"],
                                "type": col["type"],
                                "primary_key": bool(col["pk"]),
                                "nullable": not col["notnull"],
                            }
                            for col in columns
                        ],
                        "foreign_keys": [
                            {
                                "column": fk["from"],
                                "references_table": fk["table"],
                                "references_column": fk["to"],
                            }
                            for fk in fkeys
                        ],
                    }
                )
            return result
        
    def run_query(self, sql: str, max_rows: int = 1000) -> dict:
        """Run a read-only query and return its column names and rows."""
        if not _is_read_only(sql):
            raise ValueError("Only single read-only SELECT queries are allowed.")
        with self.connect() as conn:
            cursor = conn.execute(sql)
            columns = [col[0] for col in cursor.description] if cursor.description else []
            rows = [dict(row) for row in cursor.fetchmany(max_rows)]
        return {"columns": columns, "rows": rows}