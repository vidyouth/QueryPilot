from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.db import DB_PATH, Database

app = FastAPI(title="QueryPilot")

database = Database(DB_PATH)


def get_database() -> Database:
    return database


class ColumnInfo(BaseModel):
    name: str
    type: str
    primary_key: bool
    nullable: bool


class ForeignKeyInfo(BaseModel):
    column: str
    references_table: str
    references_column: str


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKeyInfo]


class SchemaResponse(BaseModel):
    tables: list[TableInfo]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def read_schema(db: Database = Depends(get_database)) -> SchemaResponse:
    return SchemaResponse(tables=db.get_schema())