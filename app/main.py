from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from openai import OpenAI
from pydantic import BaseModel

from app.db import DB_PATH, Database
from app.generator import DEFAULT_MODEL, SQLGenerator

load_dotenv()  # must run before OpenAI() so the key is in the environment

app = FastAPI(title="QueryPilot")

database = Database(DB_PATH)
openai_client = OpenAI()
generator = SQLGenerator(client=openai_client, model=DEFAULT_MODEL)


def get_database() -> Database:
    return database


def get_generator() -> SQLGenerator:
    return generator


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


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    sql: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def read_schema(db: Database = Depends(get_database)) -> SchemaResponse:
    return SchemaResponse(tables=db.get_schema())


@app.post("/query")
def query(
    request: QueryRequest,
    db: Database = Depends(get_database),
    sql_generator: SQLGenerator = Depends(get_generator),
) -> QueryResponse:
    schema = db.get_schema()
    sql = sql_generator.generate(request.question, schema)
    return QueryResponse(question=request.question, sql=sql)