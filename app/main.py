from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from openai import OpenAI
from pydantic import BaseModel

from app.agent import QueryAgent
from app.db import DB_PATH, Database
from app.generator import DEFAULT_MODEL, SQLGenerator

load_dotenv()  # must run before OpenAI() so the key is in the environment

app = FastAPI(title="QueryPilot")

database = Database(DB_PATH)
openai_client = OpenAI()
generator = SQLGenerator(client=openai_client, model=DEFAULT_MODEL)
agent = QueryAgent(generator=generator, database=database)


def get_database() -> Database:
    return database


def get_agent() -> QueryAgent:
    return agent


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


class Attempt(BaseModel):
    sql: str
    error: str | None


class QueryResponse(BaseModel):
    question: str
    success: bool
    sql: str
    explanation: str | None
    columns: list[str]
    rows: list[dict]
    attempts: list[Attempt]
    error: str | None


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def read_schema(db: Database = Depends(get_database)) -> SchemaResponse:
    return SchemaResponse(tables=db.get_schema())


@app.post("/query")
def query(
    request: QueryRequest, query_agent: QueryAgent = Depends(get_agent)
) -> QueryResponse:
    result = query_agent.answer(request.question)
    return QueryResponse(**result)