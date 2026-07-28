import asyncio
from pathlib import Path
import json
import os
import pandas as pd
from IPython.display import display, HTML
import re
import time
from typing import Any, List, Dict

from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SQLDatabase, VectorStoreIndex, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent, AgentStream
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
    Event,
)

from llama_index.core.objects import SQLTableSchema, SQLTableNodeMapping, ObjectIndex
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.retrievers import SQLRetriever, NLSQLRetriever
from llama_index.core.schema import TextNode, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.utils.workflow import draw_all_possible_flows

from qdrant_client import QdrantClient, AsyncQdrantClient, models

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from util.Util import Util
from util.SystemUtil import SystemUtil


# 全局设置 Embedding 模型（支持自建 OpenAI 兼容服务可无 key）
Settings.embed_model = OpenAIEmbedding(
    model_name=SystemUtil.CONFIG.model_embedding_name,
    api_base=SystemUtil.CONFIG.model_base_url,
    api_key=SystemUtil.CONFIG.model_api_key,
    timeout=30,                      # 防止网络超时
)

# 配置全局 LLM 为 Qwen
Settings.llm = OpenAILike(
    model=SystemUtil.CONFIG.model_name,
    api_key=SystemUtil.CONFIG.model_api_key,
    api_base=SystemUtil.CONFIG.model_base_url,
    is_chat_model=True,
    is_function_calling_model=True
)

llm = OpenAILike(
    model=SystemUtil.CONFIG.model_name,
    api_base=SystemUtil.CONFIG.model_base_url,
    api_key=SystemUtil.CONFIG.model_api_key,
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=True,
    timeout=300,
)

vector_table_prefix = "car_"
use_mysql = True

def get_vector_table_name(name: str) -> str:
    return f"{vector_table_prefix}{name}"


def _qualified_table_name(identifier_preparer, table_name: str) -> str:
    quoted_table_name = identifier_preparer.quote_identifier(table_name)
    if not use_mysql:
        quoted_schema = identifier_preparer.quote_schema(SystemUtil.CONFIG.postgresql_schema)
        return f"{quoted_schema}.{quoted_table_name}"
    return quoted_table_name

def get_sql_database(use_mysql: bool = False) -> SQLDatabase:
    if use_mysql:
        db_url = SystemUtil.CONFIG.get_mysql_url()
        engine = create_engine(db_url)
        sql_database = SQLDatabase(engine)
    else:
        db_url = SystemUtil.CONFIG.get_pg_url()
        if "+aiomysql" in db_url:
            db_url = db_url.replace("+aiomysql", "+pymysql")
        elif "+asyncmy" in db_url:
            db_url = db_url.replace("+asyncmy", "+pymysql")
        elif "+asyncpg" in db_url:
            db_url = db_url.replace("+asyncpg", "+psycopg2")

        engine = create_engine(
            db_url,
            pool_size=20,
            max_overflow=300,
            pool_timeout=60,
            pool_pre_ping=True,
            echo=False,
            future=True,
        )
        sql_database = SQLDatabase(engine, schema=SystemUtil.CONFIG.postgresql_schema)
    return sql_database

sql_database = get_sql_database(use_mysql=use_mysql)


MAX_TABLE_PREVIEW_ROWS = 5
MAX_TABLE_PREVIEW_COLS = 20
MAX_CELL_CHARS = 120
MAX_TABLE_STR_CHARS = 8000
MAX_EXCLUDE_NAMES = 50
MAX_TABLE_STR_DIRECT_CHARS = 12000
TABLE_SUMMARY_CHUNK_CHARS = 9000
MAX_COMPRESSED_TABLE_STR_CHARS = 7000
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_SECONDS = 2
INDEX_CONSUMER_COUNT = 8
QDRANT_DENSE_VECTOR_NAME = "text-dense"


def _chat_with_retry(messages: List[ChatMessage]) -> ChatResponse:
    """Call LLM chat with bounded retries for transient timeout errors."""
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return llm.chat(messages)
        except Exception as exc:
            is_timeout = "timeout" in str(exc).lower()
            if (not is_timeout) or attempt == LLM_MAX_RETRIES:
                raise
            wait_seconds = LLM_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"LLM timeout (attempt {attempt}/{LLM_MAX_RETRIES}); retrying in {wait_seconds}s: {exc}")
            time.sleep(wait_seconds)

    # Unreachable fallback to satisfy static analyzers.
    raise RuntimeError("Failed to receive LLM response after retries")


def _extract_json_block(text: str) -> str:
    """Extract a JSON object from model output, handling fenced blocks."""
    if not text:
        raise ValueError("Empty model response")

    content = text.strip()
    if "```" in content:
        start = content.find("```")
        end = content.rfind("```")
        if start != -1 and end != -1 and end > start:
            content = content[start + 3 : end].strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

    left = content.find("{")
    right = content.rfind("}")
    if left == -1 or right == -1 or right <= left:
        raise ValueError(f"No JSON object found in response: {text}")
    return content[left : right + 1]


def _predict_table_info(ref_table_name: str, prompt_tmpl: ChatPromptTemplate, table_str: str, exclude_table_name_list: str) -> TableInfo:
    """Predict table info; prefer structured_predict, then fallback to chat JSON parsing."""
    try:
        structured_llm = OpenAILike(
            model=SystemUtil.CONFIG.model_name,
            api_key=SystemUtil.CONFIG.model_api_key,
            api_base=SystemUtil.CONFIG.model_base_url,
            context_window=128000,
            is_chat_model=True,
            is_function_calling_model=False,
            timeout=300,
        )
        structured_result = super(OpenAI, structured_llm).structured_predict(
            TableInfo,
            prompt_tmpl,
            table_str=table_str,
            exclude_table_name_list=exclude_table_name_list,
        )
        if not isinstance(structured_result, TableInfo):
            structured_result = TableInfo.model_validate(structured_result)
        return structured_result.model_copy(update={"ref_table_name": ref_table_name})
    except Exception as e:
        messages = prompt_tmpl.format_messages(
            table_str=table_str,
            exclude_table_name_list=exclude_table_name_list,
        )

        chat_response = _chat_with_retry(messages)
        payload = _extract_json_block(chat_response.message.content or "")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError(f"Table summary response must be a JSON object, got: {type(data)}")
        # Always take the true DB table name from code, not model output.
        data["ref_table_name"] = ref_table_name
        result = TableInfo.model_validate(data)
        return result

def _predict_table_columns_info(prompt_tmpl: ChatPromptTemplate, table_str: str) -> TableColumnsInfo:
    """Predict table columns info; prefer structured_predict, then fallback to chat JSON parsing."""
    try:
        structured_llm = OpenAILike(
            model=SystemUtil.CONFIG.model_name,
            api_key=SystemUtil.CONFIG.model_api_key,
            api_base=SystemUtil.CONFIG.model_base_url,
            context_window=128000,
            is_chat_model=True,
            is_function_calling_model=False,
            timeout=300,
        )
        structured_result = super(OpenAI, structured_llm).structured_predict(
            TableColumnsInfo,
            prompt_tmpl,
            table_str=table_str
        )
        if not isinstance(structured_result, TableColumnsInfo):
            structured_result = TableColumnsInfo.model_validate(structured_result)
        return structured_result
    except Exception as e:
        messages = prompt_tmpl.format_messages(
            table_str=table_str
        )

        chat_response = _chat_with_retry(messages)
        payload = _extract_json_block(chat_response.message.content or "")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError(f"Table columns response must be a JSON object, got: {type(data)}")
        # Always take the true DB table name from code, not model output.
        result = TableColumnsInfo.model_validate(data)
        return result

def _summarize_table_chunk(table_name: str, chunk_text: str) -> str:
    """Summarize one oversized table chunk while preserving schema/value signal."""
    prompt = (
        "You are compressing table content for a Text-to-SQL system.\n"
        "Keep signal needed for SQL generation, including:\n"
        "1) column names exactly as shown,\n"
        "2) data types/pattern hints,\n"
        "3) representative values and notable categories,\n"
        "4) potential key columns and join hints if visible.\n"
        "Return plain text only, compact and factual.\n\n"
        f"Table: {table_name}\n"
        "Chunk:\n"
        f"{chunk_text}"
    )
    response = _chat_with_retry([ChatMessage.from_str(prompt, role="user")])
    return (response.message.content or "").strip()


def _compress_large_table_str(table_name: str, raw_table_str: str) -> str:
    """Compress very large table text via chunked LLM summaries."""
    chunks = [
        raw_table_str[i : i + TABLE_SUMMARY_CHUNK_CHARS]
        for i in range(0, len(raw_table_str), TABLE_SUMMARY_CHUNK_CHARS)
    ]

    partial_summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        summary = _summarize_table_chunk(table_name, chunk)
        partial_summaries.append(f"[Chunk {idx}]\n{summary}")

    merged = "\n\n".join(partial_summaries)
    if len(merged) <= MAX_COMPRESSED_TABLE_STR_CHARS:
        return merged

    # One more pass if merged summaries are still too long.
    final_prompt = (
        "You are merging chunk summaries for a Text-to-SQL system.\n"
        "Preserve all useful column/value cues, but keep the result concise.\n"
        "Return plain text only.\n\n"
        f"Table: {table_name}\n"
        "Summaries:\n"
        f"{merged}"
    )
    final_resp = _chat_with_retry([ChatMessage.from_str(final_prompt, role="user")])
    return (final_resp.message.content or "").strip()


def _build_table_str_for_prompt(table_name: str, rows: List[tuple], columns: List[str]) -> str:
    """Build a structured prompt payload (schema + sampled rows) for table understanding."""
    def _normalize_row(row: Any) -> tuple:
        # Ensure row shape always aligns with DataFrame columns.
        if hasattr(row, "_mapping"):
            return tuple(row._mapping.get(col) for col in columns)
        if isinstance(row, (tuple, list)):
            seq = tuple(row)
            if len(seq) >= len(columns):
                return seq[: len(columns)]
            return seq + (None,) * (len(columns) - len(seq))
        if len(columns) == 1:
            return (row,)
        return (row,) + (None,) * (len(columns) - 1)

    if not rows:
        empty_payload = {
            "table_name": table_name,
            "columns": columns,
            "sample_rows": [],
        }
        return json.dumps(empty_payload, ensure_ascii=False, indent=2)

    normalized_rows = [_normalize_row(row) for row in rows]
    raw_df = pd.DataFrame(normalized_rows, columns=columns)
    sample_rows = raw_df.iloc[:MAX_TABLE_PREVIEW_ROWS].to_dict(orient="records")
    raw_payload = {
        "table_name": table_name,
        "columns": columns,
        "sample_rows": sample_rows,
    }
    raw_table_str = json.dumps(raw_payload, ensure_ascii=False, indent=2, default=str)
    if len(raw_table_str) <= MAX_TABLE_STR_DIRECT_CHARS:
        return raw_table_str

    try:
        return _compress_large_table_str(table_name, raw_table_str)
    except Exception:
        # Fallback only if summarization fails unexpectedly.
        fallback_df = raw_df.iloc[:MAX_TABLE_PREVIEW_ROWS, :MAX_TABLE_PREVIEW_COLS].copy()
        fallback_df = fallback_df.map(lambda v: str(v)[:MAX_CELL_CHARS])
        fallback_payload = {
            "table_name": table_name,
            "columns": list(fallback_df.columns),
            "sample_rows": fallback_df.to_dict(orient="records"),
        }
        fallback_str = json.dumps(fallback_payload, ensure_ascii=False, indent=2)
        return fallback_str[:MAX_TABLE_STR_CHARS]

class TableInfo(BaseModel):
    """Information regarding a structured table."""

    ref_table_name: str | None = Field(
        None, description="db table name (original name in database, must be the same as in db)"
    )
    extracted_table_name: str = Field(
        ..., description="extracted table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )

class TableColumnsInfo(BaseModel):
    """Column names information regarding a structured table."""

    id_column_name: str | None = Field(
        None, description="ID column name, indicating the column is id of the record"
    )
    create_time_column_name: str | None = Field(
        None, description="create time column name, indicating the column is create time of the record"
    )
    update_time_column_name: str | None = Field(
        None, description="update time column name, indicating the column is update time of the record"
    )

class TableRetrieveEvent(Event):
    """Result of running table retrieval."""

    table_context_str: str
    query: str


class TextToSQLEvent(Event):
    """Text-to-SQL event."""

    sql: str
    query: str


class TextToSQLWorkflow1(Workflow):
    """Text-to-SQL Workflow that does query-time table retrieval."""

    def __init__(
        self,
        obj_retriever,
        # table_schema_objs,
        text2sql_prompt,
        vector_index_dict,
        vector_column_index_dict,
        index_table_columns,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        selector_top_k=3,
        *args,
        **kwargs,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.obj_retriever = obj_retriever
        # self.table_schema_objs = table_schema_objs
        self.text2sql_prompt = text2sql_prompt
        self.vector_index_dict = vector_index_dict
        self.vector_column_index_dict = vector_column_index_dict
        self.index_table_columns = index_table_columns
        self.sql_retriever = sql_retriever
        self.response_synthesis_prompt = response_synthesis_prompt
        self.llm = llm
        self.selector_top_k = selector_top_k

    @step
    def retrieve_tables(
        self, ctx: Context, ev: StartEvent
    ) -> TableRetrieveEvent:
        """Retrieve tables."""
        # table_schema_objs = _select_table_schema_objs(
        #     ev.query,
        #     self.table_schema_objs,
        #     self.llm,
        #     top_k=self.selector_top_k,
        # )
        table_schema_objs = self.obj_retriever.retrieve(ev.query)
        table_context_str = get_table_context_str(table_schema_objs)
        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )

    @step
    def generate_sql(
        self, ctx: Context, ev: TableRetrieveEvent
    ) -> TextToSQLEvent:
        """Generate SQL statement."""
        fmt_messages = self.text2sql_prompt.format_messages(
            query_str=ev.query, schema=ev.table_context_str
        )
        chat_response = self.llm.chat(fmt_messages)
        dialect_name = sql_database.engine.dialect.name
        sql = parse_response_to_sql(chat_response)
        sql = _rewrite_reserved_aliases(sql, dialect_name)
        return TextToSQLEvent(sql=sql, query=ev.query)

    @step
    def generate_response(self, ctx: Context, ev: TextToSQLEvent) -> StopEvent:
        """Run SQL retrieval and generate response."""
        try:
            print(f'Generated SQL: "{ev.sql}"')
            retrieved_rows = self.sql_retriever.retrieve(ev.sql)
        except (NotImplementedError, SQLAlchemyError) as exc:
            # Return a structured fallback instead of raising, so the workflow can continue.
            fallback = (
                "I could not execute the generated SQL against the current schema. "
                "Please rephrase the request using available tables (users, roles, user_roles, issue_feedbacks).\n"
                f"Generated SQL: {ev.sql}\n"
                f"Execution error: {exc}"
            )
            return StopEvent(result=fallback)

        fmt_messages = self.response_synthesis_prompt.format_messages(
            sql_query=ev.sql,
            context_str=str(retrieved_rows),
            query_str=ev.query,
        )
        chat_response = self.llm.chat(fmt_messages)
        return StopEvent(result=chat_response)

class TextToSQLWorkflow2(TextToSQLWorkflow1):
    """Text-to-SQL Workflow that does query-time row AND table retrieval."""

    @step
    def retrieve_tables(
        self, ctx: Context, ev: StartEvent
    ) -> TableRetrieveEvent:
        """Retrieve tables."""
        table_schema_objs = self.obj_retriever.retrieve(ev.query)
        table_context_str = get_table_context_and_rows_str(
            ev.query, table_schema_objs, self.vector_index_dict, verbose=self._verbose
        )
        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )

class TextToSQLWorkflow3(TextToSQLWorkflow1):
    """Text-to-SQL Workflow that does query-time row AND table retrieval."""

    @step
    def retrieve_tables(
        self, ctx: Context, ev: StartEvent
    ) -> TableRetrieveEvent:
        """Retrieve tables."""
        table_schema_objs = self.obj_retriever.retrieve(ev.query)
        table_context_str = get_table_context_and_rows_cols_str(
            ev.query, table_schema_objs, self.vector_index_dict, self.vector_column_index_dict, self.index_table_columns, verbose=self._verbose
        )
        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )


output_dir = SystemUtil.OUTPUT_DIR
tableinfo_dir = "WikiTableQuestions_TableInfo"
tableinfo_path = output_dir.joinpath(tableinfo_dir)
Util.remove_dir(tableinfo_path)
tableinfo_path.mkdir(parents=True, exist_ok=True)


def get_data():

    sql_items = [
        ("users", "select * from users"),
        ("roles", "select * from roles"),
        ("user_roles", "select * from user_roles"),
        ("issue_feedbacks", "select * from issue_feedbacks"),
    ]
    # # async with engine.connect() as connection:
    # #     results = await connection.execute(text(sql))
    # #     for row in results:
    # #         print(row)

    # with engine.connect() as connection:
    #     results = connection.execute(text(sql))
    #     for row in results:
    #         print(row)


    # avoid accumulating stale DataFrames when run_agent is called multiple times
    dfs = []
    source_table_names = []
    # for roundtripping
    for table_name, sql in sql_items:
        with sql_database.engine.connect() as connection:
            df = pd.read_sql(sql, connection)
        dfs.append(df)
        source_table_names.append(table_name)


def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)

def parse_response_to_sql(chat_response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = chat_response.message.content or ""

    # Unwrap fenced code blocks first, e.g. ```sql ... ```.
    fenced = re.search(r"```(?:sql)?\s*(.*?)\s*```", response, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        response = fenced.group(1)

    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]

    response = response.strip()

    # Some models return a bare leading language marker ("sql") on its own line.
    response = re.sub(r"^sql\s*", "", response, flags=re.IGNORECASE)

    # Remove trailing semicolon noise and keep SQL body clean.
    return response.strip().strip(";").strip()

def _rewrite_reserved_aliases(sql: str, dialect_name: str) -> str:
    """Rewrite reserved table aliases that can break SQL parsing in MySQL."""
    if not sql or dialect_name.lower() != "mysql":
        return sql

    mysql_reserved_aliases = {
        "IF",
        "KEY",
        "ORDER",
        "GROUP",
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
    }

    sql_clause_tokens = {
        "WHERE",
        "GROUP",
        "ORDER",
        "BY",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "JOIN",
        "LEFT",
        "RIGHT",
        "INNER",
        "OUTER",
        "CROSS",
        "ON",
        "UNION",
    }

    alias_pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+`?[A-Za-z_][A-Za-z0-9_]*`?\s+(?:(AS)\s+)?([A-Za-z_][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    )
    aliases = []
    for match in alias_pattern.finditer(sql):
        as_kw = match.group(1)
        alias = match.group(2)
        alias_upper = alias.upper()

        # Root-cause fix: avoid treating clause keywords (e.g. ORDER in ORDER BY) as aliases.
        if as_kw is None and alias_upper in sql_clause_tokens:
            continue

        if alias_upper in mysql_reserved_aliases:
            aliases.append(alias)

    rewritten_sql = sql
    for i, alias in enumerate(dict.fromkeys(aliases), start=1):
        safe_alias = f"t{i}"
        rewritten_sql = re.sub(rf"\b{re.escape(alias)}\b", safe_alias, rewritten_sql)

    return rewritten_sql


def _select_table_schema_objs(
    query: str,
    table_schema_objs: List[SQLTableSchema],
    selector_llm,
    top_k: int = 3,
) -> List[SQLTableSchema]:
    """Dynamically select relevant tables for a query using the LLM."""
    if not table_schema_objs:
        return []

    candidates = "\n".join(
        [
            f"- {t.table_name}: {(t.context_str or '').strip()}"
            for t in table_schema_objs
        ]
    )
    prompt = (
        "You are selecting SQL tables for a user query.\n"
        f"User query: {query}\n\n"
        "Candidate tables:\n"
        f"{candidates}\n\n"
        f"Return only the most relevant table names, comma-separated, up to {top_k}.\n"
        "Use exact names from the candidate list only."
    )

    try:
        resp = selector_llm.chat([ChatMessage.from_str(prompt, role="user")])
        raw = (resp.message.content or "").strip()
    except Exception:
        return table_schema_objs

    selected_names = []
    for token in raw.replace("\n", ",").split(","):
        name = token.strip().strip("`\"'")
        if name:
            selected_names.append(name)

    selected_set = set(selected_names)
    filtered = [t for t in table_schema_objs if t.table_name in selected_set]
    if filtered:
        return filtered[:top_k]

    # Fallback: keep behavior flexible and safe if selector output is malformed.
    return table_schema_objs

async def load_vector_index(sql_database: SQLDatabase, client: QdrantClient, aclient: AsyncQdrantClient, index_table_columns: Dict[str, List[str]]):
    vector_index_dict = dict()
    vector_column_index_dict = dict()

    for table_name in sql_database.get_usable_table_names():
        vector_table_name = get_vector_table_name(table_name)
        if not await aclient.collection_exists(collection_name=vector_table_name):
            # Some tables are intentionally skipped during indexing (empty/too large/failed).
            continue
        count_response = await aclient.count(collection_name=vector_table_name, exact=True)
        existing_points_number = count_response.count if count_response else 0
        if existing_points_number <= 0:
            # Do not register retrievers for empty collections.
            continue
        vector_store = QdrantVectorStore(
            collection_name=vector_table_name,
            client=client,
            aclient=aclient,
            prefer_grpc=True,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
        )
        vector_index_dict[table_name] = VectorStoreIndex.from_vector_store(
            vector_store,
            # Embedding model should match the original embedding model
            # embed_model=Settings.embed_model
        )

    for table_name, columns_to_index in index_table_columns.items():
        for column_name in columns_to_index:
            vector_column_table_name = get_vector_table_name(f"{table_name}_{column_name}")
            if not await aclient.collection_exists(collection_name=vector_column_table_name):
                # Some column tables are intentionally skipped during indexing (empty/too large/failed).
                continue
            count_response = await aclient.count(collection_name=vector_column_table_name, exact=True)
            existing_points_number = count_response.count if count_response else 0
            if existing_points_number <= 0:
                # Do not register retrievers for empty collections.
                continue
            vector_store = QdrantVectorStore(
                collection_name=vector_column_table_name,
                client=client,
                aclient=aclient,
                prefer_grpc=True,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm25",
            )
            vector_column_index_dict[f"{table_name}_{column_name}"] = VectorStoreIndex.from_vector_store(
                vector_store,
                # Embedding model should match the original embedding model
                # embed_model=Settings.embed_model
            )

    return vector_index_dict, vector_column_index_dict

def load_table_infos(client: QdrantClient, aclient: AsyncQdrantClient) -> List[TableInfo]:
    table_infos = []
    vector_table_name = get_vector_table_name("table_info")

    vector_store = QdrantVectorStore(
        collection_name=vector_table_name,
        client=client,
        aclient=aclient,
        prefer_grpc=True,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )

    nodes = vector_store.get_nodes()
    table_infos = [
        TableInfo(
            ref_table_name=node.metadata['ref_table_name'],
            extracted_table_name=node.metadata['extracted_table_name'],
            table_summary=node.metadata['table_summary']
        )
        for node in nodes
    ]
    
    return table_infos

async def init_vector_db(sql_database: SQLDatabase, client: QdrantClient, aclient: AsyncQdrantClient, index_table_columns: Dict[str, List[str]] = None):
    await index_vector_db_tables(sql_database, client, aclient, index_table_columns)
    await index_vector_db_table_infos(sql_database, client, aclient)

async def index_vector_db_tables(sql_database: SQLDatabase, client: QdrantClient, aclient: AsyncQdrantClient, index_table_columns: Dict[str, List[str]] = None):
    try:
        async with asyncio.TaskGroup() as task_group:
            queue = asyncio.Queue(100)
            producers = [task_group.create_task(produce_queue(queue, aclient, sql_database))]
            consumers = list()
            for index in range(10):
                consumers.append(task_group.create_task(
                    consume_queue(queue, client, aclient, sql_database, index_table_columns)
                ))
            await asyncio.gather(*producers)
            await queue.join()
            for consumer in consumers:
                consumer.cancel()
    except Exception as e:
            raise e

async def index_vector_db_table_infos(sql_database: SQLDatabase, client: QdrantClient, aclient: AsyncQdrantClient):
    vector_table_name = get_vector_table_name("table_info")

    vector_store = QdrantVectorStore(
        collection_name=vector_table_name,
        client=client,
        aclient=aclient,
        prefer_grpc=True,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    prompt_str = """\
    Create a concise, specific summary for this table.

    Input table payload is JSON, not CSV. It has exactly these keys:
    - table_name: string
    - columns: string[]
    - sample_rows: object[]

    Use only the provided JSON fields (table_name, columns, sample_rows).

    Return JSON only with exactly these keys:
    - extracted_table_name
    - table_summary

    Output rules:
    - extracted_table_name must be concise, unique, and descriptive.
    - extracted_table_name must not be generic (e.g. table, my_table).
    - extracted_table_name must not be one of: {exclude_table_name_list}
    - table_summary should describe purpose, main entities, and key value patterns.
    - No markdown, no code fences, no comments, no extra keys.

    Table JSON:
    {table_str}
    """

    prompt_tmpl = ChatPromptTemplate(
        message_templates=[ChatMessage.from_str(prompt_str, role="user")]
    )

    extracted_table_names = set()
    docs = []

    # a = sql_database.get_single_table_info(table_name=table_name)
    # b = sql_database.get_table_columns(table_name=table_name)
    engine = sql_database.engine
    identifier_preparer = engine.dialect.identifier_preparer
    for table_name in sql_database.get_usable_table_names():
        # print(f"Indexing rows in table: {table_name}")
        with engine.connect() as conn:
            qualified_table_name = _qualified_table_name(identifier_preparer, table_name)
            cursor = conn.execute(text(f"SELECT * FROM {qualified_table_name} LIMIT 10"))
            rows = cursor.fetchall()
            row_tups = []
            for row in rows:
                row_tups.append(tuple(row))
        df_str = _build_table_str_for_prompt(table_name, row_tups, list(cursor.keys()))

        print(f"Process table: {table_name}")
        table_info = _predict_table_info(
            ref_table_name=table_name,
            prompt_tmpl=prompt_tmpl,
            table_str=df_str,
            exclude_table_name_list=str(list(extracted_table_names)),
        )
        extracted_table_name = table_info.extracted_table_name
        if extracted_table_name not in extracted_table_names:
            extracted_table_names.add(extracted_table_name)

            # index each row, put into vector store index
            docs.append(Document(
                text=str(table_info),
                metadata={
                    "ref_table_name": table_info.ref_table_name,
                    "extracted_table_name": table_info.extracted_table_name,
                    "table_summary": table_info.table_summary
                }
            ))
    
    # put into vector store index (may fail if embedding provider returns invalid vectors)
    try:
        index = VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=storage_context,
            use_async=True,
            embed_model=Settings.embed_model,
        )
    except Exception as exc:
        print(f"Skip indexing table '{table_name}' due to embedding error: {exc}")


max_embedding_number = 10000


def _collection_has_expected_dense_vector(collection_info: Any) -> bool:
    """Check whether a Qdrant collection has the named dense vector used by LlamaIndex."""
    try:
        vectors_cfg = collection_info.config.params.vectors
    except Exception:
        return False

    return isinstance(vectors_cfg, dict) and QDRANT_DENSE_VECTOR_NAME in vectors_cfg


async def _is_collection_schema_compatible(
    aclient: AsyncQdrantClient,
    collection_name: str,
) -> bool:
    try:
        collection_info = await aclient.get_collection(collection_name=collection_name)
    except Exception:
        return False
    return _collection_has_expected_dense_vector(collection_info)

async def produce_queue(queue: asyncio.Queue, aclient: AsyncQdrantClient, sql_database: SQLDatabase):
    engine = sql_database.engine
    identifier_preparer = engine.dialect.identifier_preparer
    for table_name in sql_database.get_usable_table_names():
        vector_table_name = get_vector_table_name(table_name)
        try:
            with engine.connect() as conn:
                qualified_table_name = _qualified_table_name(identifier_preparer, table_name)
                cursor = conn.execute(text(f"SELECT COUNT(*) AS total FROM {qualified_table_name}"))
                total = cursor.scalar()
            existed = await aclient.collection_exists(collection_name=vector_table_name)
            if existed:
                compatible = await _is_collection_schema_compatible(aclient, vector_table_name)
                if not compatible:
                    await aclient.delete_collection(collection_name=vector_table_name)
                    existed = False

            if existed:
                count_response = await aclient.count(collection_name=vector_table_name, exact=True)
                existing_points_number = count_response.count if count_response else 0

                if total > max_embedding_number:
                    print(f"Skip table '{table_name}' due to too many rows ({total} > {max_embedding_number})")
                    continue
                elif total == 0:
                    print(f"Skip table '{table_name}' due to no rows")
                    continue

                if existing_points_number == 0:
                    await queue.put(table_name)
                elif existing_points_number < total:
                    await aclient.delete_collection(collection_name=vector_table_name)
                    await queue.put(table_name)
            else:
                if total > max_embedding_number:
                    print(f"Skip table '{table_name}' due to too many rows ({total} > {max_embedding_number})")
                    continue
                elif total == 0:
                    print(f"Skip table '{table_name}' due to no rows")
                    continue
                await queue.put(table_name)

            print(f"Table '{table_name}' has {total} rows; existing points: {existing_points_number if existed else 0}")
        except Exception as exc:
            print(f"Skip table '{table_name}' while checking collection existence: {exc}")
            continue

async def consume_queue(queue: asyncio.Queue, client: QdrantClient, aclient: AsyncQdrantClient, sql_database: SQLDatabase, index_table_columns: Dict[str, List[str]] = None):
    engine = sql_database.engine
    identifier_preparer = engine.dialect.identifier_preparer

    prompt_str = """\
        Identify the table's ID/create-time/update-time columns.

        Input table payload is JSON, not CSV. It has exactly these keys:
        - table_name: string
        - columns: string[]
        - sample_rows: object[]

        Use only the provided JSON fields (table_name, columns, sample_rows).
        Do not invent columns that are not present in columns.

        Return JSON only with exactly these keys:
        - id_column_name
        - create_time_column_name
        - update_time_column_name

        Output rules:
        - Each value must be either a column name from columns or null.
        - If uncertain, return null.
        - No markdown, no code fences, no comments, no extra keys.

        Table JSON:
        {table_str}
        """

    prompt_tmpl = ChatPromptTemplate(
        message_templates=[ChatMessage.from_str(prompt_str, role="user")]
    )

    while True:
        table_name = await queue.get()
        try:
            print(f"Indexing rows in table: {table_name}")
            vector_table_name = get_vector_table_name(table_name)
            vector_store = QdrantVectorStore(
                collection_name=vector_table_name,
                client=client,
                aclient=aclient,
                prefer_grpc=True,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm25",
                batch_size=20
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            with engine.connect() as conn:
                qualified_table_name = _qualified_table_name(identifier_preparer, table_name)
                cursor = conn.execute(text(f"SELECT * FROM {qualified_table_name} LIMIT 10"))
                rows = cursor.fetchall()
                row_tups = []
                for row in rows:
                    row_tups.append(tuple(row))
            table_str = _build_table_str_for_prompt(table_name, row_tups, list(cursor.keys()))
            try:
                table_columns_info = _predict_table_columns_info(
                    prompt_tmpl=prompt_tmpl,
                    table_str=table_str
                )
            except Exception as exc:
                print(f"Fallback column metadata for table '{table_name}' due to inference error: {exc}")
                table_columns_info = TableColumnsInfo(
                    id_column_name=None,
                    create_time_column_name=None,
                    update_time_column_name=None,
                )

            # print("> Table Columns Info:", table_columns_info)

            # order_by_columns = []
            # if table_columns_info.create_time_column_name:
            #     order_by_columns.append(f"{table_columns_info.create_time_column_name} ASC")
            
            # if table_columns_info.update_time_column_name:
            #     order_by_columns.append(f"{table_columns_info.update_time_column_name} DESC")

            # order_by_sql = ""
            # if order_by_columns:
            #     order_by_sql = " ORDER BY " + ", ".join(order_by_columns)

            try:
                # index rows
                docs = []
                with engine.connect() as conn:
                    qualified_table_name = _qualified_table_name(identifier_preparer, table_name)
                    # cursor = conn.execute(text(f"SELECT * FROM {qualified_table_name}{order_by_sql}"))
                    cursor = conn.execute(text(f"SELECT * FROM {qualified_table_name}"))
                    rows = cursor.fetchall()

                    for row in rows:
                        row_mapping = row._mapping
                        docs.append(
                            Document(
                                text=str(tuple(row)),
                                metadata={
                                    "ref_table_name": table_name,
                                    "ref_id_column_name": table_columns_info.id_column_name,
                                    "ref_create_time_column_name": table_columns_info.create_time_column_name,
                                    "ref_update_time_column_name": table_columns_info.update_time_column_name,
                                    "ref_id": row_mapping.get(table_columns_info.id_column_name, None),
                                    "ref_create_time": row_mapping.get(table_columns_info.create_time_column_name, None),
                                    "ref_update_time": row_mapping.get(table_columns_info.update_time_column_name, None),
                                },
                            )
                        )
                VectorStoreIndex.from_documents(
                    documents=docs,
                    storage_context=storage_context,
                    use_async=True,
                    embed_model=Settings.embed_model,
                )

                # index columns
                if index_table_columns and table_name in index_table_columns:
                    columns_to_index = index_table_columns[table_name]
                    for column_name in columns_to_index:
                        column_vector_table_name = get_vector_table_name(f"{vector_table_name}_{column_name}")
                        await aclient.delete_collection(collection_name=column_vector_table_name)

                        with engine.connect() as conn:
                            qualified_table_name = _qualified_table_name(identifier_preparer, table_name)
                            cursor = conn.execute(text(f"SELECT DISTINCT {column_name} FROM {qualified_table_name}"))
                            rows = cursor.fetchall()
                        column_docs = []
                        for row in rows:
                            row_mapping = row._mapping
                            column_value = str(row[0])
                            column_docs.append(
                                Document(
                                    text=column_value,
                                    metadata={
                                        "ref_table_name": table_name,
                                        "ref_column_name": column_name,
                                        "ref_column_value": column_value,
                                    },
                                )
                            )

                        column_vector_store = QdrantVectorStore(
                            collection_name=column_vector_table_name,
                            client=client,
                            aclient=aclient,
                            prefer_grpc=True,
                            enable_hybrid=True,
                            fastembed_sparse_model="Qdrant/bm25",
                            batch_size=20
                        )
                        column_storage_context = StorageContext.from_defaults(vector_store=column_vector_store)
                        VectorStoreIndex.from_documents(
                            documents=column_docs,
                            storage_context=column_storage_context,
                            use_async=True,
                            embed_model=Settings.embed_model,
                        )
            except Exception as exc:
                print(f"Skip indexing table '{table_name}' due to embedding error: {exc}")
        except Exception as exc:
            print(f"Skip indexing table '{table_name}' due to runtime error: {exc}")
        finally:
            queue.task_done()

async def get_collection_points(aclient: AsyncQdrantClient, collection_name: str, order_by_columns: list[dict] = None) -> List[models.ScoredPoint]:
    count_response = await aclient.count(
        collection_name=collection_name,
        # count_filter=models.Filter(
        #     must=[
        #         models.FieldCondition(key="color", match=models.MatchValue(value="red")),
        #     ]
        # ),
        exact=True,
    )
    points_count = count_response.count if count_response else 0

    if points_count == 0:
        return []

    query = None
    valid_order_by_columns = []
    if order_by_columns:
        valid_order_by_columns = [
            col for col in order_by_columns
            if isinstance(col, dict) and col.get("column_name")
        ]
        if valid_order_by_columns:
            # Qdrant OrderByQuery currently accepts a single order_by (str or OrderBy), not a list.
            first_order_by = valid_order_by_columns[0]
            query = models.OrderByQuery(
                order_by=models.OrderBy(
                    key=first_order_by["column_name"],
                    direction=models.Direction.ASC if first_order_by.get("asc", True) else models.Direction.DESC,
                )
            )

    try:
        points_response = await aclient.query_points(
            collection_name=collection_name,
            with_payload=True,
            query=query,
            limit=points_count
        )
        points = points_response.points if points_response and points_response.points else []
        if query is None and valid_order_by_columns:
            return points
        return points
    except Exception as exc:
        # Qdrant order_by requires payload range index. Fallback to plain fetch + in-memory sort.
        if query is not None and "No range index for `order_by` key" in str(exc):
            fallback_response = await aclient.query_points(
                collection_name=collection_name,
                with_payload=True,
                query=None,
                limit=points_count,
            )
            fallback_points = fallback_response.points if fallback_response and fallback_response.points else []
            return fallback_points
        raise

def get_table_context_and_rows_str(
    query_str: str,
    table_schema_objs: List[SQLTableSchema],
    vector_index_dict: Dict[str, VectorStoreIndex],
    verbose: bool = False,
):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        # first append table info + additional context
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        # also lookup vector index to return relevant table rows
        vector_index = vector_index_dict.get(table_schema_obj.table_name)
        # vector_index = vector_index_dict
        if vector_index is not None:
            vector_retriever = vector_index.as_retriever(similarity_top_k=2)
            relevant_nodes = vector_retriever.retrieve(query_str)
            if len(relevant_nodes) > 0:
                table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
                for node in relevant_nodes:
                    table_row_context += str(node.get_content()) + "\n"
                table_info += table_row_context
        else:
            print(f"> No row index found for table: {table_schema_obj.table_name}")

        if verbose:
            print(f"> Table Info: {table_info}")

        context_strs.append(table_info)
    return "\n\n".join(context_strs)

def get_table_context_and_rows_cols_str(
    query_str: str,
    table_schema_objs: List[SQLTableSchema],
    vector_index_dict: Dict[str, VectorStoreIndex],
    vector_column_index_dict: Dict[str, VectorStoreIndex],
    index_table_columns: Dict[str, List[str]],
    verbose: bool = False,
):
    """Get table context string."""
    context_strs = []
    user_queried = False
    for table_schema_obj in table_schema_objs:
        # first append table info + additional context
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        # also lookup vector index to return relevant table rows
        vector_index = vector_index_dict.get(table_schema_obj.table_name)
        # vector_index = vector_index_dict
        if vector_index is not None:
            vector_retriever = vector_index.as_retriever(
                similarity_top_k=3,
                sparse_top_k=12,
                vector_store_query_mode="hybrid"
            )
            relevant_nodes = vector_retriever.retrieve(query_str)
            if len(relevant_nodes) > 0:
                table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
                for node in relevant_nodes:
                    table_row_context += str(node.get_content()) + "\n"
                table_info += table_row_context
        else:
            print(f"> No row index found for table: {table_schema_obj.table_name}")


        # user_vector_index = vector_index_dict.get('users')
        # if not user_queried and user_vector_index is not None:
        #     user_vector_retriever = user_vector_index.as_retriever(
        #         similarity_top_k=2,
        #         sparse_top_k=5,
        #         vector_store_query_mode="hybrid"
        #     )
        #     relevant_user_nodes = user_vector_retriever.retrieve(query_str)
        #     if len(relevant_user_nodes) > 0:
        #         table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
        #         for node in relevant_user_nodes:
        #             table_row_context += str(node.get_content()) + "\n"
        #         table_info += table_row_context
        #     user_queried = True


        for column_name in index_table_columns.get(table_schema_obj.table_name, []):
            vector_column_index = vector_column_index_dict.get(f"{table_schema_obj.table_name}_{column_name}")
            if vector_column_index is not None:
                vector_column_retriever = vector_column_index.as_retriever(
                    similarity_top_k=2,
                    sparse_top_k=12,
                    vector_store_query_mode="hybrid"
                )
                relevant_column_nodes = vector_column_retriever.retrieve(query_str)
                if len(relevant_column_nodes) > 0:
                    table_column_context = f"\nHere are some relevant example values for column '{column_name}'\n"
                    for node in relevant_column_nodes:
                        table_column_context += str(node.get_content()) + "\n"
                    table_info += table_column_context
            else:
                print(f"> No column index found for table: {table_schema_obj.table_name}, column: {column_name}")

        if verbose:
            print(f"> Table Info: {table_info}")

        context_strs.append(table_info)
    return "\n\n".join(context_strs)

async def run_agent():
    engine = sql_database.engine
    identifier_preparer = engine.dialect.identifier_preparer

    client = QdrantClient(
        host=SystemUtil.CONFIG.qdrant_host,
        port=SystemUtil.CONFIG.qdrant_port,
        timeout=10000
    )
    aclient = AsyncQdrantClient(
        host=SystemUtil.CONFIG.qdrant_host,
        port=SystemUtil.CONFIG.qdrant_port,
        timeout=10000
    )


    index_table_columns: Dict[str, List[str]] = {
        "users": ["name", "email"],
        "issue_feedbacks": ["title", "summary"]
    }

    init_db = True  # Set to True to index all tables; False to load existing vector store indices.
    if init_db:
        print("Indexing all tables started")
        await init_vector_db(sql_database, client, aclient, index_table_columns)
        print("Indexing all tables completed")


    vector_index_dict, vector_column_index_dict = await load_vector_index(sql_database, client, aclient, index_table_columns)
    table_infos = load_table_infos(client, aclient)
    print("index loaded")

    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=t.ref_table_name, context_str=t.table_summary)
        for t in table_infos
    ]  # add a SQLTableSchema for each table

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
        # index_cls=index_table_columns
    )
    obj_retriever = obj_index.as_retriever(
        similarity_top_k=4,
        # sparse_top_k=8,
        # vector_store_query_mode="hybrid"
    )
    sql_retriever = SQLRetriever(sql_database)

    # default retrieval (return_raw=False)
    # nl_sql_retriever = NLSQLRetriever(
    #     sql_database, tables=["city_stats"], return_raw=False, rows_retrievers=None, cols_retrievers=None
    # )
    
    text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
        dialect=sql_database.engine.dialect.name
    )
    text2sql_prompt.template += (
        "\n\nWhen generating SQL for MySQL:"
        "\n- Do not use reserved keywords as table aliases (e.g., IF, KEY, ORDER, GROUP)."
        "\n- Prefer safe aliases like t1, t2, u, r, ur, fb."
        # "\n- For human/name filters from natural language (e.g., 'created by tom', 'user john'), prefer case-insensitive fuzzy matching instead of exact equality."
        # "\n- Use patterns like LOWER(TRIM(u.name)) LIKE CONCAT('%', LOWER(TRIM('<name>')), '%') when matching user names."
        # "\n- Only use '=' for names when the user explicitly asks for exact match."
    )
    # print(text2sql_prompt.template)

    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )
    response_synthesis_prompt = PromptTemplate(
        response_synthesis_prompt_str,
    )

    queries = [
        "list issues created by cariad employee zhanbo 18 days ago, please include issue link and issue id, title, description, user name, create time",
        "show users and their roles",
    ]
    
    # run some queries
    workflow1 = TextToSQLWorkflow1(
        obj_retriever,
        text2sql_prompt,
        vector_index_dict,
        vector_column_index_dict,
        index_table_columns,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        selector_top_k=3,
        timeout=180,
        verbose=True,
    )

    # response = await workflow.run(
    #     query=queries[1]
    # )
    # print(str(response))

    workflow2 = TextToSQLWorkflow2(
        obj_retriever,
        text2sql_prompt,
        vector_index_dict,
        vector_column_index_dict,
        index_table_columns,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        verbose=False,
        timeout=100
    )

    workflow3 = TextToSQLWorkflow3(
        obj_retriever,
        text2sql_prompt,
        vector_index_dict,
        vector_column_index_dict,
        index_table_columns,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        verbose=False,
        timeout=1000000
    )
    response = await workflow3.run(query=queries[0])
    print("final result: ")

    print(str(response))



