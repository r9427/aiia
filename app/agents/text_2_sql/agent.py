from pathlib import Path
import json
import os
import pandas as pd
from IPython.display import display, HTML
import re
from typing import List, Dict

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

from llama_index.core.objects import SQLTableSchema
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.schema import TextNode, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.utils.workflow import draw_all_possible_flows

from qdrant_client import QdrantClient, AsyncQdrantClient

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


db_url = SystemUtil.CONFIG.get_mysql_url()
# engine = create_async_engine(
#     db_url,
#     pool_size=20,
#     max_overflow=300,
#     pool_timeout=60,
#     # pool_recycle=120,
#     # pool_pre_ping=True,
#     echo=False,
#     future=True
# )
engine = create_engine(db_url)
sql_database = SQLDatabase(engine)

llm = OpenAILike(
    model=SystemUtil.CONFIG.model_name,
    api_base=SystemUtil.CONFIG.model_base_url,
    api_key=SystemUtil.CONFIG.model_api_key,
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=True,
    timeout=120,
)


MAX_TABLE_PREVIEW_ROWS = 5
MAX_TABLE_PREVIEW_COLS = 20
MAX_CELL_CHARS = 120
MAX_TABLE_STR_CHARS = 8000
MAX_EXCLUDE_NAMES = 50
MAX_TABLE_STR_DIRECT_CHARS = 12000
TABLE_SUMMARY_CHUNK_CHARS = 9000
MAX_COMPRESSED_TABLE_STR_CHARS = 7000

qdrant_collection = "test2"


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


def _predict_table_info(db_table_name: str, prompt_tmpl: ChatPromptTemplate, table_str: str, exclude_table_name_list: str) -> TableInfo:
    """Predict table info; prefer structured_predict, then fallback to chat JSON parsing."""
    try:
        structured_llm = OpenAILike(
            model=SystemUtil.CONFIG.model_name,
            api_key=SystemUtil.CONFIG.model_api_key,
            api_base=SystemUtil.CONFIG.model_base_url,
            context_window=128000,
            is_chat_model=True,
            is_function_calling_model=False,
            timeout=120,
        )
        structured_result = super(OpenAI, structured_llm).structured_predict(
            TableInfo,
            prompt_tmpl,
            table_str=table_str,
            exclude_table_name_list=exclude_table_name_list,
        )
        if not isinstance(structured_result, TableInfo):
            structured_result = TableInfo.model_validate(structured_result)
        return structured_result.model_copy(update={"db_table_name": db_table_name})
    except Exception as e:
        messages = prompt_tmpl.format_messages(
            table_str=table_str,
            exclude_table_name_list=exclude_table_name_list,
        )

        chat_response = llm.chat(messages)
        payload = _extract_json_block(chat_response.message.content or "")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError(f"Table summary response must be a JSON object, got: {type(data)}")
        # Always take the true DB table name from code, not model output.
        data["db_table_name"] = db_table_name
        result = TableInfo.model_validate(data)
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
    response = llm.chat([ChatMessage.from_str(prompt, role="user")])
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
    final_resp = llm.chat([ChatMessage.from_str(final_prompt, role="user")])
    return (final_resp.message.content or "").strip()


def _build_table_str_for_prompt(table_name: str, rows: List[tuple], columns: List[str]) -> str:
    """Use full sampled table content when possible; compress with LLM only if oversized."""
    if not rows:
        return pd.DataFrame(columns=columns).to_csv(index=False)

    raw_df = pd.DataFrame(rows, columns=columns)
    raw_table_str = raw_df.to_csv(index=False)
    if len(raw_table_str) <= MAX_TABLE_STR_DIRECT_CHARS:
        return raw_table_str

    try:
        return _compress_large_table_str(table_name, raw_table_str)
    except Exception:
        # Fallback only if summarization fails unexpectedly.
        fallback_df = raw_df.iloc[:MAX_TABLE_PREVIEW_ROWS, :MAX_TABLE_PREVIEW_COLS].copy()
        fallback_df = fallback_df.map(lambda v: str(v)[:MAX_CELL_CHARS])
        fallback_str = fallback_df.to_csv(index=False)
        return fallback_str[:MAX_TABLE_STR_CHARS]

class TableInfo(BaseModel):
    """Information regarding a structured table."""

    db_table_name: str | None = Field(
        None, description="db table name (original name in database, must be the same as in db)"
    )
    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
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
        table_schema_objs,
        text2sql_prompt,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        selector_top_k=3,
        *args,
        **kwargs,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.table_schema_objs = table_schema_objs
        self.text2sql_prompt = text2sql_prompt
        self.sql_retriever = sql_retriever
        self.response_synthesis_prompt = response_synthesis_prompt
        self.llm = llm
        self.selector_top_k = selector_top_k

    @step
    def retrieve_tables(
        self, ctx: Context, ev: StartEvent
    ) -> TableRetrieveEvent:
        """Retrieve tables."""
        table_schema_objs = _select_table_schema_objs(
            ev.query,
            self.table_schema_objs,
            self.llm,
            top_k=self.selector_top_k,
        )
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
        sql = parse_response_to_sql(chat_response)
        sql = _rewrite_reserved_aliases(sql, engine.dialect.name)
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
        table_schema_objs = self.table_schema_objs
        table_context_str = get_table_context_and_rows_str(
            ev.query, table_schema_objs, verbose=self._verbose
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
        with engine.connect() as connection:
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
    response = chat_response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


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

    alias_pattern = re.compile(r"\b(?:FROM|JOIN)\s+`?[A-Za-z_][A-Za-z0-9_]*`?\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)
    aliases = []
    for match in alias_pattern.finditer(sql):
        alias = match.group(1)
        if alias.upper() in mysql_reserved_aliases:
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

def index_all_tables(
    sql_database: SQLDatabase, table_index_dir: str = "output"
) -> Dict[str, VectorStoreIndex]:
    vector_index_dict = {}
    engine = sql_database.engine
    identifier_preparer = engine.dialect.identifier_preparer


    client = QdrantClient (
        host="localhost",
        port=6333
    )

    aclient = AsyncQdrantClient(
        host="localhost",
        port=6333
    )

    vector_store = QdrantVectorStore(
        collection_name=qdrant_collection,
        client=client,
        aclient=aclient,
        prefer_grpc=True,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    init_vector_db = False
    if init_vector_db:
        for table_name in sql_database.get_usable_table_names():
            print(f"Indexing rows in table: {table_name}")
            with engine.connect() as conn:
                quoted_table_name = identifier_preparer.quote_identifier(table_name)
                cursor = conn.execute(text(f"SELECT * FROM {quoted_table_name}"))
                result = cursor.fetchall()
                row_tups = []
                for row in result:
                    row_tups.append(tuple(row))

            # index each row, put into vector store index
            docs = [Document(text=str(t)) for t in row_tups]

            # put into vector store index (may fail if embedding provider returns invalid vectors)
            try:
                index = VectorStoreIndex.from_documents(
                    documents=docs,
                    storage_context=storage_context,
                    use_async=True,
                    # embed_model=Settings.embed_model,
                )
            except Exception as exc:
                print(f"Skip indexing table '{table_name}' due to embedding error: {exc}")
                continue

        # vector_index_dict[table_name] = index
    else:
        # rebuild storage context
        print(f"Loading existing index for table")
        # load index
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            # Embedding model should match the original embedding model
            # embed_model=Settings.embed_model
        )

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        # Embedding model should match the original embedding model
        # embed_model=Settings.embed_model
    )
    return index

print("Indexing all tables started")
vector_index_dict = index_all_tables(sql_database)
# vector_index_dict: Dict[str, VectorStoreIndex] = {}
print("Indexing all tables  completed")


def get_table_context_and_rows_str(
    query_str: str,
    table_schema_objs: List[SQLTableSchema],
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

        # also lookup vector index to return relevant table rows (optional)
        # vector_index = vector_index_dict.get(table_schema_obj.table_name)
        vector_index = vector_index_dict
        if vector_index is not None:
            vector_retriever = vector_index.as_retriever(similarity_top_k=2)
            relevant_nodes = vector_retriever.retrieve(query_str)
            if len(relevant_nodes) > 0:
                table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
                for node in relevant_nodes:
                    table_row_context += str(node.get_content()) + "\n"
                table_info += table_row_context
        elif verbose:
            print(f"> No row index found for table: {table_schema_obj.table_name}")

        if verbose:
            # print(f"> Table Info: {table_info}")
            pass

        context_strs.append(table_info)
    return "\n\n".join(context_strs)

def get_table_infos(sql_database: SQLDatabase) -> List[TableInfo]:
    table_infos = []
    init_vector_db = False


    client = QdrantClient (
        host="localhost",
        port=6333
    )

    aclient = AsyncQdrantClient(
        host="localhost",
        port=6333
    )

    vector_store = QdrantVectorStore(
        collection_name='table_info',
        client=client,
        aclient=aclient,
        prefer_grpc=True,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if init_vector_db:
        prompt_str = """\
        Give me a summary of the table with the following JSON format.

        - The table name must be unique to the table and describe it while being concise.
        - Do NOT output a generic table name (e.g. table, my_table).
        - Output JSON only, with exactly these keys: table_name, table_summary.
        - Do not include markdown code fences or extra commentary.

        Do NOT make the table name one of the following: {exclude_table_name_list}

        Table:
        {table_str}

        Summary: """

        prompt_tmpl = ChatPromptTemplate(
            message_templates=[ChatMessage.from_str(prompt_str, role="user")]
        )

        summarized_table_names = set()
        docs = []

        # a = sql_database.get_single_table_info(table_name=table_name)
        # b = sql_database.get_table_columns(table_name=table_name)
        engine = sql_database.engine
        identifier_preparer = engine.dialect.identifier_preparer
        for table_name in sql_database.get_usable_table_names():
            # print(f"Indexing rows in table: {table_name}")
            with engine.connect() as conn:
                quoted_table_name = identifier_preparer.quote_identifier(table_name)
                cursor = conn.execute(text(f"SELECT * FROM {quoted_table_name} LIMIT 10"))
                rows = cursor.fetchall()
                row_tups = []
                for row in rows:
                    row_tups.append(tuple(row))
            df_str = _build_table_str_for_prompt(table_name, row_tups, list(cursor.keys()))

            print(f"Process table: {table_name}")
            table_info = _predict_table_info(
                db_table_name=table_name,
                prompt_tmpl=prompt_tmpl,
                table_str=df_str,
                exclude_table_name_list=str(list(summarized_table_names)),
            )
            summarized_table_name = table_info.table_name
            if summarized_table_name not in summarized_table_names:
                summarized_table_names.add(summarized_table_name)
                table_infos.append(table_info)

                # index each row, put into vector store index
                docs.append(Document(
                    text=str(table_info),
                    metadata={
                        "db_table_name": table_info.db_table_name,
                        "table_name": table_info.table_name,
                        "table_summary": table_info.table_summary
                    }
                ))
        
        # put into vector store index (may fail if embedding provider returns invalid vectors)
        try:
            index = VectorStoreIndex.from_documents(
                documents=docs,
                storage_context=storage_context,
                use_async=True,
                # embed_model=Settings.embed_model,
            )
        except Exception as exc:
            print(f"Skip indexing table '{table_name}' due to embedding error: {exc}")
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            # Embedding model should match the original embedding model
            # embed_model=Settings.embed_model
        )
        nodes = vector_store.get_nodes()
        table_infos = [
            TableInfo(
                db_table_name=node.metadata['db_table_name'],
                table_name=node.metadata['table_name'],
                table_summary=node.metadata['table_summary']
            )
            for node in nodes
        ]
    
    return table_infos

async def run_agent():
        
    print("progress: 1")

    table_infos = get_table_infos(sql_database)
    
    # print("********** Start ************")
    # for e in table_infos:
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>")
    #     print(e)

    # print("********* End *************")

    print("progress: 2")


    table_schema_objs = [
        SQLTableSchema(table_name=t.db_table_name, context_str=t.table_summary)
        for t in table_infos
    ]  # add a SQLTableSchema for each table

    sql_retriever = SQLRetriever(sql_database)
    
    text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
        dialect=engine.dialect.name
    )
    text2sql_prompt.template += (
        "\n\nWhen generating SQL for MySQL:"
        "\n- Do not use reserved keywords as table aliases (e.g., IF, KEY, ORDER, GROUP)."
        "\n- Prefer safe aliases like t1, t2, u, r, ur, fb."
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
    

    # draw_all_possible_flows(
    #     TextToSQLWorkflow1, filename="text_to_sql_table_retrieval.html"
    # )

    # Read the contents of the HTML file
    # with open("text_to_sql_table_retrieval.html", "r") as file:
    #     html_content = file.read()

    # Display the HTML content
    # display(HTML(html_content))


    # run some queries
    workflow1 = TextToSQLWorkflow1(
        table_schema_objs,
        text2sql_prompt,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        selector_top_k=3,
        timeout=180,
        verbose=True,
    )


    queries = [
        "list issues associated with yiming, please include issue link and issue id, user name",
        "show users and their roles",
    ]
    # response = await workflow.run(
    #     query=queries[1]
    # )
    # print(str(response))
    print("progress: 3")

    # vector_index_dict = index_all_tables(sql_database)
    # print("progress: 4")

    workflow2 = TextToSQLWorkflow2(
        table_schema_objs,
        text2sql_prompt,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        verbose=False,
        timeout=100
    )

    response = await workflow2.run(query=queries[0])
    print("progress: 4")

    print(str(response))



