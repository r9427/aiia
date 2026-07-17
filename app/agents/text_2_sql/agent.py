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
from llama_index.core.schema import TextNode
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.embeddings.openai import OpenAIEmbedding

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


def _predict_table_info(prompt_tmpl: ChatPromptTemplate, table_str: str, exclude_table_name_list: str) -> TableInfo:
    """Get table info without tool-calling to avoid tool_choice conflicts."""
    messages = prompt_tmpl.format_messages(
        table_str=table_str,
        exclude_table_name_list=exclude_table_name_list,
    )
    chat_response = llm.chat(messages)
    payload = _extract_json_block(chat_response.message.content or "")
    data = json.loads(payload)
    return TableInfo.model_validate(data)

class TableInfo(BaseModel):
    """Information regarding a structured table."""

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

dfs = []
source_table_names = []

def get_data():

    print("============= Sql start ===============")

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
    dfs.clear()
    source_table_names.clear()

    # for roundtripping
    for table_name, sql in sql_items:
        with engine.connect() as connection:
            df = pd.read_sql(sql, connection)
        dfs.append(df)
        source_table_names.append(table_name)

    # for row in rows:
    #     print(row)

    print("============= Sql end ===============")

def _get_tableinfo_with_index(idx: int):
    results_gen = tableinfo_path.glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.model_validate_json(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )

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
    """Index all tables."""
    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    vector_index_dict = {}
    engine = sql_database.engine
    identifier_preparer = engine.dialect.identifier_preparer
    for table_name in sql_database.get_usable_table_names():
        # print(f"Indexing rows in table: {table_name}")
        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            # get all rows from table
            with engine.connect() as conn:
                quoted_table_name = identifier_preparer.quote_identifier(table_name)
                cursor = conn.execute(text(f"SELECT * FROM {quoted_table_name}"))
                result = cursor.fetchall()
                row_tups = []
                for row in result:
                    row_tups.append(tuple(row))

            # index each row, put into vector store index
            nodes = [TextNode(text=str(t)) for t in row_tups]

            # put into vector store index (may fail if embedding provider returns invalid vectors)
            try:
                index = VectorStoreIndex(nodes)
            except Exception as exc:
                print(f"Skip indexing table '{table_name}' due to embedding error: {exc}")
                continue

            # save index
            index.set_index_id("vector_index")
            index.storage_context.persist(f"{table_index_dir}/{table_name}")
        else:
            # rebuild storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{table_index_dir}/{table_name}"
            )
            # load index
            index = load_index_from_storage(
                storage_context, index_id="vector_index"
            )
        vector_index_dict[table_name] = index

    return vector_index_dict

vector_index_dict: Dict[str, VectorStoreIndex] = {}


def _has_valid_embedding_api_key() -> bool:
    """Allow self-hosted embedding endpoints to run without an API key."""
    base_url = (SystemUtil.CONFIG.model_base_url or "").strip().lower()
    is_self_hosted = (
        base_url.startswith("http://")
        and "dashscope.aliyuncs.com" not in base_url
        and "api.openai.com" not in base_url
    )

    if is_self_hosted:
        return True

    key = (SystemUtil.CONFIG.model_api_key or "").strip()
    if not key:
        return False
    return key.lower() not in {"na", "n/a", "none", "null", "your-api-key"}


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
        vector_index = vector_index_dict.get(table_schema_obj.table_name)
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
            print(f"> Table Info: {table_info}")

        context_strs.append(table_info)
    return "\n\n".join(context_strs)

async def run_agent():
    global vector_index_dict

    get_data()

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

    

    table_names = set()
    table_infos = []
    for idx, df in enumerate(dfs):
        table_info = _get_tableinfo_with_index(idx)
        if table_info:
            # Keep existing metadata and reserve name to avoid collisions.
            table_names.add(table_info.table_name)
        else:
            while True:
                df_str = df.head(10).to_csv()
                table_info = _predict_table_info(
                    prompt_tmpl,
                    table_str=df_str,
                    exclude_table_name_list=str(list(table_names)),
                )
                table_name = table_info.table_name
                print(f"Processed table: {table_name}")
                if table_name not in table_names:
                    table_names.add(table_name)
                    break
                else:
                    # try again
                    print(f"Table name {table_name} already exists, trying again.")
                    pass

            # out_file = f"{tableinfo_dir}/{idx}_{table_name}.json"
            out_file = f"{idx}_{table_name}.json"
            # tableinfo_path.joinpath(out_file).write_text(table_info.model_dump_json(indent=4))
            output_path = tableinfo_path.joinpath(out_file)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(table_info.model_dump(), f)

        table_infos.append(table_info)
    
    # print("********** Start ************")
    # for e in table_infos:
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>")
    #     print(e)

    # print("********* End *************")

    # test = True
    # if test:
    #     return

    table_schema_objs = [
        SQLTableSchema(table_name=source_table_names[idx], context_str=t.table_summary)
        for idx, t in enumerate(table_infos)
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
    print(text2sql_prompt.template)

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
    

    draw_all_possible_flows(
        TextToSQLWorkflow1, filename="text_to_sql_table_retrieval.html"
    )

    # Read the contents of the HTML file
    with open("text_to_sql_table_retrieval.html", "r") as file:
        html_content = file.read()

    # Display the HTML content
    display(HTML(html_content))


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
        "list issue_feedbacks associated with yiming",
        "show users and their roles",
    ]
    # response = await workflow.run(
    #     query=queries[1]
    # )
    # print(str(response))

    vector_index_dict = index_all_tables(sql_database)
    workflow2 = TextToSQLWorkflow2(
        table_schema_objs,
        text2sql_prompt,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        verbose=True,
    )
    response = await workflow2.run(query=queries[0])

    print(str(response))



