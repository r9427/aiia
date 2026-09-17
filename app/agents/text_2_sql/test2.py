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

from llama_index.core.objects import SQLTableSchema, ObjectIndex, SQLTableNodeMapping
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


class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )






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

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
    for t in table_infos
]

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

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

