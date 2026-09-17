from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

from llama_index.core import (
	SQLDatabase,
	StorageContext,
	VectorStoreIndex,
	load_index_from_storage,
)
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from util.SystemUtil import SystemUtil


QDRANT_COLLECTION = "aiia_text_to_sql"
QDRANT_TABLE_COLLECTION = f"{QDRANT_COLLECTION}_tables"


@dataclass(slots=True)
class QueryResult:
	question: str
	sql: str
	answer: str


def _mysql_identifier_quote(engine) -> callable:
	preparer = engine.dialect.identifier_preparer
	return preparer.quote_identifier


def _safe_table_context(sql_database: SQLDatabase, table_name: str) -> str:
	return sql_database.get_single_table_info(table_name)


def _build_llm() -> OpenAILike:
	return OpenAILike(
		model=SystemUtil.CONFIG.model_name,
		api_key=SystemUtil.CONFIG.model_api_key,
		api_base=SystemUtil.CONFIG.model_base_url,
		context_window=128000,
		is_chat_model=True,
		is_function_calling_model=True,
		timeout=120,
	)


def _build_embed_model() -> FastEmbedEmbedding:
	return FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")


def _build_qdrant_client() -> tuple[QdrantClient, AsyncQdrantClient]:
	host = os.getenv("QDRANT_HOST", "localhost")
	port = int(os.getenv("QDRANT_PORT", "6333"))
	url = os.getenv("QDRANT_URL")
	api_key = os.getenv("QDRANT_API_KEY") or None

	if url:
		return (
			QdrantClient(url=url, api_key=api_key),
			AsyncQdrantClient(url=url, api_key=api_key),
		)

	return (
		QdrantClient(host=host, port=port, api_key=api_key),
		AsyncQdrantClient(host=host, port=port, api_key=api_key),
	)


def _build_sql_database(engine, include_tables: Sequence[str] | None = None) -> SQLDatabase:
	if include_tables:
		return SQLDatabase(engine, include_tables=list(include_tables))
	return SQLDatabase(engine)


def _load_or_build_row_index(
	sql_database: SQLDatabase,
	table_name: str,
	embed_model: FastEmbedEmbedding,
	vector_store: QdrantVectorStore,
):
	engine = sql_database.engine
	quote_identifier = _mysql_identifier_quote(engine)
	with engine.connect() as connection:
		rows = connection.execute(text(f"SELECT * FROM {quote_identifier(table_name)}")).fetchall()

	nodes = [TextNode(text=str(tuple(row))) for row in rows]
	if not nodes:
		return None

	storage_context = StorageContext.from_defaults(
		vector_store=vector_store,
	)
	index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
	return index


def _build_table_schema_objs(sql_database: SQLDatabase) -> List[SQLTableSchema]:
	table_schema_objs: List[SQLTableSchema] = []
	for table_name in sql_database.get_usable_table_names():
		table_schema_objs.append(
			SQLTableSchema(
				table_name=table_name,
				context_str=_safe_table_context(sql_database, table_name),
			)
		)
	return table_schema_objs


def _build_query_engine(
	sql_database: SQLDatabase,
	table_schema_objs: Sequence[SQLTableSchema],
	llm: OpenAILike,
	embed_model: FastEmbedEmbedding,
	qdrant_client: QdrantClient,
	async_qdrant_client: AsyncQdrantClient,
):
	schema_vector_store = QdrantVectorStore(
		collection_name=QDRANT_TABLE_COLLECTION,
		client=qdrant_client,
		aclient=async_qdrant_client,
		prefer_grpc=True,
		enable_hybrid=True,
		fastembed_sparse_model="Qdrant/bm25",
	)

	table_node_mapping = SQLTableNodeMapping(sql_database)
	object_index = ObjectIndex.from_objects(
		table_schema_objs,
		table_node_mapping,
		VectorStoreIndex,
		storage_context=StorageContext.from_defaults(vector_store=schema_vector_store),
		embed_model=embed_model,
	)

	rows_retrievers: Dict[str, object] = {}
	for table_schema_obj in table_schema_objs:
		row_vector_store = QdrantVectorStore(
			collection_name=f"{QDRANT_COLLECTION}_rows_{table_schema_obj.table_name}",
			client=qdrant_client,
			aclient=async_qdrant_client,
			prefer_grpc=True,
			enable_hybrid=True,
			fastembed_sparse_model="Qdrant/bm25",
		)
		try:
			row_index = _load_or_build_row_index(
				sql_database=sql_database,
				table_name=table_schema_obj.table_name,
				embed_model=embed_model,
				vector_store=row_vector_store,
			)
		except SQLAlchemyError:
			row_index = None

		if row_index is not None:
			rows_retrievers[table_schema_obj.table_name] = row_index.as_retriever(similarity_top_k=2)

	# Use the table retriever engine to keep the SQL generation tied to the DB schema.
	query_engine = SQLTableRetrieverQueryEngine(
		sql_database,
		object_index.as_retriever(similarity_top_k=3),
		llm=llm,
		rows_retrievers=rows_retrievers,
	)

	return query_engine


def _normalize_response(response) -> str:
	if hasattr(response, "message") and getattr(response.message, "content", None):
		return response.message.content
	return str(response)


def build_text_to_sql_service(include_tables: Sequence[str] | None = None):
	llm = _build_llm()
	embed_model = _build_embed_model()
	qdrant_client, async_qdrant_client = _build_qdrant_client()
	engine = create_engine(SystemUtil.CONFIG.get_mysql_url())
	sql_database = _build_sql_database(engine, include_tables)
	table_schema_objs = _build_table_schema_objs(sql_database)

	query_engine = _build_query_engine(
		sql_database=sql_database,
		table_schema_objs=table_schema_objs,
		llm=llm,
		embed_model=embed_model,
		qdrant_client=qdrant_client,
		async_qdrant_client=async_qdrant_client,
	)

	return {
		"engine": engine,
		"sql_database": sql_database,
		"query_engine": query_engine,
		"llm": llm,
		"table_schema_objs": table_schema_objs,
	}


def ask(question: str, include_tables: Sequence[str] | None = None) -> QueryResult:
	service = build_text_to_sql_service(include_tables=include_tables)
	query_engine = service["query_engine"]
	response = query_engine.query(question)
	return QueryResult(
		question=question,
		sql=getattr(response, "metadata", {}).get("sql_query", ""),
		answer=_normalize_response(response),
	)


async def run():
	service = build_text_to_sql_service()
	query_engine = service["query_engine"]

	examples = [
		"show users and their roles",
		"list issues associated with yiming, please include issue link and issue id",
	]

	for question in examples:
		response = query_engine.query(question)
		print(f"Question: {question}")
		print(_normalize_response(response))
		metadata = getattr(response, "metadata", {})
		if metadata.get("sql_query"):
			print(f"SQL: {metadata['sql_query']}")


__all__ = ["ask", "build_text_to_sql_service", "run", "QueryResult"]
