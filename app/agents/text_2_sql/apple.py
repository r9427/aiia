import logging
import sys
import os

from llama_index.llms.openai_like import OpenAILike

from qdrant_client import QdrantClient, AsyncQdrantClient
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings

from llama_index.embeddings.openai import OpenAIEmbedding

from util.SystemUtil import SystemUtil



Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Settings.embed_model = OpenAIEmbedding(
#     model_name=SystemUtil.CONFIG.model_embedding_name,
#     api_base=SystemUtil.CONFIG.model_base_url,
#     api_key=SystemUtil.CONFIG.model_api_key,
#     timeout=30,                      # 防止网络超时
# )

# 配置全局 LLM 为 Qwen
Settings.llm = OpenAILike(
    model=SystemUtil.CONFIG.model_name,
    api_key=SystemUtil.CONFIG.model_api_key,
    api_base=SystemUtil.CONFIG.model_base_url,
    is_chat_model=True,
    is_function_calling_model=True
)


# load documents
docs_path = SystemUtil.BASE_DIR.joinpath("app", "agents", "text_2_sql", "docs")
documents = SimpleDirectoryReader(docs_path).load_data()

async def test_qdrant():
    client = QdrantClient (
        # you can use :memory: mode for fast and light-weight experiments,
        # it does not require to have Qdrant deployed anywhere
        # but requires qdrant-client >= 1.1.1
        # location=":memory:"
        # otherwise set Qdrant instance address with:
        # url="http://<host>:<port>"
        # otherwise set Qdrant instance with host and port:
        host="localhost",
        port=6333
        # set API KEY for Qdrant Cloud
        # api_key="<qdrant-api-key>",
    )

    aclient = AsyncQdrantClient(
        # you can use :memory: mode for fast and light-weight experiments,
        # it does not require to have Qdrant deployed anywhere
        # but requires qdrant-client >= 1.1.1
        host="localhost",
        port=6333
        # otherwise set Qdrant instance address with:
        # uri="http://<host>:<port>"
        # set API KEY for Qdrant Cloud
        # api_key="<qdrant-api-key>",
    )


    vector_store = QdrantVectorStore(
        collection_name="test1",
        client=client,
        aclient=aclient,
        prefer_grpc=True,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        use_async=True,
    )

    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine(
        use_async=True,
        vector_store_query_mode="hybrid",
        sparse_top_k=2,
        similarity_top_k=2,
        hybrid_top_k=3,
    )

    user_msgs = [
        "What did the author do in college? Also, what's 7 * 8?",
        "林晚做的资料叫什么？并告诉我你的回答所基于的资料出处，路径，或链接",
        "宋来遂什么时候占领了东城？他不是历史人物，是内部资料里提到的一个人物，告诉我你的回答所基于的资料出处，路径，或链接",
        "宋来遂什么时候怎么死的？并告诉我你的回答所基于的资料出处，路径，或链接",
    ]

    response = await query_engine.aquery(user_msgs[1])
    print("response: ", response)
    display(Markdown(f"<b>{response}</b>"))

    loaded_index = VectorStoreIndex.from_vector_store(
        vector_store,
        # Embedding model should match the original embedding model
        # embed_model=Settings.embed_model
    )

    print("done loading index from vector store")