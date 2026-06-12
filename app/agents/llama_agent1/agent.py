from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.workflow import Context

from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding

from util.SystemUtil import SystemUtil


# SystemUtil.CONFIG.model_qwen_model_name,
#         SystemUtil.CONFIG.model_qwen_api_key,
#     SystemUtil.CONFIG.model_qwen_base_url

# 全局设置 Qwen Embedding 模型
Settings.embed_model = DashScopeEmbedding(
    model_name=SystemUtil.CONFIG.model_qwen_embedding_name,  # 千问轻量级嵌入模型，性价比高
    api_key=SystemUtil.CONFIG.model_qwen_api_key,
    timeout=30                       # 防止网络超时
)

# 配置全局 LLM 为 Qwen
Settings.llm = OpenAILike(
    model=SystemUtil.CONFIG.model_qwen_model_name,
    api_key=SystemUtil.CONFIG.model_qwen_api_key,
    api_base=SystemUtil.CONFIG.model_qwen_base_url,
    is_chat_model=True,
    is_function_calling_model=True
)


# Save the index
# index.storage_context.persist("storage")

# Later, load the index
# from llama_index.core import StorageContext, load_index_from_storage

# storage_context = StorageContext.from_defaults(persist_dir="storage")
# index = load_index_from_storage(storage_context)
# query_engine = index.as_query_engine()

# Create a RAG tool using LlamaIndex
docs_path = SystemUtil.BASE_DIR.joinpath("app", "agents", "llama_agent1", "docs")
docs_path.mkdir(exist_ok=True)
documents = SimpleDirectoryReader(docs_path).load_data()
# print(f"成功加载 {len(documents)} 个文档")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do in college ?")
# print("查询结果：")
# print(response)

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham, story about 宋来遂, and story about 林晚."""
   
    response = await query_engine.aquery(query)
    return str(response)

async def run_agent():

    print("qwen model config name='{}', api_key='{}', base_url='{}'".format(
        SystemUtil.CONFIG.model_qwen_model_name,
        SystemUtil.CONFIG.model_qwen_api_key,
        SystemUtil.CONFIG.model_qwen_base_url
    ))

    llm = OpenAILike(
        model=SystemUtil.CONFIG.model_qwen_model_name,
        api_base=SystemUtil.CONFIG.model_qwen_base_url,
        api_key=SystemUtil.CONFIG.model_qwen_api_key,
        context_window=128000,
        is_chat_model=True,
        is_function_calling_model=True,
    )

    # Create an agent workflow with our calculator tool
    agent = FunctionAgent(
        tools=[multiply, search_documents],
        llm=llm,
        system_prompt="""You are a helpful assistant that can perform calculations and search through documents to answer questions.""",
    )

    # create context
    ctx = Context(agent)
    # run agent with context
    # response = await agent.run(ctx=ctx, user_msg="My name is Logan")

    user_msgs = [
        "What did the author do in college? Also, what's 7 * 8?",
        "林晚做的资料叫什么？",
        "宋来遂什么时候占领了东城？",
        "宋来遂什么时候怎么死的？"
    ]

    response = await agent.run(
        ctx=ctx,
        user_msg=user_msgs[2]
    )
    print(f"User: {user_msgs[2]}")
    print(str(response))
    
    # print("============== Response Start ==================")

    # for msg in user_msgs:
    #     print(f"User: {msg}")
    #     response = await agent.run(
    #         ctx=ctx,
    #         user_msg=msg
    #     )
    #     print(response)
    
    # print("============== Response End ==================")
    # print(str(response))
