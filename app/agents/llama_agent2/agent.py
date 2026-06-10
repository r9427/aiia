from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.embeddings.dashscope import DashScopeEmbedding

from util.SystemUtil import SystemUtil


# 全局设置 Qwen Embedding 模型
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",  # 千问轻量级嵌入模型，性价比高
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


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

async def set_name(ctx: Context, name: str) -> str:
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["name"] = name

    return f"Name set to {name}"

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
        tools=[multiply, add, set_name],
        llm=llm,
        system_prompt="You are an agent that can perform basic mathematical operations using tools. You can set a name.",
        initial_state={"name": "unset"}
    )

    # create context
    ctx = Context(agent)
    # run agent with context
    # response = await agent.run(ctx=ctx, user_msg="My name is Logan")

    user_msgs = [
        # "What is 20+(2*4)?",
        "What's my initial name ?",
        "My name is Locke !",
        "What's my name again ?"
    ]

    for index, msg in enumerate(user_msgs):
        response = await agent.run(ctx=ctx, user_msg=msg)
        state = await ctx.store.get("state")
        print(f"============= Round {index + 1} =================")
        print(f"User: {msg}")
        print(response)
        print(f"name = ({state['name']})")

    print("============= End =================")
    
