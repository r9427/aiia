from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent, AgentStream
from llama_index.core.workflow import Context
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.tools.tavily_research import TavilyToolSpec

from util.SystemUtil import SystemUtil


tavily_tool = TavilyToolSpec(api_key=SystemUtil.CONFIG.tools_tavily_api_key)

# 全局设置 Qwen Embedding 模型
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

    llm = OpenAILike(
        model=SystemUtil.CONFIG.model_name,
        api_base=SystemUtil.CONFIG.model_base_url,
        api_key=SystemUtil.CONFIG.model_api_key,
        context_window=128000,
        is_chat_model=True,
        is_function_calling_model=True,
    )

    # Create an agent workflow with our calculator tool
    agent = FunctionAgent(
        tools=tavily_tool.to_tool_list(),
        llm=llm,
        system_prompt="You are an agent that can perform basic mathematical operations using tools. You can set a name. You can search web for information.",
        initial_state={"name": "unset"}
    )

    # create context
    ctx = Context(agent)
    # run agent with context
    # response = await agent.run(ctx=ctx, user_msg="My name is Logan")

    user_msgs = [
        "What's the weather like in Shanghai ?"
    ]
    handler = agent.run(ctx=ctx, user_msg=user_msgs[0])

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)

    
