from pathlib import Path

from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent, AgentStream, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.workflow import (
    Context,
    InputRequiredEvent,
    HumanResponseEvent,
)
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

def delete_file(path: str) -> str:
    """Delete a file and returns a success message if successful, an error message otherwise"""
    try:
        target_path = Path(path)
        print(f"Attempting to delete file at path: {target_path}")
        if target_path.exists():
            target_path.unlink()
        return "File deleted successfully."
    except OSError:
        return "Error occurred while deleting the file."

# a tool that performs a dangerous task
async def dangerous_task(ctx: Context) -> str:
    """A dangerous task to delete files that requires human confirmation."""

    print("Performing a dangerous task that requires human confirmation...")
    # emit an event to the external stream to be captured
    ctx.write_event_to_stream(
        InputRequiredEvent(
            prefix="Are you sure you want to proceed? ",
            user_name="Locke",
        )
    )

    # wait until we see a HumanResponseEvent
    response = await ctx.wait_for_event(
        HumanResponseEvent, requirements={"user_name": "Locke"}
    )

    # act on the input from the event
    if response.response.strip().lower() == "yes":
        delete_file("C:/uvw/work/project/aiia/app/agents/llama_agent4/test.txt")
        return "Dangerous task completed successfully."
    else:
        return "Dangerous task aborted."


async def run_agent():
    print("model config name='{}', api_key='{}', base_url='{}'".format(
        SystemUtil.CONFIG.model_name,
        SystemUtil.CONFIG.model_api_key,
        SystemUtil.CONFIG.model_base_url
    ))

    llm = OpenAILike(
        model=SystemUtil.CONFIG.model_name,
        api_base=SystemUtil.CONFIG.model_base_url,
        api_key=SystemUtil.CONFIG.model_api_key,
        context_window=128000,
        is_chat_model=True,
        is_function_calling_model=True,
    )

    workflow = FunctionAgent(
        tools=[dangerous_task],
        llm=llm,
        system_prompt="You are a helpful assistant that can perform dangerous tasks.",
    )
    ctx = Context(workflow)


    user_msgs = [
        "What's the weather like in Shanghai ?",
        "I want to proceed with the dangerous task to delete file test.txt."
    ]

    handler = workflow.run(ctx=ctx, user_msg=user_msgs[1])

    async for event in handler.stream_events():
        # capture InputRequiredEvent
        if isinstance(event, InputRequiredEvent):
            # capture keyboard input
            response = input(event.prefix)
            # send our response back
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response,
                    user_name=event.user_name,
                )
            )

    response = await handler
    print(str(response))
