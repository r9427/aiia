from pathlib import Path

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from util.SystemUtil import SystemUtil


SYSTEM_PROMPT = """
You are a helpful AI assistant.
"""

async def run_agent():

    llm = ChatOpenAI(
        model=SystemUtil.CONFIG.model_name,
        api_key=SystemUtil.CONFIG.model_api_key,
        base_url=SystemUtil.CONFIG.model_base_url,
        temperature=0.2
    )

    checkpointer = InMemorySaver()

    agent = create_deep_agent(
        model=llm,
        # tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer
    )

    messages = []
    config = {"configurable": {"thread_id": "12345"}}  # Unique thread ID for memory

    print("Deep Agent Chat (type 'q' to exit)")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        
        # Append user message
        messages.append({"role": "user", "content": user_input})
        
        # Invoke the agent
        result = await agent.ainvoke(
            {"messages": messages},
            config=config
        )
        
        # Update message history with the agent's response
        messages = result["messages"]
        latest_message = messages[-1]
        
        # Print the agent's response
        content = getattr(latest_message, "content", None)
        if content:
            print(f"\nAssistant: {content}")
