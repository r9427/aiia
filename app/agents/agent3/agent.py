import os
from pathlib import Path
from typing import Literal
import csv, io

from langchain.agents import create_agent
from langsmith.sandbox import SandboxClient
from langchain.agents.middleware import TodoListMiddleware
from deepagents.backends.langsmith import LangSmithSandbox
from deepagents.middleware import (
    FilesystemMiddleware,
    SkillsMiddleware,
    SubAgentMiddleware,
    SummarizationMiddleware
)
from deepagents.backends import LocalShellBackend, StateBackend
from deepagents import create_deep_agent, SubAgent
from langchain_openai import ChatOpenAI

from util.SystemUtil import SystemUtil

# from common.utils import get_output_path



# client = SandboxClient(api_key=os.getenv("LANGSMITH_API_KEY"))
# sandbox = client.create_sandbox()
# backend = LangSmithSandbox(sandbox=sandbox)
# backend = StateBackend()
targetPath = SystemUtil.BASE_DIR.joinpath("app/agents/agent3")
backend = LocalShellBackend(
    root_dir=targetPath,
    virtual_mode=True,
    # env={"PATH": "/usr/bin:/bin"}
)

def get_agent():

    print("qwen model config name='{}', api_key='{}', base_url='{}'".format(
        SystemUtil.CONFIG.model_qwen_model_name,
        SystemUtil.CONFIG.model_qwen_api_key,
        SystemUtil.CONFIG.model_qwen_base_url
    ))
    llm = ChatOpenAI(
        model=SystemUtil.CONFIG.model_qwen_model_name,
        api_key=SystemUtil.CONFIG.model_qwen_api_key,
        base_url=SystemUtil.CONFIG.model_qwen_base_url
    )

    visualizer: SubAgent = {
        "name": "visualizer",
        "model": llm,
        "description": "Generates charts and visualizations from data files in the working directory. Uses matplotlib and seaborn. Saves all figures as PNG files in the working directory. Check if the results are saved, and tell the user.",
        "system_prompt": "You are a data visualization specialist. Write Python scripts using matplotlib and seaborn. Save all figures as PNG files.",
        "tools": [],
    }

    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            FilesystemMiddleware(backend=backend),
            SummarizationMiddleware(model=llm, backend=backend),
            SkillsMiddleware(backend=backend, sources=["./skills/"]),
            TodoListMiddleware(),
            SubAgentMiddleware(backend=backend, subagents=[visualizer]),
        ]
    )

    return agent


def run_agent():
    # Create sample sales data
    data = [
        ["Date", "Product", "Units Sold", "Revenue"],
        ["2025-08-01", "Widget A", 10, 250],
        ["2025-08-02", "Widget B", 5, 125],
        ["2025-08-03", "Widget A", 7, 175],
        ["2025-08-04", "Widget C", 3, 90],
        ["2025-08-05", "Widget B", 8, 200],
    ]

    # Convert to CSV bytes
    text_buf = io.StringIO()
    writer = csv.writer(text_buf)
    writer.writerows(data)
    csv_bytes = text_buf.getvalue().encode("utf-8")
    text_buf.close()

    # Upload to backend
    backend.upload_files([(str(targetPath.joinpath("sales_data.csv")), csv_bytes)])
    # backend.download_files(list_of_filepaths)
    agent = get_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Analyze sales_data.csv. Summarize trends."}]
    })

    print(result["messages"][-1].content)
