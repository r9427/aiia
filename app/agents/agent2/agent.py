import os
from pathlib import Path
import urllib.error
import urllib.request

from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from playwright.sync_api import sync_playwright


# ⚠️ 必须在导入 WebBaseLoader 之前设置 USER_AGENT
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser

from util.SystemUtil import SystemUtil

SYSTEM_PROMPT = """You are a excellent text extractor, you can summary a long text and extract key information.

## Capabilities

- `fetch_content_from_url_with_playwright`: fetches content from a URL using Playwright, suitable for pages that require JavaScript execution.

Do not guess line counts or positions—ground them in tool results from the saved file."""


@tool
async def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL.
    """
    req = urllib.request.Request(
        url,
        # "https://www.gutenberg.org/files/64317/64317-0.txt",
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text

@tool
async def fetch_content_from_url(url: str) -> str:
    """抓取指定url对应网页的内容"""

    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    loader = WebBaseLoader(
        web_path=url,
        requests_kwargs={
            "headers": custom_headers,
            "timeout": (5, 95)
        }
    )
    docs = loader.load()
    return docs[0].page_content

@tool
async def fetch_content_from_url_with_playwright(url: str) -> str:
    """使用 Playwright 抓取指定 URL 对应网页的内容，适用于需要执行 JavaScript 的网页"""

    sync_browser = create_sync_playwright_browser()
    try:
        page = sync_browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=320000)  # 等待网络空闲，最长等待120秒
        page.evaluate("() => { window.scrollTo(0, document.body.scrollHeight); }")  # 滚动到页面底部，触发懒加载
        page.wait_for_timeout(2000)  # 等待2秒，确保内容加载完成
        content = page.content()
        return content
    finally:
        sync_browser.close()

async def save_content(content: str):
    print("Saving content")
    output_dir = SystemUtil.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "post.md", "w", encoding='utf-8') as f:
        f.write(content)
    print("Post saved to output/post.md")


async def run_agent():

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

    checkpointer = InMemorySaver()

    # agent = create_agent(
    #     model=llm,
    #     tools=[fetch_text_from_url],
    #     system_prompt=SYSTEM_PROMPT,
    #     checkpointer=checkpointer,
    # )

    # 1. 初始化 Playwright 同步浏览器
    # sync_browser = create_sync_playwright_browser()

    # # 2. 构建 Playwright 浏览器工具箱并获取所有内置工具
    # toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    # tools = toolkit.get_tools() 
    # tools 列表中包含了 navigate_browser, extract_text, click_element 等强大工具


    deep_agent = create_deep_agent(
        model=llm,
        tools=[fetch_content_from_url_with_playwright],
        # tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    content = f"""Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
    from URL: https://www.gutenberg.org/files/64317/64317-0.txt

    Answer as much as you can:

    1) How many lines in the complete Gutenberg file contain the substring `Gatsby` (count lines, not occurrences within a line, each line ends with a line break).
    2) The 1-based line number of the first line in the file that contains `Daisy`.
    3) A two-sentence neutral synopsis.

    Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
    your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
    the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""

    content2 = f"""Get content from webpage https://walnuttree.xyz/documentation

    
    Summarize key information about the web page after the page loaded.
    """

    try:
        # agent_result = agent.invoke(
        #     {"messages": [{"role": "user", "content": content}]},
        #     config={"configurable": {"thread_id": "great-gatsby-lc"}},
        # )
        deep_agent_result = await deep_agent.ainvoke(
            {"messages": [{"role": "user", "content": content2}]},
            config={"configurable": {"thread_id": "website-explore-da"}},
        )
        # print(agent_result["messages"][-1].content_blocks)
        contents = deep_agent_result["messages"][-1].content_blocks
        print(contents)
        content = "No content fetched."
        if contents:
            first_block = contents[0]
            if isinstance(first_block, dict):
                content = first_block.get("content") or first_block.get("text") or first_block.get("message") or content
            else:
                content = getattr(first_block, "content", None) or getattr(first_block, "text", None) or content
        
        await save_content(content)

        print("\n")
    finally:
        pass
        # 7. 务必在任务结束后关闭浏览器释放资源
        # sync_browser.close() 