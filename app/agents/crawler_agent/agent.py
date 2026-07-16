from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from playwright.async_api import async_playwright

from util.SystemUtil import SystemUtil

SYSTEM_PROMPT = """You are an expert website researcher and summarizer.
Use the website extraction tools to retrieve actual page text, then summarize the content accurately.
If the page contains many sections, provide a concise summary and list the most important points.
Do not invent facts; if the page content cannot be fully retrieved, say so clearly.
"""


@tool
async def fetch_website_text(url: str) -> str:
    """Fetch rendered website text using Playwright."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_default_navigation_timeout(1000 * 120)
            await page.set_user_agent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
            )
            await page.goto(url, wait_until="domcontentloaded", timeout=1000 * 120)
            await page.wait_for_timeout(1500)
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(800)

            try:
                button = page.get_by_text("Documentation")
                if await button.count() > 0:
                    await button.first.click()
                    await page.wait_for_timeout(1000 * 5)
            except Exception:
                pass

            body_text = await page.inner_text("body")
            return body_text
        except Exception as exc:
            return f"Error fetching website text: {exc}"
        finally:
            await browser.close()


async def run_agent():
    url = "https://baike.baidu.com/item/%E9%9F%A9%E4%BF%A1/5321"
    # url = "https://walnuttree.xyz/"
    # url = "https://developers.llamaindex.ai/python/framework/use_cases/chatbots/"
    # url = "https://react.dev/"
    # url = "https://docs.astral.sh/uv/guides/tools/#installing-tools"

    print("Starting async website summarization agent...")

    llm = ChatOpenAI(
        model=SystemUtil.CONFIG.model_name,
        api_key=SystemUtil.CONFIG.model_api_key,
        base_url=SystemUtil.CONFIG.model_base_url,
        temperature=0.2,
    )

    tools = [fetch_website_text]

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    user_prompt = (
        f"Retrieve and summarize the website content from the URL: {url}. "
        "Use the fetch_website_text tool to inspect the page and provide a concise, accurate summary."
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )
    final_message = result["messages"][-1].content
    print("\n=== Website Summary ===\n")
    print(final_message)
    return final_message

