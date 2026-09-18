from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent, AgentStream
from llama_index.core.workflow import Context
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.tools.tavily_research import TavilyToolSpec

from util.SystemUtil import SystemUtil

import httpx


def collect_pages(page: dict) -> list[dict]:
    """Flatten a wiki page tree into a list of {path, url, remoteUrl, order, content}."""
    pages = [
        {
            "path": page.get("path"),
            "url": page.get("url"),
            "remoteUrl": page.get("remoteUrl"),
            "order": page.get("order"),
            "content": page.get("content"),
            "isParentPage": page.get("isParentPage", False),
        }
    ]
    for sub in page.get("subPages") or []:
        pages.extend(collect_pages(sub))
    return pages


async def fetch_wiki_page(url: str, auth: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url,
            params={
                "recursionLevel": "Full",
                "includeContent": "false",
                "api-version": "7.1",
            },
            headers={
                "Authorization": f"Basic {auth}",
            },
        )
        response.raise_for_status()
        return response.json()


async def run_test():
    azure_url = "https://dev.azure.com/Car-Software-Organisation/E2E_12/_apis/wiki/wikis/E2E_12_wiki/pages/32406"
    azure_auth = "OjJsUWxVY0Jtb255UDJiSzN4TWtKc1BYdU1pY1BWM2p4a0ZiOU9ROU5hUnZkWTZBcWtQRTNKUVFKOTlDR0FDQUFBQUFSTHJhQ0FBQVNBWkRPM1JORw=="

    page = await fetch_wiki_page(azure_url, azure_auth)
    pages = collect_pages(page)

    print(f"root: {page.get('path')} (id={page.get('id')})")
    print(f"total pages: {len(pages)}")
    for p in sorted(pages, key=lambda x: (x.get("order") is None, x.get("order", 0))):
        print(f"  [{p.get('order')}] {p.get('path')}")
