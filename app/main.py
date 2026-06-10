import asyncio
from util.SystemUtil import SystemUtil
from proxy_server import start_compat_proxy
from util.Util import Util

# from agents.agent1.entry import run as run_agent
# from agents.agent2.entry import run as run_agent
# from agents.agent3.entry import run as run_agent
# from agents.llama_agent1.entry import run as run_agent
# from agents.llama_agent2.entry import run as run_agent
from agents.llama_agent3.entry import run as run_agent


async def run_app():
    await run_agent()

async def main():

    useSelfHostedModel = False
    
    if useSelfHostedModel:
        proxy_server, proxy_base_url, model_name, api_key = start_compat_proxy()
        SystemUtil.CONFIG.model_qwen_base_url = proxy_base_url
        SystemUtil.CONFIG.model_qwen_model_name = model_name
        SystemUtil.CONFIG.model_qwen_api_key = api_key

        try:
            # base_path = Util.get_base_path("startup.sh")
            # print(f"Base path: {base_path}")

            await run_app()
        
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()
    else:
        await run_app()


if __name__ == "__main__":
    asyncio.run(main())
