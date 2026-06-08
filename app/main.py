# from agents.agent1.entry import run as run1
# from agents.agent2.entry import run as run2
from agents.agent3.entry import run as run3
from util.SystemUtil import SystemUtil
from proxy_server import start_compat_proxy
from util.Util import Util


def run_app():
    run3()

def main():

    useSelfHostedModel = True
    
    if useSelfHostedModel:
        proxy_server, proxy_base_url, model_name, api_key = start_compat_proxy()
        SystemUtil.CONFIG.model_qwen_base_url = proxy_base_url
        SystemUtil.CONFIG.model_qwen_model_name = model_name
        SystemUtil.CONFIG.model_qwen_api_key = api_key

        try:
            print("Hello from aiia!")
            base_path = Util.get_base_path("startup.sh")
            print(f"Base path: {base_path}")

            run_app()
        
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()
    else:
        run_app()


if __name__ == "__main__":
    main()
