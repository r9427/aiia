import configparser
import os
from string import Template

from util.Util import Util


class Config(object):

    def __init__(self, config_file_path):
        parser = configparser.ConfigParser()
        parser.read_file(config_file_path.open())
        self.model_qwen_model_name = Util.strip_string(os.getenv("MODEL_QWEN_MODEL_NAME", None) or parser['model']['qwen_model_name'])
        self.model_qwen_embedding_name = Util.strip_string(os.getenv("MODEL_QWEN_EMBEDDING_NAME", None) or parser['model']['qwen_embedding_name'])
        self.model_qwen_api_key = Util.strip_string(os.getenv("MODEL_QWEN_API_KEY", None) or parser['model']['qwen_api_key'])
        self.model_qwen_base_url = Util.strip_string(os.getenv("MODEL_QWEN_BASE_URL", None) or parser['model']['qwen_base_url'])
        self.tools_tavily_api_key = Util.strip_string(os.getenv("TOOLS_TAVILY_API_KEY", None) or parser['tools']['tavily_api_key'])


