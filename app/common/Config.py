import configparser
import os
from string import Template

from util.Util import Util


class Config(object):

    def __init__(self, config_file_path):
        parser = configparser.ConfigParser()
        parser.read_file(config_file_path.open())
        self.model_name = Util.strip_string(os.getenv("MODEL_NAME", None) or parser['model']['name'])
        self.model_embedding_name = Util.strip_string(os.getenv("MODEL_EMBEDDING_NAME", None) or parser['model']['embedding_name'])
        self.model_api_key = Util.strip_string(os.getenv("MODEL_API_KEY", None) or parser['model']['api_key'])
        self.model_base_url = Util.strip_string(os.getenv("MODEL_BASE_URL", None) or parser['model']['base_url'])
        self.postgresql_host = Util.strip_string(os.getenv("POSTGRESQL_HOST", None) or parser['postgresql']['host'])
        self.postgresql_port = int(Util.strip_string(os.getenv("POSTGRESQL_PORT", None) or parser['postgresql']['port']))
        self.postgresql_name = Util.strip_string(os.getenv("POSTGRESQL_NAME", None) or parser['postgresql']['name'])
        self.postgresql_schema = Util.strip_string(os.getenv("POSTGRESQL_SCHEMA", None) or parser['postgresql']['schema'])
        self.postgresql_username = Util.strip_string(os.getenv("POSTGRESQL_USERNAME", None) or parser['postgresql']['username'])
        self.postgresql_password = Util.strip_string(os.getenv("POSTGRESQL_PASSWORD", None) or parser['postgresql']['password'])
        self.postgresql_sslmode = Util.strip_string(os.getenv("POSTGRESQL_SSLMODE", None) or parser['postgresql']['sslmode'])
        self.mysql_host = Util.strip_string(os.getenv("MYSQL_HOST", None) or parser['mysql']['host'])
        self.mysql_port = int(Util.strip_string(os.getenv("MYSQL_PORT", None) or parser['mysql']['port']))
        self.mysql_name = Util.strip_string(os.getenv("MYSQL_NAME", None) or parser['mysql']['name'])
        self.mysql_username = Util.strip_string(os.getenv("MYSQL_USERNAME", None) or parser['mysql']['username'])
        self.mysql_password = Util.strip_string(os.getenv("MYSQL_PASSWORD", None) or parser['mysql']['password'])
        self.qdrant_host = Util.strip_string(os.getenv("QDRANT_HOST", None) or parser['qdrant']['host'])
        self.qdrant_port = int(Util.strip_string(os.getenv("QDRANT_PORT", None) or parser['qdrant']['port']))
        self.tools_tavily_api_key = Util.strip_string(os.getenv("TOOLS_TAVILY_API_KEY", None) or parser['tools']['tavily_api_key'])

    def get_db_url(self):
        # return self.get_pg_url()
        return self.get_mysql_url()

    def get_pg_url(self):
        return Template(
            'postgresql+asyncpg://${db_username}:${db_password}@${db_host}:${db_port}/${db_name}'
        ).substitute(db_username=self.postgresql_username,
                     db_password=self.postgresql_password,
                     db_host=self.postgresql_host,
                     db_port=self.postgresql_port,
                     db_name=self.postgresql_name)
    
    def get_mysql_url(self):
        # return Template(
        #     'mysql+asyncmy://${db_username}:${db_password}@${db_host}:${db_port}/${db_name}'
        # ).substitute(db_username=self.mysql_username,
        #              db_password=self.mysql_password,
        #              db_host=self.mysql_host,
        #              db_port=self.mysql_port,
        #              db_name=self.mysql_name)
        return Template(
            'mysql+pymysql://${db_username}:${db_password}@${db_host}:${db_port}/${db_name}'
        ).substitute(db_username=self.mysql_username,
                     db_password=self.mysql_password,
                     db_host=self.mysql_host,
                     db_port=self.mysql_port,
                     db_name=self.mysql_name)

