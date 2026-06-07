
from common.Config import Config
from util.Util import Util


class SystemUtil(object):
    BASE_DIR = Util.get_base_path('startup.bat')
    CONFIG_DIR = BASE_DIR.joinpath('config')
    CONFIG = Config(CONFIG_DIR.joinpath('application.ini'))




