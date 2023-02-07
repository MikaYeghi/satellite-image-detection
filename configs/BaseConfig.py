from logger import get_logger

class BaseConfig:
    def __init__(self):
        self.logger = get_logger("Config logger")