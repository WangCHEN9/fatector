from fatector.get_solid import GetSolid
from loguru import logger


logger.add("./log/file_{time}.log", rotation="00:00")
logger.debug("That's it, beautiful and simple logging!")
