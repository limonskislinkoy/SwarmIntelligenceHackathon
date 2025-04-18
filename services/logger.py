import logging
from pathlib import Path

from config import LOG_DIR


def set_logger_config(log_level="INFO"):
    Path("logs/").mkdir(parents=True, exist_ok=True)

    level = logging.INFO
    filename = "info.log"

    if log_level == "DEBUG":
        level = logging.DEBUG
        filename = "debug.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=LOG_DIR + filename
    )
