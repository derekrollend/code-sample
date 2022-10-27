import logging


def get_simple_logger(
    name=__name__, base_level=logging.DEBUG, console_level=logging.DEBUG
):
    logger = logging.getLogger(name)
    formatter = logging.Formatter("%(asctime)s: (%(name)s) %(levelname)s: %(message)s")
    logger.setLevel(level=base_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
