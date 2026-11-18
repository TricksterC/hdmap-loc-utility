import logging
import sys


def get_logger(name: str, level=logging.INFO, file=None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    logger.addHandler(h)
    if file:
        fh = logging.FileHandler(file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
