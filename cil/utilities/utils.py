import enum
import logging
import os
import time


class OutputType(enum.Enum):
    LOGIT = 0
    PROBABILITY = 1


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_name(variable):
    """ Get the class name of an variable """
    return type(variable).__name__


def get_logger(log_dir, model):
    # file name: model_start-time.log
    timestamp = time.localtime()
    time_string = time.strftime("%m-%d_%H-%M", timestamp)
    file_name = get_name(model) + '_' + time_string + '.log'

    ensure_dir(log_dir)
    path = os.path.join(log_dir, file_name)

    # Set up logger
    logger = logging.getLogger('cil_logger')
    logger.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s      %(message)s')
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
