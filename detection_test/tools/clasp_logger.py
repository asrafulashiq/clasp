import os
import sys
import datetime
from loguru import logger
import logging
from colorama import Fore, Back
from tqdm import tqdm


class ClaspLogger():
    def __init__(self):
        self.logger = logger

        # sys.stdout logger
        self.logger.configure(handlers=[
            dict(
                sink=lambda x: tqdm.write(x, end=''),
                level='DEBUG',
                colorize=True,
                format=
                "<green>{time: MM-DD at HH:mm}</green>  <level>{message}</level>"
            )
        ])

        now = datetime.datetime.now()
        filename = os.path.join(
            "./logs", "{}_{}_{}_{}.txt".format(now.year, now.month, now.day,
                                               now.hour))

        self.logger.add(sink=filename,
                        level='INFO',
                        format="{time: MM-DD at HH:mm} | {message}")
        self.pre_msg = ""

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def addinfo(self, filenum, cam, frame):
        self.pre_msg = "{} - {}".format(filenum, frame)

    def log(self, msg, level="INFO"):
        self.logger.log(level, msg)

    def clasp_log(self, msg):
        self.logger.info("{} :: {}".format(self.pre_msg, msg))
        # self.info(msg)
