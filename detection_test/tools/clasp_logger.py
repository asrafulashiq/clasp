import os
import sys
import datetime
from loguru import logger
import logging
from colorama import Fore, Back


class ClaspLogger():
    def __init__(self, name="__clasp__"):
        now = datetime.datetime.now()
        filename = os.path.join("./logs",
                                "{}_{}_{}.txt".format(now.month, now.day, now.hour))
        self.logger = logger
        self.logger.configure(
            handlers=[
                dict(
                    sink=sys.stdout,
                    level='INFO',
                    colorize=True,
                    format="<green>{time: MM-DD at HH:mm}</green>  <level>{message}</level>"
                )
            ]
        )

        now = datetime.datetime.now()
        filename = os.path.join("./logs",
                                "{}_{}_{}.txt".format(now.month, now.day, now.hour))
        self.logger.add(
            sink=filename,
            level='DEBUG',
            format="{message}"
        )

        self.pre_msg = ""

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def addinfo(self, filenum, cam, frame):
        self.pre_msg = "{},{},{}".format(filenum, cam, frame)

    def clasp_log(self, msg):
        self.logger.debug("%s,%s", self.pre_msg, msg)
        self.info(msg)
