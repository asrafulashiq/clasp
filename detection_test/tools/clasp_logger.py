import logging
import os
import datetime

class ClaspLogger():
    def __init__(self, name="__clasp__"):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        self.logger.addHandler(ch)

        now = datetime.datetime.now()
        filename = os.path.join("./logs",
                                "{}_{}_{}.txt".format(now.month, now.day, now.hour))
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        self.clasp_logger = logging.getLogger("clasp")
        self.clasp_logger.setLevel(logging.DEBUG)
        self.clasp_logger.addHandler(fh)

        self.pre_msg = ""

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def addinfo(self, filenum, cam, frame):
        self.pre_msg = "{},{},{}".format(filenum, cam, frame)

    def clasp_log(self, msg):
        self.clasp_logger.info("%s,%s", self.pre_msg, msg)
        self.info(msg)
