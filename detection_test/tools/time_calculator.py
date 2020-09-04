import time
import os
import psutil
import tracemalloc
from loguru import logger
import numpy as np
import torch


class ComplexityAnalysis(object):
    def __init__(self) -> None:
        tracemalloc.start()
        self.process = psutil.Process(os.getpid())
        logger.info(f"process id: {self.process.pid}")

        self.list_time = []
        self.list_memory = []

    def start_time(self):
        self.time_0 = time.time()

    def batch_end_time(self):
        time_now = time.time()
        time_passed = time_now - self.time_0
        self.list_time.append(time_passed)

        logger.info(f"time passed {time_passed} sec")

    def process_memory(self):
        memory_mb = round(self.process.memory_info()[0] / (1024 * 1024))
        logger.info(f"Process memory mb : {memory_mb} MB")
        return memory_mb

    def current_memory_usage(self):
        current, peak = tracemalloc.get_traced_memory()
        max_gpu = torch.cuda.max_memory_allocated(
            device=torch.device('cuda:0'))

        logger.info(
            f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB, Max gpu: {max_gpu / 10**6}MB"
        )
        self.list_memory.append(current)
        return current, peak

    def final_info(self):
        mean_time = np.mean(self.list_time)
        max_time = np.max(self.list_time)
        logger.info(
            f" avegare batch time: {mean_time}, maximum batch time: {max_time} "
        )

        logger.debug(f"average gpu memory: {np.mean(self.list_memory)}")

    def close(self):
        tracemalloc.stop()