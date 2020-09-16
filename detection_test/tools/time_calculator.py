import time
import os
import psutil
import tracemalloc
from loguru import logger
import numpy as np
import torch
from collections import defaultdict


class ComplexityAnalysis(object):
    def __init__(self) -> None:
        tracemalloc.start()
        self.process = psutil.Process(os.getpid())
        logger.info(f"process id: {self.process.pid}")

        self.list_time = []
        self.list_gpu_memory = []
        self.list_cpu_memory = []

        self.dict_time = {}
        self.dict_start = {}
        self.dict_start = defaultdict(float)

    def start(self, field=None):
        if field is None:
            raise ValueError("Provide a name for this timing")

        if field not in self.dict_time:
            self.dict_time[field] = []
        self.dict_start[field] = time.time()

    def pause(self, field=None):
        if field is None:
            raise ValueError("Provide a name for this timing")
        assert field in self.dict_time
        assert field in self.dict_start

        time_passed = time.time() - self.dict_start[field]
        self.dict_time[field].append(time_passed)

    def get_time_info(self):
        logger.info("Time info")
        for k in self.dict_time:
            if len(self.dict_time[k]) == 0:
                continue
            avg, _max = np.mean(self.dict_time[k]), max(self.dict_time[k])
            logger.info(f"{k} :: avg : {avg:.4f} s, max : {_max:.4f} s")

    def process_memory(self):
        memory_mb = round(self.process.memory_info()[0] / (1024 * 1024))
        self.list_cpu_memory.append(memory_mb)
        logger.info(f"Process memory mb : {memory_mb} MB")
        return memory_mb

    def current_memory_usage(self):
        # current, peak = tracemalloc.get_traced_memory()
        max_gpu = torch.cuda.max_memory_allocated(
            device=torch.device('cuda:0'))
        current = round(self.process.memory_info()[0])

        logger.info(
            f"Current memory usage is {current / (1024 * 1024)}MB; Max gpu: {max_gpu / (1024 * 1024)}MB"
        )

        self.list_cpu_memory.append(current / (1024 * 1024))
        self.list_gpu_memory.append(max_gpu / (1024 * 1024))

        memory_mb = round(self.process.memory_info()[0] / (1024 * 1024))
        logger.info(f"Process memory mb : {memory_mb} MB")
        return current

    def final_info(self):
        logger.debug(
            f"gpu memory: avg: {np.mean(self.list_gpu_memory)} MB, max: {max(self.list_gpu_memory)} MB"
        )
        logger.debug(
            f"cpu memory: avg: {np.mean(self.list_cpu_memory)} MB, max: {max(self.list_cpu_memory)} MB"
        )

    def close(self):
        tracemalloc.stop()