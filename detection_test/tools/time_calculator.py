import time
import os
import psutil
import tracemalloc
from loguru import logger
import numpy as np
import torch
from collections import defaultdict
from contextlib import contextmanager


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
        self.dict_batch_mode = defaultdict(bool)

        self.frames_per_batch = 1

    @contextmanager
    def __call__(self, field=None, batch_mode=True):
        self.start(field, batch_mode)
        yield
        self.pause(field)

    def start(self, field=None, batch_mode=True):
        if field is None:
            raise ValueError("Provide a name for this timing")

        if field not in self.dict_time:
            self.dict_time[field] = []
        self.dict_batch_mode[field] = batch_mode
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
            if not self.dict_batch_mode[k]:
                avg *= self.frames_per_batch
                _max *= self.frames_per_batch

            logger.info(f"{k} :: avg : {avg:.4f} s, max : {_max:.4f} s")

        np.save("ata_logs/time.npy", self.dict_time, allow_pickle=True)

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
        self.get_time_info()
        logger.debug(
            f"gpu memory: avg: {np.mean(self.list_gpu_memory)} MB, max: {max(self.list_gpu_memory)} MB"
        )
        logger.debug(
            f"cpu memory: avg: {np.mean(self.list_cpu_memory)} MB, max: {max(self.list_cpu_memory)} MB"
        )

    def close(self):
        tracemalloc.stop()