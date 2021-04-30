import time
import os
import psutil
import tracemalloc
from loguru import logger
import numpy as np
import torch
from collections import defaultdict
from contextlib import contextmanager
from prettytable import PrettyTable
import datetime


def get_time(time_posix, format="%H:%M:%S"):
    return datetime.datetime.fromtimestamp(time_posix).strftime(format)


class ComplexityAnalysis(object):
    def __init__(self) -> None:
        tracemalloc.start()
        self.process = psutil.Process(os.getpid())
        logger.info(f"process id: {self.process.pid}")

        self.list_time = []
        self.list_gpu_memory = []
        self.list_cpu_memory = []

        self.dict_time = {}
        self.dict_start = defaultdict(float)
        self.dict_end = defaultdict(float)
        self.dict_batch_mode = defaultdict(bool)

        self.frames_per_batch = 40

    @contextmanager
    def __call__(self, field=None, batch_mode=True, disable=False):
        if not disable:
            self.start(field, batch_mode)

        yield

        if not disable:
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

        self.dict_end[field] = time.time()
        time_passed = self.dict_end[field] - self.dict_start[field]
        self.dict_time[field].append(time_passed)

    def get_time_info(self):
        logger.info("Time info")
        table = PrettyTable(
            field_names=["Metric", "Avg", "Current", "Max", "Start", "End"])
        for k, vlist in self.dict_time.items():
            if len(vlist) == 0:
                continue
            if not self.dict_batch_mode[k]:
                vlist = [
                    sum(vlist[i:i + self.frames_per_batch])
                    for i in range(0, len(vlist), self.frames_per_batch)
                ]
            current = vlist[-1]
            avg, _max = np.mean(vlist), max(vlist)

            time_start = get_time(self.dict_start[k])
            time_end = get_time(self.dict_end[k])
            table.add_row([
                k,
                f"{avg:.4f}",
                f"{current:.4f}",
                f"{_max:.4f}",
                f"{time_start}",
                f"{time_end}",
            ])

        logger.info(f"\n{table.get_string()}\n")
        # np.save("ata_logs/time.npy", self.dict_time, allow_pickle=True)

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
        if self.list_gpu_memory:
            logger.debug(
                f"gpu memory: avg: {np.mean(self.list_gpu_memory)} MB, max: {max(self.list_gpu_memory)} MB"
            )
        if self.list_cpu_memory:
            logger.debug(
                f"cpu memory: avg: {np.mean(self.list_cpu_memory)} MB, max: {max(self.list_cpu_memory)} MB"
            )

    def close(self):
        tracemalloc.stop()