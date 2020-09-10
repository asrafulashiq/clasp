import requests
import json
import zipfile
import io
import os
import yaml
import time
import psutil
import tracemalloc
import argparse


def GetFlags(flag_file):
    flags = {}
    with open(flag_file, 'r') as stream:
        flags = yaml.safe_load(stream)
    return flags


def WriteFlags(flags, flag_file):
    with open(flag_file, 'w') as file:
        yaml.dump(flags, file)


def CleanInputDir(inputdir):
    framelist = [f for f in os.listdir(inputdir) if f.endswith(".jpg")]
    for f in framelist:
        os.remove(os.path.join(inputdir, f))
    return


parser = argparse.ArgumentParser()
parser.add_argument("--start_frame", type=float, default=120.0)
parser.add_argument("--duration", type=float, default=4)
args = parser.parse_args()

tracemalloc.start()
process = psutil.Process(os.getpid())
current_ram_usage = []

myurl = 'http://127.0.0.1:5000/frames'

#dataset = str(input("Enter the dataset:"))
dataset = "exp2training"
cameralist = '2,5,9,11,13,14,20'
timeoffset = args.start_frame
duration = args.duration
duration = "{0:.2f}".format(float(duration))
count = 0

inputdir = "/home/rpi/data/wrapper_data/"
Wrapper_Flag_File = "/home/rpi/data/wrapper_log/Flags_Wrapper.yaml"
RPI_Flag_File = "/home/rpi/data/wrapper_log/Flags_RPI.yaml"
NEU_Flag_File = "/home/rpi/data/wrapper_log/Flags_NEU.yaml"
MU_Flag_File = "/home/rpi/data/wrapper_log/Flags_MU.yaml"
CleanInputDir(inputdir)
#Set wrapper flag defaults
wrapperFlags = {
    "Frames_Ready_MU": "FALSE",
    "Frames_Ready_NEU": "FALSE",
    "Frames_Ready_RPI": "FALSE",
    "No_More_Frames": "FALSE",
    "Next_Batch_Requested": "FALSE"
}
#Set RPI flag defaults
RPIFlags = {
    "Bin_Processed": "FALSE",
    "Batch_Processed": "FALSE",
}

#Set MU flag defaults
MUFlags = {"People_Processed": "FALSE"}

#Set NEU flag defaults
NEUFlags = {"Association_Ready": "FALSE"}

#Write all flag Defaults
try:
    WriteFlags(wrapperFlags, Wrapper_Flag_File)
    WriteFlags(RPIFlags, RPI_Flag_File)
    WriteFlags(NEUFlags, NEU_Flag_File)
    WriteFlags(MUFlags, MU_Flag_File)
except Exception as ex:
    print(ex)

current, peak = tracemalloc.get_traced_memory()
current_ram_usage.append(current)
batch_times = []
total_start_time = time.time()
batchIteration = 0
while True:
    rpiFlags = GetFlags(RPI_Flag_File)
    if rpiFlags is None:
        continue
    if batchIteration == 0:
        # Reset frame ready flags and reset next batch flag
        while True:
            wrapperFlags = GetFlags(Wrapper_Flag_File)
            if wrapperFlags is None:
                continue
            else:
                break
        for flag in wrapperFlags:
            if str(flag).startswith("Frames"):
                wrapperFlags[flag] = "FALSE"
            if str(flag).startswith("Next"):
                wrapperFlags[flag] = "TRUE"
        try:
            WriteFlags(wrapperFlags, Wrapper_Flag_File)
        except Exception as ex:
            print(ex)
        inputParams = {
            "dataset": dataset,
            "cameralist": cameralist,
            "timeoffset": timeoffset,
            "duration": duration,
            "filesize": "1920x1080",
            "inputdir": inputdir
        }
        jsoninputParams = json.dumps(inputParams)
        jsonParams = {"APIParams": jsoninputParams}
        response = requests.get(url=myurl, params=jsonParams)
        current, peak = tracemalloc.get_traced_memory()
        current_ram_usage.append(current)
        while True:
            wrapperFlags = GetFlags(Wrapper_Flag_File)
            if wrapperFlags is None:
                continue
            else:
                break
        for flag in wrapperFlags:
            if str(flag).startswith("Frames"):
                wrapperFlags[flag] = "TRUE"
            if str(flag).startswith("Next"):
                wrapperFlags[flag] = "FALSE"
        try:
            WriteFlags(wrapperFlags, Wrapper_Flag_File)
        except Exception as ex:
            print(ex)
        current, peak = tracemalloc.get_traced_memory()
        print(
            f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB"
        )
        current_ram_usage.append(current)
        timeoffset = float(timeoffset) + float(duration)
        timeoffset = round(timeoffset, 2)
        count = count + 1
        print("Batch count : " + str(count))
        batchIteration = batchIteration + 1
    elif rpiFlags["Batch_Processed"] == "TRUE":
        # Reset frame ready flags and reset next batch flag
        while True:
            wrapperFlags = GetFlags(Wrapper_Flag_File)
            if wrapperFlags is None:
                continue
            else:
                break
        for flag in wrapperFlags:
            if str(flag).startswith("Frames"):
                wrapperFlags[flag] = "FALSE"
            if str(flag).startswith("Next"):
                wrapperFlags[flag] = "TRUE"
        try:
            WriteFlags(wrapperFlags, Wrapper_Flag_File)
        except Exception as ex:
            print(ex)
        if batchIteration != 1:
            batch_elapsed_time = time.time() - batch_start_time
            batch_times.append(batch_elapsed_time)
            print("Batch Processing Time: " +
                  time.strftime("%H:%M:%S", time.gmtime(batch_elapsed_time)))
        batch_start_time = time.time()
        inputParams = {
            "dataset": dataset,
            "cameralist": cameralist,
            "timeoffset": timeoffset,
            "duration": duration,
            "filesize": "1920x1080",
            "inputdir": inputdir
        }
        jsoninputParams = json.dumps(inputParams)
        jsonParams = {"APIParams": jsoninputParams}
        response = requests.get(url=myurl, params=jsonParams)
        current, peak = tracemalloc.get_traced_memory()
        current_ram_usage.append(current)
        if response.text == 'No More Frames':
            print('Reached end of video!')
            while True:
                wrapperFlags = GetFlags(Wrapper_Flag_File)
                if wrapperFlags is None:
                    continue
                else:
                    break
            wrapperFlags["No_More_Frames"] = "TRUE"
            try:
                WriteFlags(wrapperFlags, Wrapper_Flag_File)
            except Exception as ex:
                print(ex)
            total_elapsed_time = time.time() - total_start_time
            print("Total Processing Time: " +
                  time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time)))
            avg_batch_time = sum(batch_times) / len(batch_times)
            print("Average Batch Processing Time: " +
                  time.strftime("%H:%M:%S", time.gmtime(avg_batch_time)))
            max_batch_time = sum(batch_times) / len(batch_times)
            print("Max Batch Processing Time: " +
                  time.strftime("%H:%M:%S", time.gmtime(max_batch_time)))
            current, peak = tracemalloc.get_traced_memory()
            current_ram_usage.append(current)
            print(
                f"Average memory usage is {(sum(current_ram_usage) / len(current_ram_usage)) / 10 ** 6}MB; "
            )
            print(f"Peak was {peak / 10 ** 6}MB")
            tracemalloc.stop()
            break
        else:
            while True:
                wrapperFlags = GetFlags(Wrapper_Flag_File)
                if wrapperFlags is None:
                    continue
                else:
                    break
            for flag in wrapperFlags:
                if str(flag).startswith("Frames"):
                    wrapperFlags[flag] = "TRUE"
                if str(flag).startswith("Next"):
                    wrapperFlags[flag] = "FALSE"
            try:
                WriteFlags(wrapperFlags, Wrapper_Flag_File)
            except Exception as ex:
                print(ex)
            current, peak = tracemalloc.get_traced_memory()
            current_ram_usage.append(current)
            timeoffset = float(timeoffset) + float(duration)
            timeoffset = round(timeoffset, 2)
            count = count + 1
            print("Batch count : " + str(count))
            batchIteration = batchIteration + 1
