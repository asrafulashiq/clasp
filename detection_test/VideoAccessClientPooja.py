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


tracemalloc.start()
current_ram_usage = []

myurl = 'http://127.0.0.1:5000/frames'

parser = argparse.ArgumentParser()
parser.add_argument("--start_frame", "-s", type=str, default='120.00')
args = parser.parse_args()

#dataset = str(input("Enter the dataset:"))
dataset = "exp2training"
cameralist = '2,5,9,11,13,14,20'
timeoffset = args.start_frame
duration = 4.0
duration = "{0:.2f}".format(float(duration))
framerate = "10"
count = 0
filesize = "720x1080"  #or "720x1080"

inputdir = "/data/ALERT-SHARE/alert-api-wrapper-data/"
Wrapper_Flag_File = "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_Wrapper.yaml"
RPI_Flag_File = "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_RPI.yaml"
NEU_Flag_File = "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_NEU.yaml"
MU_Flag_File = "/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/Flags_MU.yaml"
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
    "Batch_Processed": "FLASE",
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
overall_batch_times = []
wrapper_batch_times = []
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
            "filesize": filesize,
            "inputdir": inputdir,
            "framerate": framerate
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
        wrapper_batch_start_time = time.time()
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
            overall_batch_elapsed_time = time.time() - overall_batch_start_time
            overall_batch_times.append(overall_batch_elapsed_time)
            print("Batch Processing Time: " + time.strftime(
                "%H:%M:%S", time.gmtime(overall_batch_elapsed_time)))
        overall_batch_start_time = time.time()
        inputParams = {
            "dataset": dataset,
            "cameralist": cameralist,
            "timeoffset": timeoffset,
            "duration": duration,
            "filesize": filesize,
            "inputdir": inputdir,
            "framerate": framerate
        }
        jsoninputParams = json.dumps(inputParams)
        jsonParams = {"APIParams": jsoninputParams}
        response = requests.get(url=myurl, params=jsonParams)
        wrapper_batch_elapsed_time = time.time() - wrapper_batch_start_time
        wrapper_batch_times.append(wrapper_batch_elapsed_time)
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
            overall_avg_batch_time = sum(overall_batch_times) / len(
                overall_batch_times)
            print(
                "Average Oerall Batch Processing Time: " +
                time.strftime("%H:%M:%S", time.gmtime(overall_avg_batch_time)))
            overall_max_batch_time = max(overall_batch_times)
            print(
                "Max Overall Batch Processing Time: " +
                time.strftime("%H:%M:%S", time.gmtime(overall_max_batch_time)))
            wrapper_avg_batch_time = sum(wrapper_batch_times) / len(
                wrapper_batch_times)
            print("Average Wrapper Batch Processing Time: " +
                  str(wrapper_avg_batch_time))
            wrapper_max_batch_time = max(wrapper_batch_times)
            print("Max Wrapper Batch Processing Time: " +
                  str(wrapper_max_batch_time))
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
