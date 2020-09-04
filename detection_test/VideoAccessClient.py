import requests
import json
import zipfile
import io
import os
import yaml
import time
import psutil
import tracemalloc


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
process = psutil.Process(os.getpid())
current_ram_usage = []

myurl = 'http://127.0.0.1:5000/frames'

dataset = str(input("Enter the dataset:"))
cameralist = '2,5,9,11,13,14,20'
timeoffset = '0.00'
duration = 4.0
duration = "{0:.2f}".format(float(duration))

inputdir = "/home/rpi/data/wrapper_data/"
Wrapper_Flag_File = "/home/rpi/data/wrapper_log/FrameSyncFlags.yaml"
RPI_Flag_File = "/home/rpi/data/wrapper_log/Flags_RPI.yaml"
CleanInputDir(inputdir)
#Set wrapper flag defaults
wrapperFlags = GetFlags(Wrapper_Flag_File)
for flag in wrapperFlags:
    wrapperFlags[flag] = "FALSE"
WriteFlags(wrapperFlags, Wrapper_Flag_File)

#Set RPI flag default
while True:
    try:
        rpiFlags = GetFlags(RPI_Flag_File)
        rpiFlags["Batch_Processed"] = "TRUE"
        WriteFlags(rpiFlags, RPI_Flag_File)
        break
    except Exception as ex:
        print(ex)
        continue
current, peak = tracemalloc.get_traced_memory()
current_ram_usage.append(current)
batch_times = []
total_start_time = time.time()

while True:
    rpiFlags = GetFlags(RPI_Flag_File)
    if rpiFlags["Batch_Processed"] == "TRUE":

        #Reset RPI flag
        while True:
            try:
                rpiFlags = GetFlags(RPI_Flag_File)
                rpiFlags["Batch_Processed"] = "FALSE"
                WriteFlags(rpiFlags, RPI_Flag_File)
                break
            except Exception as ex:
                print(ex)
                continue

        #Reset wrapper flags
        wrapperFlags = GetFlags(Wrapper_Flag_File)
        for flag in wrapperFlags:
            wrapperFlags[flag] = "FALSE"
        WriteFlags(wrapperFlags, Wrapper_Flag_File)

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
            wrapperFlags = GetFlags(Wrapper_Flag_File)
            wrapperFlags["No_More_Frames"] = "TRUE"
            WriteFlags(wrapperFlags, Wrapper_Flag_File)
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
                f"Average memory usage is {(sum(current_ram_usage)/len(current_ram_usage)) / 10 ** 6}MB; "
            )
            print(f"Peak was {peak / 10 ** 6}MB")
            tracemalloc.stop()
            break
        else:
            wrapperFlags = GetFlags(Wrapper_Flag_File)
            for flag in wrapperFlags:
                if str(flag).startswith("Frames"):
                    wrapperFlags[flag] = "TRUE"
            WriteFlags(wrapperFlags, Wrapper_Flag_File)
            current, peak = tracemalloc.get_traced_memory()
            current_ram_usage.append(current)
            timeoffset = float(timeoffset) + float(duration)
            timeoffset = round(timeoffset, 2)
            batch_elapsed_time = time.time() - batch_start_time
            batch_times.append(batch_elapsed_time)
