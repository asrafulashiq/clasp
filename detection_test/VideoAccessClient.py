import requests
import json
import zipfile
import io
import os
import yaml
import time

def GetFlags():
    flags = {}
    with open("/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/FrameSyncFlags.yaml", 'r') as stream:
        flags = yaml.safe_load(stream)
    return flags

def WriteFlags(flags):
    with open("/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/FrameSyncFlags.yaml", 'w') as file:
        yaml.dump(flags, file)

def CleanInputDir(inputdir):
    framelist = [f for f in os.listdir(inputdir) if f.endswith(".jpg")]
    for f in framelist:
        os.remove(os.path.join(inputdir, f))
    return

myurl = 'http://127.0.0.1:5000/frames'

#Prepare all the inputs
#Examples
#dataset = 'exp2training'
dataset = str(input("Enter the dataset:"))
cameralist = '2,5,9,11,13,14'
timeoffset = '0.00'
duration = 1.20
duration = "{0:.2f}".format(float(duration))

inputdir = "/data/ALERT-SHARE/alert-api-wrapper-data/"
CleanInputDir(inputdir)
flags = GetFlags()
for flag in flags:
    if flag == "Batch_Processed":
        flags[flag] = "TRUE"
    else:
        flags[flag] = "FALSE"
WriteFlags(flags)
total_start_time = time.time()
while True:
    flags = GetFlags()
    if flags["Batch_Processed"] == "TRUE":
        flags["Batch_Processed"] == "FALSE"
        WriteFlags(flags)
        batch_start_time = time.time()
        inputParams = {"dataset": dataset, "cameralist": cameralist,"timeoffset": timeoffset, "duration": duration, "filesize": "1920x1080","inputdir": inputdir}
        jsoninputParams = json.dumps(inputParams)
        jsonParams = {"APIParams":jsoninputParams}
        response = requests.get(url = myurl, params = jsonParams)
        if response.text == 'No More Frames':
            print('Reached end of video!')
            flags = GetFlags()
            flags["No_More_Frames"] = "TRUE"
            WriteFlags(flags)
            total_elapsed_time = time.time() - total_start_time
            print("Total Processing Time:" + str(total_elapsed_time))
            break
        else:
            #zf = zipfile.ZipFile(io.BytesIO(response.content))
            #zf.extractall("/data/ALERT-SHARE/alert-api-wrapper-data")
            flags = GetFlags()
            for flag in flags:
                if str(flag).startswith("Frames"):
                    flags[flag] = "TRUE"
            WriteFlags(flags)
            timeoffset = float(timeoffset) + float(duration)
            timeoffset = round(timeoffset,2)
            batch_end_time = time.time() - batch_start_time
            print("Batch Processing Time:" + str(batch_end_time))
            #print(round(timeoffset,2))

