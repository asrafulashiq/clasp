import json
import collections
import sys
import math
import numpy as np
import cv2
import csv
from subprocess import getstatusoutput as cmd

# tool version
scoringToolVersion = "4.0"

# Parse optional input args...
if "-v" in sys.argv:
    VERBOSE = True
else:
    VERBOSE = False
if "-ioupax" in sys.argv:
    pax_intersectionOverUnionThreshold = sys.argv[sys.argv.index("-ioupax") +
                                                  1]
else:
    pax_intersectionOverUnionThreshold = .3

if "-ioudvi" in sys.argv:
    dvi_intersectionOverUnionThreshold = sys.argv[sys.argv.index("-ioudvi") +
                                                  1]
else:
    dvi_intersectionOverUnionThreshold = .5

if "-xfr" in sys.argv:
    xfrThreshold = int(sys.argv[sys.argv.index("-xfr") + 1])
else:
    xfrThreshold = 30

if "-c" in sys.argv:
    camera_filter = sys.argv[sys.argv.index("-c") + 1]
else:
    camera_filter = ''

if "-m" in sys.argv:
    camera = []
    movie = [sys.argv[sys.argv.index("-m") + 1]]
    idx = 1
    for entry in sys.argv[sys.argv.index("-m") + 1:]:
        idx = idx + 1
        try:
            movie.append(sys.argv[sys.argv.index("-m") + idx])

            cam = entry[0:entry.index("exp")].replace("cam",
                                                      '').replace("./", '')
            camera.append(cam)
        except:
            break
    i = 0
    for mov in movie:
        if "mp4" not in mov:
            del movie[i:]
        i = i + 1

    if camera_filter != '':
        camera = [camera_filter]

    experiment = movie[0].replace(".mp4", '').replace("cam", '')
    experiment = experiment[experiment.index("exp"):]
else:
    experiment = ''
    movie = []
    camera = []

if "-g" in sys.argv:
    try:
        gt_in = str(sys.argv[sys.argv.index("-g") + 1])
        gt_output = open(gt_in, 'r').read().split('\n')
    except:
        print("Failed to load ground truth file " + str(gt_in) + "...")
        exit(-1)
else:
    try:
        gt_in = "./gt.txt"
        gt_output = open(gt_in, 'r').read().split('\n')
    except:
        print("Failed to load ground truth file file gt.txt...")
        exit(-1)

if "-a" in sys.argv:
    try:
        ata_in = str(sys.argv[sys.argv.index("-a") + 1])
        ata_output = open(ata_in, 'r').read().split('\n')
    except:
        print("Failed to load algorithm file " + str(ata_in) + "...")
        exit(-1)
else:
    try:
        ata_in = "./ata.txt"
        ata_output = open(ata_in, 'r').read().split('\n')
    except:
        print("Failed to load algorithm file ata.txt...")
        exit(-1)

# Global variables...

# where results are tabulated
results = {
    "# PD (PAX)": 0,
    "# PD (DVI)": 0,
    "# PD (XFR)": 0,
    "# PFA (PAX)": 0,
    "# PFA (DVI)": 0,
    "# PFA (XFR)": 0,
    "# PAX total": 0,
    "# DVI total": 0,
    "# XFR total": 0,
    "# pax switch": 0,
    "# dvi switch": 0,
    "# mismatch": 0
}

# reusing ID warning
WARNING = False

# create and cultivate gt-debug and ata-debug
gtDebug = {"pax": {}, "dvi": {}, "xfr": {}}
ataDebug = {"pax": {}, "dvi": {}, "xfr": {}}

# output logs for both ata and gt
GtoutFile = "gt-debug.txt"
AtaoutFile = "ata-debug.txt"

# TODO switch the following two variable names...

# For ID tracking...
FrameObjects = {}

# objects in current frame
obs = {}

# total frames in videos from gt
totalFrames = 0

# Functions


# helps determine if a mismatch has occured
def priorFrameObjects(gtTag, ataTag):
    if ataTag != FrameObjects[gtTag][0]:
        return True
    return False


# returns the current ATA tracking ID
def getFrameObjects(gtTag):
    tag = ''
    end = len(FrameObjects[gtTag]) - 1
    tag = FrameObjects[gtTag][end]
    return tag


# sets a ATA tracking ID
def setFrameObjects(gtTag, ataTag):
    FrameObjects[gtTag].append(ataTag)


# creates an entry in the dict for a new ID
def createFrameObjects(gtTag):
    FrameObjects[gtTag] = []


# NOTE: Adds labels to boxes...
# annotate BB on available frame
def annotateBB(frame, entry, t, iou):
    frame = int(frame) + 1
    f = "cam" + str(entry[entry.index("camera-num") +
                          1]) + experiment + "/" + str(frame).zfill(5) + ".jpg"
    img = cv2.imread(f)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    lineType = 2

    x1 = int(entry[entry.index("bb") + 1])
    x2 = int(entry[entry.index("bb") + 3])
    y1 = int(entry[entry.index("bb") + 2])
    y2 = int(entry[entry.index("bb") + 4])

    if t == "ata":
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        bottomLeftCornerOfText = (x1 + 15, y1 + 35)
        fontColor = (0, 0, 255)
        cv2.putText(img, str(entry[entry.index("id") + 1]),
                    bottomLeftCornerOfText, font, fontScale, fontColor,
                    lineType)

        x1 = int(entry[entry.index("bb") + 1])
        x2 = int(entry[entry.index("bb") + 3])
        y1 = int(entry[entry.index("bb") + 2])
        y2 = int(entry[entry.index("bb") + 4])

        bottomLeftCornerOfText = (x1 + 15, y1 + ((y2 - y1) / 2))
        cv2.putText(img, "{0:.2f}".format(iou), bottomLeftCornerOfText, font,
                    fontScale, fontColor, lineType)
    elif t == "gt":
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 15)
        bottomLeftCornerOfText = (x1 + 15, y2 - 15)
        fontColor = (0, 255, 0)

        try:
            cv2.putText(img,
                        str(getFrameObjects(entry[entry.index("id") + 1])),
                        bottomLeftCornerOfText, font, fontScale, fontColor,
                        lineType)
        except KeyError:
            cv2.putText(img, str(entry[entry.index("id") + 1]),
                        bottomLeftCornerOfText, font, fontScale, fontColor,
                        lineType)
    elif t == "fa":
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 5)
        bottomLeftCornerOfText = (x1 + 15, y1 + 35)
        fontColor = (0, 255, 255)
        cv2.putText(img, str(entry[entry.index("id") + 1]),
                    bottomLeftCornerOfText, font, fontScale, fontColor,
                    lineType)
    else:
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 15)

    cv2.imwrite(f, img)
    return


# annotate text on available frame
def annotateText(frame, entry, t, ataFrame):
    frame = int(frame) + 1
    f = "cam" + str(entry[entry.index("camera-num") +
                          1]) + experiment + "/" + str(frame).zfill(5) + ".jpg"
    img = cv2.imread(f)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    lineType = 2

    if t == "gt":
        bottomLeftCornerOfText = (10, 50)
        fontColor = (0, 255, 0)
        cv2.putText(img, 'GT  XFR EVENT', bottomLeftCornerOfText, font,
                    fontScale, fontColor, lineType)
    elif t == "ata":
        bottomLeftCornerOfText = (10, 100)
        fontColor = (0, 0, 255)

        if int(ataFrame) + 1 - int(frame) < 0:
            cv2.putText(
                img, "ATA XFR EVENT at " +
                str(int(ataFrame) + 1 - int(frame)) + " frames",
                bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        elif int(ataFrame) + 1 - int(frame) > 0:
            cv2.putText(
                img, "ATA XFR EVENT at +" +
                str(int(ataFrame) + 1 - int(frame)) + " frames",
                bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        else:
            cv2.putText(img, "ATA XFR EVENT +/- 0 frames",
                        bottomLeftCornerOfText, font, fontScale, fontColor,
                        lineType)
    elif t == "fa":
        bottomLeftCornerOfText = (10, 100)
        fontColor = (0, 255, 255)

        if int(ataFrame) + 1 - int(frame) < 0:
            cv2.putText(
                img, "ATA XFR EVENT False Alarm at " +
                str(int(ataFrame) + 1 - int(frame)) + " frames",
                bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        elif int(ataFrame) + 1 - int(frame) > 0:
            cv2.putText(
                img, "ATA XFR EVENT False Alarm at +" +
                str(int(ataFrame) + 1 - int(frame)) + " frames",
                bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        else:
            cv2.putText(img, "ATA XFR EVENT False Alarm at +/- 0 frames",
                        bottomLeftCornerOfText, font, fontScale, fontColor,
                        lineType)
    else:
        bottomLeftCornerOfText = (10, 150)
        fontColor = (255, 255, 255)
        cv2.putText(img, ataFrame, bottomLeftCornerOfText, font, fontScale,
                    fontColor, lineType)

    cv2.imwrite(f, img)

    return


# log a string to a file
def log(logFile, string):
    f = open(logFile, 'a')
    f.write(string + "\n")
    f.close()


# call createFrameset.py if you need to output the BB info to the frames for comparison
def createFramesets():
    global movie

    for mov in movie:
        cmd_string = "python ./createFrameset.py " + mov
        print(cmd_string)
        resp = cmd(cmd_string)
        if resp[0] != 0:
            return -1
    return 0


# Formats the string that will act as a header
def formatHeader():
    ret_string = ''

    # HEADER
    try:
        ret_string = ret_string + "# ##### HEADER for " + str(
            experiment) + " #####\n"
    except NameError:
        ret_string = ret_string + "# ##### HEADER #####\n"
    # Date & time
    ret_string = ret_string + "# time: " + str(
        cmd("date +%Y-%m-%d\" \"%H:%M:%S")[1]) + "\n"

    # ata logfile name
    ret_string = ret_string + "# ata logfile: " + str(ata_in) + "\n"
    # gt logfile name
    ret_string = ret_string + "# gt logfile: " + str(gt_in) + "\n"

    # total frames in experiment
    ret_string = ret_string + "# total frames: " + str(totalFrames) + "\n"

    # # of frames w/ events a.k.a. records
    ret_string = ret_string + "# total records: " + str(
        int(results["# PAX total"] + results["# DVI total"] +
            results["# XFR total"])) + "\n"

    # total events by type
    ret_string = ret_string + "# total GT PAX location events: " + str(
        results["# PAX total"]) + "\n"
    ret_string = ret_string + "# total GT DVI location events: " + str(
        results["# DVI total"]) + "\n"
    ret_string = ret_string + "# total GT XFR events: " + str(
        results["# XFR total"]) + "\n"

    # which cameras?
    ret_string = ret_string + "# cameras: " + str(camera).replace(
        '[', '').replace('\'', '').replace(']', '') + "\n"
    # iou threshold
    ret_string = ret_string + "# PAX IoU threshold: " + str(
        pax_intersectionOverUnionThreshold) + "\n"
    ret_string = ret_string + "# DVI IoU threshold: " + str(
        dvi_intersectionOverUnionThreshold) + "\n"

    # xfr thresdhold
    ret_string = ret_string + "# XFR threshold: " + str(xfrThreshold) + "\n"
    # version of scoring tool
    ret_string = ret_string + "# scoring Tool Version: " + str(
        scoringToolVersion) + "\n\n"

    return ret_string


# Create and format results.txt
def formatScoreReport(results, gtDebug, ataDebug):
    scoreCard = open("results.txt", 'w')

    scoreCard.write(formatHeader())

    # RESULTS
    try:
        scoreCard.write("PD (PAX) = %" + "{0:.1f}".format(
            float((float(results["# PD (PAX)"])) / results["# PAX total"]) *
            100) + " (" + str(results["# PD (PAX)"]) + "/" +
                        str(results["# PAX total"]) + ")\n")
    except ZeroDivisionError:
        scoreCard.write("PD (PAX) = N/A\n")
    try:
        scoreCard.write("PD (DVI) = %" + "{0:.1f}".format(
            float((float(results["# PD (DVI)"])) / results["# DVI total"]) *
            100) + " (" + str(results["# PD (DVI)"]) + "/" +
                        str(results["# DVI total"]) + ")\n")
    except ZeroDivisionError:
        scoreCard.write("PD (DVI) = N/A\n")
    try:
        scoreCard.write("PD (XFR) = %" + "{0:.1f}".format(
            float((float(results["# PD (XFR)"])) / results["# XFR total"]) *
            100) + " (" + str(results["# PD (XFR)"]) + "/" +
                        str(results["# XFR total"]) + ")\n")
    except ZeroDivisionError:
        scoreCard.write("PD (XFR) = N/A\n")

    try:
        scoreCard.write("PFA (PAX) = %" + "{0:.1f}".format(
            float((float(results["# PFA (PAX)"])) /
                  round(totalFrames * len(camera) / 100)) * 100) + " (" +
                        str(results["# PFA (PAX)"]) + "/" +
                        str(int(round(totalFrames * len(camera) / 100.0))) +
                        ")\n")
    except ZeroDivisionError:
        scoreCard.write("PFA (PAX) = N/A\n")

    try:
        scoreCard.write("PFA (DVI) = %" + "{0:.1f}".format(
            float((float(results["# PFA (DVI)"])) /
                  round(totalFrames * len(camera) / 100)) * 100) + " (" +
                        str(results["# PFA (DVI)"]) + "/" +
                        str(int(round(totalFrames * len(camera) / 100.0))) +
                        ")\n")
    except ZeroDivisionError:
        scoreCard.write("PFA (DVI) = N/A\n")
    try:
        scoreCard.write("PFA (XFR) = %" + "{0:.1f}".format(
            float((float(results["# PFA (XFR)"])) /
                  round(totalFrames * len(camera))) * 100) + " (" +
                        str(results["# PFA (XFR)"]) + "/" +
                        str(int(round(totalFrames * len(camera)))) + ")\n")
    except ZeroDivisionError:
        scoreCard.write("PFA (XFR) = N/A\n")

    try:
        scoreCard.write("P (PAX switch) = %" + "{0:.1f}".format(
            float(
                float(results["# pax switch"]) /
                float(results["# PAX total"])) * 100) + " (" +
                        str(results["# pax switch"]) + "/" +
                        str(results["# PAX total"]) + ")\n")
    except ZeroDivisionError:
        scoreCard.write("P (PAX switch) = N/A\n")
    try:
        scoreCard.write("P (DVI switch) = %" + "{0:.1f}".format(
            float(
                float(results["# dvi switch"]) /
                float(results["# DVI total"])) * 100) + " (" +
                        str(results["# dvi switch"]) + "/" +
                        str(results["# DVI total"]) + ")\n")
    except ZeroDivisionError:
        scoreCard.write("P (DVI switch) = N/A\n")

    try:
        scoreCard.write("P (mismatch) = %" + "{0:.1f}".format(
            float(
                float(results["# mismatch"]) / float(results["# XFR total"])) *
            100) + " (" + str(results["# mismatch"]) + "/" +
                        str(results["# XFR total"]) + ")\n")
    except ZeroDivisionError:
        scoreCard.write("P (mismatch) = N/A\n")

    scoreCard.close()
    return


# Import data from input files into python data structures
def formatData(data):
    global camera
    global experiment
    global totalFrames

    formattedData = {}

    for d in data:
        entry = d.lower().replace(":", '').replace("\r",
                                                   '').replace(',',
                                                               '').split(" ")
        if entry[0] == "#" and entry[1] == "total-frames":
            totalFrames = int(entry[2])

        if entry == [''] or entry[0] == "#" or entry[0] == "########":
            if experiment == '' and entry[0] == "########":
                experiment = entry[1]
            continue
        frame = entry[entry.index("frame") + 1].zfill(5)
        cam = entry[entry.index("camera-num") + 1]

        if cam not in camera:
            camera.append(cam)

        try:
            formattedData[frame]
            formattedData[frame].append(entry)
        except KeyError:
            formattedData[frame] = ['']
            formattedData[frame][0] = entry

    if camera_filter != '':
        camera = [camera_filter]

    return formattedData


# tests if its needed to stop and test the xfr event and associated dvi loc event
def stopAndTest_xfr(gt, ata, idx):
    try:
        ata[str(idx).zfill(5)]
        if "xfr" in str(ata[str(idx).zfill(5)]):
            return True
    except KeyError:
        pass

    try:
        gt[str(idx).zfill(5)]
        if "xfr" in str(gt[str(idx).zfill(5)]):
            return True
    except KeyError:
        pass

    return False


# Stops every 100th frame and test all events
def stopAndTest_loc(gt, ata, idx):
    if idx % 100 == 0:
        return True
    return False


# Calculate the area of a rectangle
def area(rectangle):
    area = 0.0
    area = float(rectangle["width"]) * float(rectangle["height"])

    return float(area)


# intersection over union
def IoU(r1, r2):
    iou = 0.0

    # initialize variables
    rect1 = {
        "x": float(r1[0]),
        "width": float(r1[2]) - float(r1[0]),
        "y": float(r1[1]),
        "height": float(r1[3]) - float(r1[1])
    }
    rect2 = {
        "x": float(r2[0]),
        "width": float(r2[2]) - float(r2[0]),
        "y": float(r2[1]),
        "height": float(r2[3]) - float(r2[1])
    }

    # Calculate area of intersection
    x_overlap = max(
        0,
        min(rect1["x"] + rect1["width"], rect2["x"] + rect2["width"]) -
        max(rect1["x"], rect2["x"]))
    y_overlap = max(
        0,
        min(rect1["y"] + rect1["height"], rect2["y"] + rect2["height"]) -
        max(rect1["y"], rect2["y"]))
    intersection_area = float(x_overlap) * float(y_overlap)

    # Calculate area of union
    union_area = abs(area(rect1)) + abs(area(rect2)) - intersection_area
    '''
	print str(rect1)
	print str(area(rect1))
	print
	print str(rect2)
	print str(area(rect2))
	print
	print str(intersection_area)
	print str(union_area)
	'''

    iou = float(intersection_area) / float(union_area)
    return iou


# find the match for a loc event
def match_loc_event(ata, event, frame, t):
    global gtDebug
    global ataDebug
    global WARNING
    global obs

    try:
        ata[frame]
    except KeyError:
        if t == "pax":
            gtDebug["pax"][frame][str(event)] = "not detected..."
        elif t == "dvi":
            gtDebug["dvi"][frame][str(event)] = "not detected..."

        annotateBB(frame, event, "gt", '')
        return False

    for entry in ata[frame]:
        # Events from other cameras are not in play
        if entry[entry.index("camera-num") +
                 1] != event[event.index("camera-num") + 1]:
            continue

        iou = IoU(entry[entry.index("bb") + 1:entry.index("bb") + 5],
                  event[event.index("bb") + 1:event.index("bb") + 5])

        if float(iou) > float(dvi_intersectionOverUnionThreshold) and entry[entry.index("type") + 1] == "dvi" or \
         float(iou) > float(pax_intersectionOverUnionThreshold) and entry[entry.index("type") + 1] == "pax":

            if VERBOSE:
                print("\nGT:  " + str(event))
                print("...with " + str(iou) + " ...")
                print("ATA: " + str(entry) + "\n")

            if t == "pax":
                gtDebug["pax"][frame][str(event)] = entry
                ataDebug["pax"][frame][str(entry)] = event
            elif t == "dvi":
                gtDebug["dvi"][frame][str(event)] = entry
                ataDebug["dvi"][frame][str(entry)] = event

            try:
                getFrameObjects(event[event.index("id") + 1])
                if getFrameObjects(event[event.index("id") +
                                         1]) != entry[entry.index("id") + 1]:
                    setFrameObjects(event[event.index("id") + 1],
                                    entry[entry.index("id") + 1])
                    print("Switch occured!!!")

                    if event[event.index("type") + 1] == "dvi":
                        results["# dvi switch"] = results["# dvi switch"] + 1
                    if event[event.index("type") + 1] == "pax":
                        results["# pax switch"] = results["# pax switch"] + 1
            except KeyError:
                createFrameObjects(event[event.index("id") + 1])
                setFrameObjects(event[event.index("id") + 1],
                                entry[entry.index("id") + 1])

            annotateBB(frame, event, "gt", '')
            annotateBB(frame, entry, "ata", iou)

            # NOTE Warns the user if the algorithm is assigning multiple objects the same ID within a frame
            # NOTE when the tool matches multiple log entries in the ata to a single gt location event
            # then the warning will be displayed.
            if entry[entry.index("id") + 1] in list(obs.keys()):
                if obs[entry[entry.index("id") + 1]]["entry"][3] == entry[
                        entry.index("type") +
                        1] and obs[entry[entry.index("id") +
                                         1]]["camera-num"] == entry[
                                             entry.index("camera-num") + 1]:
                    #if obs[entry[entry.index("id") + 1]]["entry"] != entry:
                    print("reused ID at frame " + str(frame) + ": " +
                          str(entry[entry.index("id") + 1]))
                    WARNING = True
            obs[entry[entry.index("id") + 1]] = {
                "camera-num": entry[entry.index("camera-num") + 1],
                "entry": entry
            }

            # raw_input(obs)

            return True

    if t == "pax":
        gtDebug["pax"][frame][str(event)] = "not detected..."
    elif t == "dvi":
        gtDebug["dvi"][frame][str(event)] = "not detected..."

    annotateBB(frame, event, "gt", '')
    return False


# Find a match for a xfr event
def findATAXFR(ata, tag, cam, frame):
    ata_frame = frame
    xfr_idx = -1
    mismatch = False

    low = int(frame) - 30
    high = int(frame) + 30
    idx = [0]
    for i in range(0, xfrThreshold):
        idx.append(i + 1)
        idx.append(-1 * (i + 1))

    i = 0
    for iteration in range(len(idx)):
        xfr_idx = 0
        f = str(int(frame) + idx[i]).zfill(5)

        try:
            ata[f]
        except:
            i = i + 1
            continue

        for item in ata[f]:
            used = False
            try:
                if ((str(item) in list(ataDebug["xfr"][f].keys())) and
                    (ataDebug["xfr"][f][str(item)] != "false alarm...")
                    ) or str(item) in list(gtDebug["xfr"][frame].values()):
                    used = True
            except KeyError:
                pass

            if used == False and item[0] == "xfr" and item[
                    item.index("camera-num") + 1] == cam:
                if item[item.index("type") +
                        1] == "from" and priorFrameObjects(
                            tag, item[item.index("dvi-id") + 1]
                        ):  # and item[item.index("dvi-id") + 1] != tag:
                    print("mismatch event detected...")
                    results["# mismatch"] = results["# mismatch"] + 1
                    mismatch = True
                if VERBOSE:
                    print("xfr event found at frame " + str(f))
                ata_frame = f
                return ata_frame, xfr_idx, mismatch
            xfr_idx = xfr_idx + 1
        i = i + 1
    return ata_frame, -1, mismatch


# XFR question graded against XFR criteria in CLASP SPEC
def match_xfr_event(gt, ata, gtFrame, event):
    xfr_idx = -1

    try:
        if str(event) in list(ataDebug["xfr"][f].keys(
        )) and ataDebug["xfr"][f][str(event)] != "false alarm...":
            used = True
    except KeyError:
        pass

    frame, xfr_idx, mismatch = findATAXFR(ata,
                                          event[event.index("dvi-id") + 1],
                                          event[event.index("camera-num") + 1],
                                          gtFrame)

    try:
        ataDebug["xfr"][frame][str(ata[frame][xfr_idx])] = str(event)
    except KeyError:
        ataDebug["xfr"][frame] = {}
        ataDebug["xfr"][frame][str(ata[frame][xfr_idx])] = str(event)
    if xfr_idx == -1:
        print("\nGT:  " + str(event))
        print("xfr event could not be located...\n")

        gtDebug["xfr"][gtFrame][str(event)] = "not detected..."
        return False

    if VERBOSE:
        print("\nGT:  " + str(event))
        print("...with...")
        print("ATA: " + str(ata[frame][xfr_idx]) + "\n")

    annotateText(gtFrame, event, "ata", frame)

    gtDebug["xfr"][gtFrame][str(event)] = str(ata[frame][xfr_idx])

    return True


# When a xfr event is detected on non-100th frames
def xfr_prompt(gt, ata, idx):
    try:
        gt[idx]

        for entry in gt[idx]:
            found = False
            if camera_filter != '':
                if camera_filter != entry[entry.index("camera-num") + 1]:
                    continue

            if entry[0] == "xfr":
                print("\n~ frame " + str(idx) + " ~")
                print("matching... " + str(entry[0]))
                results["# XFR total"] = results["# XFR total"] + 1
                annotateText(idx, entry, "gt", '')
                if match_xfr_event(gt, ata, idx, entry):
                    # annotateText(idx, entry, "gt", '')
                    results["# PD (XFR)"] = results["# PD (XFR)"] + 1
                    print("           [ PASSED ]")
                else:
                    print("           [ FAILED ]")
                print()
                # match the corresponding dvi-id loc events if in gt
                dviTag = entry[entry.index("dvi-id") + 1]
                for event in gt[idx]:
                    if event[0] == "xfr":
                        continue
                    if event[event.index("id") + 1] == dviTag:
                        print("\n~ frame " + str(idx) + " ~")
                        print("matching... " + str(event[0]))
                        results["# DVI total"] = results["# DVI total"] + 1
                        if match_loc_event(ata, event, idx,
                                           event[event.index("type") + 1]):
                            results["# PD (DVI)"] = results["# PD (DVI)"] + 1
                            print("           [ PASSED ]")
                        else:
                            print("           [ FAILED ]")
                        found = True
                    if found == True:
                        break
            print()
    # If there is no ground truth double check what the ata is saying
    except KeyError:
        pass

    try:
        ata[idx]
        for entry in ata[idx]:
            if camera_filter != '':
                if camera_filter != entry[entry.index("camera-num") + 1]:
                    continue

            if entry[0] == "xfr":
                try:
                    if str(entry) in list(ataDebug["xfr"][idx].keys()):
                        continue
                except KeyError:
                    pass
                ataDebug["xfr"][idx][str(entry)] = "false alarm..."
    except KeyError:
        pass
    return


# Test everything within the frame
def prompt(gt, ata, idx):
    try:
        gt[idx]

        for entry in gt[idx]:
            if camera_filter != '':
                if camera_filter != entry[entry.index("camera-num") + 1]:
                    continue

            print("\n~ frame " + str(idx) + " ~")
            print("matching... " + str(entry[0]))
            if entry[0] == "loc":
                if entry[entry.index("type") + 1] == "dvi":
                    results["# DVI total"] = results["# DVI total"] + 1
                elif entry[entry.index("type") + 1] == "pax":
                    results["# PAX total"] = results["# PAX total"] + 1
                if match_loc_event(ata, entry, idx,
                                   entry[entry.index("type") + 1]):
                    if entry[entry.index("type") + 1] == "dvi":
                        results["# PD (DVI)"] = results["# PD (DVI)"] + 1
                    elif entry[entry.index("type") + 1] == "pax":
                        results["# PD (PAX)"] = results["# PD (PAX)"] + 1
                    print("           [ PASSED ]")
                else:
                    print("           [ FAILED ]")
            elif entry[0] == "xfr":
                results["# XFR total"] = results["# XFR total"] + 1
                if match_xfr_event(gt, ata, idx, entry):
                    annotateText(idx, entry, "gt", '')
                    results["# PD (XFR)"] = results["# PD (XFR)"] + 1
                    print("           [ PASSED ]")
                else:
                    print("           [ FAILED ]")
            print()
    # If there is no ground truth double check what the ata is saying
    except KeyError:
        pass

    try:
        ata[idx]
        for entry in ata[idx]:
            if camera_filter != '':
                if camera_filter != entry[entry.index("camera-num") + 1]:
                    continue

            if entry[0] == "loc":
                try:
                    if str(entry) in list(
                            ataDebug["pax"][idx].keys()) or str(entry) in list(
                                gtDebug["pax"][idx].values()):
                        continue
                except:
                    pass
                try:
                    if str(entry) in list(
                            ataDebug["dvi"][idx].keys()) or str(entry) in list(
                                gtDebug["dvi"][idx].values()):
                        continue
                except:
                    pass

                if entry[entry.index("type") + 1] == "pax":
                    ataDebug["pax"][idx][str(entry)] = "false alarm..."
                    results["# PFA (PAX)"] = results["# PFA (PAX)"] + 1

                elif entry[entry.index("type") + 1] == "dvi":
                    ataDebug["dvi"][idx][str(entry)] = "false alarm..."
                    results["# PFA (DVI)"] = results["# PFA (DVI)"] + 1

                annotateBB(idx, entry, "fa", '')

            elif entry[0] == "xfr":
                try:
                    if str(entry) in list(ataDebug["xfr"][idx].keys(
                    )):  # or str(entry) in gtDebug["xfr"][idx].values():
                        continue
                except KeyError:
                    pass
                ataDebug["xfr"][idx][str(entry)] = "false alarm..."
    except KeyError:
        pass
    return


######### MAIN #########
# clear logfiles
WARNING = False

f = open(AtaoutFile, "w")
f.close()
f = open(GtoutFile, "w")
f.close()

# Format the ata and gt data
gt = formatData(gt_output)
ata = formatData(ata_output)

# how many frames are in the video dataset
frame_idx = len(sorted(ata.keys()))

# Create framesets from mp4
if movie:
    print("creating frameset(s)...")
    if createFramesets() != 0:
        print("failed to create frameset...")
        exit(-1)

# Iterate over all frames
for i in range(int(totalFrames) + 1):
    obs = {}

    # When dealing with early ata xfr events its possible the x[t][i] has already been created
    for t in ["pax", "dvi", "xfr"]:
        try:
            gtDebug[t][str(i).zfill(5)]
        except KeyError:
            gtDebug[t][str(i).zfill(5)] = {}
        try:
            ataDebug[t][str(i).zfill(5)]
        except KeyError:
            ataDebug[t][str(i).zfill(5)] = {}

    if stopAndTest_loc(gt, ata, i):
        prompt(gt, ata, str(i).zfill(5))
    elif stopAndTest_xfr(gt, ata, i):
        xfr_prompt(gt, ata, str(i).zfill(5))
    else:
        # Need to mark the extra ata frames as unmatched (NA)
        try:
            ata[str(i).zfill(5)]
            for item in ata[str(i).zfill(5)]:
                if item[0] == "loc" and item[2] == "pax":
                    ataDebug["pax"][str(i).zfill(5)][str(
                        item)] = "not considered..."
                elif item[0] == "loc" and item[2] == "dvi":
                    ataDebug["dvi"][str(i).zfill(5)][str(
                        item)] = "not considered..."
                elif item[0] == "xfr":
                    ataDebug["xfr"][str(i).zfill(5)][str(
                        item)] = "not considered..."
        except KeyError:
            pass

# start headers
log(AtaoutFile, formatHeader())
log(GtoutFile, formatHeader())

# TODO csv file for statistical analysis
# camera-num, frame, type, IoU, status (0 = match, 1 = false alarm, 2=miss), BB_x1, BB_y1, BB_x2, BB_y2
# Create csv file for statistical analysis
# ofile = open("test.csv", 'w')
# writer = csv.writer(ofile, delimiter=',')

# Create/fill log files
for i in range(int(totalFrames) + 1):
    for t in ["pax", "dvi", "xfr"]:
        try:
            gtDebug[t][str(i).zfill(5)]
            for item in gtDebug[t][str(i).zfill(5)]:
                log(GtoutFile, str(i).zfill(5) + ":")
                log(GtoutFile, "GT:  " + str(item))
                log(GtoutFile,
                    "ATA: " + str(gtDebug[t][str(i).zfill(5)][item]) + "\n")
        except KeyError:
            pass
        try:
            ataDebug[t][str(i).zfill(5)]
            for item in ataDebug[t][str(i).zfill(5)]:
                log(AtaoutFile, str(i).zfill(5) + ":")
                log(AtaoutFile, "ATA: " + str(item))
                log(AtaoutFile,
                    "GT:  " + str(ataDebug[t][str(i).zfill(5)][item]) + "\n")
        except KeyError:
            pass

# ofile.close()

# calculate # pfa xfr now thatmatching is complete
for f in ataDebug["xfr"]:
    for item in ataDebug["xfr"][f]:
        if ataDebug["xfr"][f][item] == "false alarm...":
            af = (int(math.ceil(int(f) / 10.0)) * 10)
            arr_item = item.replace("[", '').replace("]", '').replace(
                ",", '').replace("\'", '').split(" ")
            annotateText(str(af).zfill(5), arr_item, "fa", f)
            results["# PFA (XFR)"] = results["# PFA (XFR)"] + 1

# NOTE: Warning for when id's are reused
if WARNING:
    print(
        "WARNING! WARNING! WARNING! IDs were reused within the same frame by the algorithm!!!"
    )

# Create the output file
formatScoreReport(results, gtDebug, ataDebug)

# format all output files to work in dos
cmd("./unix2dos ./*.txt")
