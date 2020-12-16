from subprocess import getstatusoutput as cmd
import sys

for movie in sys.argv[1:]:
    frameset = movie.replace("./", '')
    cmd_string = "mkdir ./" + frameset.replace(".mp4", '')
    cmd(cmd_string)

    cmd_string = "ffmpeg -i " + frameset + " -vf \"select=not(mod(n\,10))\" -vsync vfr -q:v 2 ./" + frameset.replace(
        ".mp4", '') + "/%05d.jpg"
    print(cmd_string)
    resp = cmd(cmd_string)
    if resp[0] != 0:
        print("... [ FAIL ]\n")
        continue
    else:
        print("... [ OK ]\n")

    for entry in cmd("ls ./" + frameset.replace(".mp4", '') +
                     "/")[1].split("\n"):
        new_name = '%05d' % int((int(entry.replace(".jpg", '')) * 10) - 9)
        cmd_string = 'mv ./' + frameset.replace(
            ".mp4", '') + "/" + entry + " ./" + frameset.replace(
                ".mp4", '') + "/_" + new_name + ".jpg"
        resp = cmd(cmd_string)
        # print cmd_string + ": " + str(resp[0])

    for entry in cmd("ls ./" + frameset.replace(".mp4", '') +
                     "/")[1].split("\n"):
        new_name = entry.replace("_", '')
        cmd_string = 'mv ./' + frameset.replace(
            ".mp4", '') + "/" + entry + " ./" + frameset.replace(
                ".mp4", '') + "/" + new_name
        resp = cmd(cmd_string)

    print()
