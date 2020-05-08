import cv2
from pathlib import Path
import numpy as np
from collections import deque


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

class ImModule():
    def __init__(self, width, height, pre_text="Camera 9 : ", padding=0.05,):
        self.width = width
        self.height = height
        self.padding = padding
        self.pre_text = pre_text

        e_width = self.width * (1 - 2 * padding)
        e_height = self.height * (1 - 2 * padding)

        # image part
        im_height = e_height * 0.9
        im_width = e_width

        x1 = width * padding
        x2 = x1 + im_width

        y1 = height * padding
        y2 = y1 + im_height

        self.im_bound = [int(i) for i in (x1, y1, x2, y2)]

        # text part
        sep_pad = e_height * 0.02
        t_height = e_height * 0.08

        y1_ = y2 + sep_pad
        y2_ = y1_ + t_height

        self.text_bound = [int(i) for i in (x1, y1_, x2, y2_)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.fontThick = 2

    def draw(self, im, text=""):
        canvas = np.full((self.height, self.width, 3), fill_value=255, dtype=np.uint8)

        x1, y1, x2, y2 = self.im_bound
        w = x2 - x1
        h = y2 - y1
        im = np.rot90(im)
        imresized = cv2.resize(im, (w, h))
        canvas[y1:y1+h, x1:x1+w] = imresized

        text = self.pre_text + text
        x1, y1, x2, y2 = self.text_bound

        ((txt_w, txt_h), _) = cv2.getTextSize(
            text, self.font, self.fontScale, self.fontThick)

        y2 = y2 - int(0.5 * txt_h)
        cv2.putText(canvas, text, (x1, y2), self.font,
                    self.fontScale, _BLACK, self.fontThick,
                    lineType=cv2.LINE_AA)

        return canvas


class FeedModule:
    def __init__(self, width, height, padding=0.03, max_msg=10):
        self.width = width
        self.height = height
        self.padding = padding
        self.formatter = "{:<5}{:^8}{:<40}"
        self.header = "{:<5}{:<8}{:<40}".format("Cam", "Time", "Message")

        self.msg_q = deque([], maxlen=max_msg)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.75
        self.fontThick = 1

        e_width = self.width * (1 - 2 * padding)
        e_height = self.height * (1 - 2 * padding)

        header_height = e_height * 0.05

        x1 = width * padding
        x2 = x1 + e_width

        y1 = height * padding
        y2 = y1 + header_height

        self.header_bound = [int(i) for i in (x1, y1, x2, y2)]

        y0 = y2
        yend = self.height * (1 - padding)

        self.msg_bounds = []

        msg_height = yend - y0

        per_height = msg_height / max_msg

        for i in range(max_msg):
            y1 = y0 + i * per_height
            y2 = y0 + (i + 1) * per_height
            self.msg_bounds.append(
                [int(i) for i in (x1, y1, x2, y2)]
            )

    def drawText(self, msg_list):
        for msg in msg_list:
            text = self.formatter.format(*msg)
            self.msg_q.append(text)

        canvas = np.full((self.height, self.width, 3),
                         fill_value=255, dtype=np.uint8)

        ((_, txt_h), _) = cv2.getTextSize(
            self.header, self.font, self.fontScale, self.fontThick)

        # draw header
        x1, y1, x2, y2 = self.header_bound
        y2 = y2 - int(0.5 * txt_h)

        cv2.putText(canvas, self.header, (x1, y2), self.font,
                    self.fontScale, _BLACK, 2 * self.fontThick,
                    lineType=cv2.LINE_AA)

        msg_rev = list(reversed(self.msg_q))
        for i, msg in enumerate(msg_rev):
            x1, y1, x2, y2 = self.msg_bounds[i]
            y2 = y2 - int(0.5 * txt_h)

            cv2.putText(canvas, msg, (x1, y2), self.font,
                        self.fontScale*0.75, _BLACK, self.fontThick,
                        lineType=cv2.LINE_AA)

        return canvas



class VisFeed():
    def __init__(self, width=1600, height=900, padding=0.02, max_msg=10):
        self.width = width
        self.height = height
        self.padding = padding  # in percentage

        m1, m2, m3 = 0.3, 0.3, 0.4  # module percentage
        # mod 1
        w_, h_ = int(self.width * m1), int(self.height)
        self.im_mod_1 = ImModule(w_, h_, pre_text="Camera 9 : ", padding=padding)

        # mod 2
        self.im_mod_2 = ImModule(w_, h_, pre_text="Camera 11 : ", padding=padding)

        # mod3
        self.im_mod_3 = ImModule(w_, h_, pre_text="Camera 13 : ", padding=padding)

        # mod 4
        w_, h_ = int(self.width * m3), int(self.height)
        self.im_mod_4 = FeedModule(w_, h_, max_msg=max_msg)

    def draw(self, im1, im2, im3, frame_num, msglist, with_feed=True):
        can1 = self.im_mod_1.draw(im1, text=str(to_sec(frame_num)).zfill(4)+' sec')
        can2 = self.im_mod_2.draw(im2, text=str(to_sec(frame_num)).zfill(4)+' sec')
        can3 = self.im_mod_3.draw(im3, text=str(to_sec(frame_num)).zfill(4)+' sec')
        can4 = self.im_mod_4.drawText(msglist)

        if with_feed:
            canvas = np.concatenate((can1, can2, can3, can4), axis=1)
        else:
            canvas = np.concatenate((can1, can2, can3), axis=1)

        return canvas


def to_sec(frame, fps=30):
    return int(frame) // fps
