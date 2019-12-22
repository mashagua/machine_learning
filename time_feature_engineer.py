# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:02:31 2019

@author: masha
"""

from PIL import Image, ImageDraw, ImageFont
import math
sz = 128


def getxy(hour):
    x = math.sin((180 - hour * 15) / 180 * 3.1415926)
    y = math.cos((180 - hour * 15) / 180 * 3.1415926)
    return x, y


def from_center(ratio):
    return lambda z: sz + sz * ratio * z


place34 = from_center(3 / 4)
place78 = from_center(7 / 8)


def rad_to_deg(x): return 180 * x / 3.1415926


def draw_circle(*points):
    def draw_time_point(x1, y1, **kwargs):

        draw.ellipse((place34(x1) - 5,
                      place34(y1) - 5,
                      place34(x1) + 5,
                      place34(y1) + 5),
                     **kwargs)

    im = Image.new('RGB', (2 * sz, 2 * sz))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0) + im.size, fill=(256, 256, 256))
    draw.ellipse((sz / 4,
                  sz / 4,
                  sz * 7 / 4,
                  sz * 7 / 4), outline=0)

    for i in range(24):
        x, y = getxy(i)
        draw.line((sz,
                   sz,
                   place34(x),
                   place34(y)),
                  fill=0)

        draw.text((place78(x) - 5 - 2 * y,  # slight rotation to align numbers
                   place78(y) - 5 + 2 * x),
                  str(i),
                  fill=0)

    if len(points):
        xx, yy = list(zip(*[getxy(p) for p in points]))
        for x1, y1 in zip(xx, yy):
            draw_time_point(x1, y1, outline=(200, 5, 5), fill=(250, 9, 9))

        xm = sum(xx) / len(xx)
        ym = sum(yy) / len(yy)
        draw_time_point(xm, ym, outline=(0, 215, 5), fill=(100, 9, 139))

        r = math.sqrt(xm ** 2 + ym ** 2)
        avg = math.atan2(ym, xm)
        arc = math.acos(r)
        print(r, avg, arc)

        draw.chord(xy=(sz / 4,
                       sz / 4,
                       sz * 7 / 4,
                       sz * 7 / 4),
                   start=rad_to_deg(avg - arc),
                   end=rad_to_deg(avg + arc),
                   outline=(0, 200, 0))

    return im

print(draw_circle(10, 11, 12))