import numpy as np
from math import sin, radians, cos

# 统一旋转矩形的w、h、angle
def standardRect(rect):
    (x, y), (w, h), angle = rect
    if w > h:
        w, h = h, w
    else:
        angle += 90
    return (x, y), (w, h), angle

# 延展线段
def expandLine(pt1, pt2, ratio):
    x1, y1 = pt1
    x2, y2 = pt2
    dx = abs(x1 - x2) / 2 * (ratio - 1)
    dy = abs(y1 - y2) / 2 * (ratio - 1)
    if x1 > x2:
        x1 += dx
        x2 -= dx
    else:
        x1 -= dx
        x2 += dx
    if y1 > y2:
        y1 += dy
        y2 -= dy
    else:
        y1 -= dy
        y2 += dy
    return (int(x1), int(y1)), (int(x2), int(y2))