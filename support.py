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

# 将统一过的矩形（灯条）转化为线，支持传入角度参数
def rectToline(rect, angle=None):
    if angle is None:
        (x, y), (_, h), angle = rect
    else:
        (x, y), (_, h), _ = rect
    rad = radians(angle)
    sin_a = sin(rad)
    cos_a = cos(rad)
    x1 = int(x - h / 2 * cos_a)
    x2 = int(x + h / 2 * cos_a)
    y1 = int(y - h / 2 * sin_a)
    y2 = int(y + h / 2 * sin_a)

    return (x1, y1), (x2, y2)