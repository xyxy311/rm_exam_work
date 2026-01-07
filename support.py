import numpy as np
# 统一旋转矩形的w、h、angle
def standardRect(Rect):
    (x, y), (w, h), angle = Rect
    if w > h:
        w, h = h, w
    else:
        angle += 90
    return (x, y), (w, h), angle