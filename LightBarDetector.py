import cv2
import numpy as np
import support as sp
from math import sin, radians, cos

# 灯条对象
class Light:
    def __init__(self, pos=None, angle=None, length=None):

        # 位置、角度、长
        self.pos = pos
        self.angle = angle
        self.length = length

    # 计算端点坐标
    def getEndPoints(self):
        x, y = self.pos
        angle = self.angle
        l = self.length
        rad = radians(angle)
        sin_a = sin(rad)
        cos_a = cos(rad)
        x1 = int(x - l / 2 * cos_a)
        x2 = int(x + l / 2 * cos_a)
        y1 = int(y - l / 2 * sin_a)
        y2 = int(y + l / 2 * sin_a)

        return (x1, y1), (x2, y2)
    
    # 绘制灯条
    def drawLight(self, frame, info=False):
        (x1, y1), (x2, y2) = self.getEndPoints()
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 3, (0, 255, 0), -1)
        if info:
            cv2.putText(frame, str(self.angle), (x1, y1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 识别灯条
class LightBarDetector:
    def __init__(self, enemy_color="red", area_thresh=25, 
                 aspect_ratio_range=(1.5, 15), angle_range=(60, 120)):
        self.enemy_color = enemy_color  # 配置敌方颜色
        self.area_thresh = area_thresh  # 灯条最小面积
        self.aspect_ratio_range = aspect_ratio_range  # 高宽比范围（令h > w）
        self.angle_range = angle_range  # 允许的倾斜角度
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

    # 对颜色通道图二值化，同时进行形态学操作
    def preprocess(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if self.enemy_color == "red":
            # 红方则提取红色区域
            channel = frame[:, :, 2]
            channel = clahe.apply(channel)
            mask1 = cv2.inRange(hsv, (0,29,193), (30, 255, 255))
            mask2 = cv2.inRange(hsv, (170,29,193), (180, 170, 255))
            mask = cv2.bitwise_or(mask1, mask2)

        else:  # blue
            channel = frame[:, :, 0]
            channel = clahe.apply(channel)
            mask = cv2.inRange(hsv, (100, 29, 193), (120, 170, 255))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)  # 给一点膨胀

        # 提取掩码部分
        channel = clahe.apply(channel)
        channel = cv2.bitwise_and(channel, channel, mask=mask)

        # 对每个区域单独二值化，减少光晕的影响
        num_labels, labels = cv2.connectedComponents(channel, connectivity=8)
        for i in range(1, num_labels):
            zone = channel[labels == i]
            max_value = np.max(zone)
            # mean_value = int(np.mean(zone[zone>0]))
            # if mean_value < 180:
            #     zone[zone <= 255 - (180 - mean_value)] += 180 - mean_value
            #     channel[labels == i] = zone     
            thresh = max_value / 255 * 240 #(mean_value + max_value) / 2
            channel[labels == i] = \
                np.where(channel[labels == i] > thresh, max_value, 0)

            # print(mean_value)
            # cv2.waitKey(0)


        # 整体二值化
        _, binary = cv2.threshold(channel, 160, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('binary', binary)

        return binary

    # 检测灯条
    def detect(self, binary):

        lights = []
        # 提取轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
        for cnt in contours:

            # 计算轮廓面积，过滤过小目标
            area = cv2.contourArea(cnt)
            if area < self.area_thresh:
                continue
            
            # 生成旋转矩形
            min_rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = sp.standardRect(min_rect)
            aspect_ratio = h / w
            
            # 筛选：长宽比+角度符合要求
            if (
                self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]
                and
                self.angle_range[0] <= angle <= self.angle_range[1]
                ):

                light = Light((x, y), angle, h)     
                lights.append(light)  
        return lights
    
    # 运行流程
    def run(self, frame):
        binary = self.preprocess(frame)
        lights = self.detect(binary)
        return lights