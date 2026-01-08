import cv2
import numpy as np
import support as sp

# 识别灯条
class LightBarDetector:
    def __init__(self, enemy_color="blue", area_thresh=50, 
                 aspect_ratio_range=(2, 15), angle_range=(60, 120)):
        self.enemy_color = enemy_color  # 配置敌方颜色
        self.area_thresh = area_thresh  # 灯条最小面积
        self.aspect_ratio_range = aspect_ratio_range  # 高宽比范围（令h > w）
        self.angle_range = angle_range  # 允许的倾斜角度

    # 对颜色通道图二值化，同时进行形态学操作
    def preprocess(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if self.enemy_color == "red":
            # 红方则提取红色区域
            channel = frame[:, :, 2]
            mask1 = cv2.inRange(hsv, (0,29,193), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170,29,193), (180, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)

        else:  # blue
            channel = frame[:, :, 0]
            mask = cv2.inRange(hsv, (100, 29, 193), (120, 255, 255))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)  # 给一点膨胀

        # 提取掩码部分
        channel = clahe.apply(channel)
        channel = cv2.bitwise_and(channel, channel, mask=mask)

        # 对每个区域单独二值化，减少光晕的影响
        num_labels, labels = cv2.connectedComponents(channel, connectivity=8)
        for i in range(1, num_labels):
            max = np.max(channel[labels == i])
            channel[labels == i] = np.where(channel[labels == i] > max / 255 * 245, max, 0)

        # 整体二值化
        _, binary = cv2.threshold(channel, 160, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    # 检测灯条
    def detect(self, frame):
        binary = self.preprocess(frame)

        # 提取轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        light_bars = []
        for cnt in contours:

            # 计算轮廓面积，过滤过小目标
            area = cv2.contourArea(cnt)
            if area < self.area_thresh:
                continue
            
            # 生成旋转矩形
            min_rect = cv2.minAreaRect(cnt)
            _, (w, h), angle = sp.standardRect(min_rect)
            aspect_ratio = h / w
            
            # 筛选：长宽比+角度符合要求
            if (
                self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]
                and
                self.angle_range[0] <= angle <= self.angle_range[1]
                ):

                light_bars.append(min_rect)
        
        return light_bars