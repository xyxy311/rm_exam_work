import cv2
import numpy as np
import support as sp

class LightBarMatch:
    def __init__(self, thresh_angle=10):
        self.thresh_angle = thresh_angle  # 允许的最大角度差

    # 初步筛选灯条对
    def filterLight(self, rect1, rect2):
        angle_diff = abs(rect1[2] - rect2[2])
        x_diff = abs(rect1[0][0] - rect2[0][0])
        y_diff = abs(rect1[0][1] - rect2[0][1])
        thresh_x = (rect1[1][1] + rect2[1][1]) * 1.72
        ''' 注：
        1.72 = 230(大装甲板宽度) / 67(灯条高度) / 2(取两个灯条的平均值作为平均宽度)
        '''
        thresh_y = (rect1[1][1] + rect2[1][1]) / 2

        if angle_diff > self.thresh_angle or \
            x_diff > thresh_x or y_diff > thresh_y:
            return None        # 超过阈值，直接斩杀
        else:
            return angle_diff  # 返回角度差
        
    # 计算白色像素占比（数字是白色的）
    def caculateWhites(self, rect1, rect2, graya_img):
        x1, y1 = rect1[0]
        x2, y2 = rect2[0]
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        white_pixes = 0
        for x in range(int(min(x1, x2)), int(max(x1, x2))):
            y = int(k * x + b)
            if graya_img[y, x] >= 127:
                white_pixes += 1
        return white_pixes / abs(x2 - x1)

    # 将配对成功的灯条转换成装甲板矩形
    def rect2Armor(self, bar1, bar2):
        # 确保 bar1 在左边
        if bar1[0][0] > bar2[0][0]:
            bar1, bar2 = bar2, bar1

        # 转换为矩形顶点
        box1 = cv2.boxPoints(bar1).astype(np.int32)
        box2 = cv2.boxPoints(bar2).astype(np.int32)

        # 合并矩形
        box = np.array((box1, box2)).reshape((8, 2))
        armor = cv2.minAreaRect(box)

        return armor  # 返回旋转矩形

    # 灯条配对算法，返回配对成功的矩形（即装甲板）
    def matchLight(self, light_bars, gray_img):
        armors = []  # 配对成功的装甲板矩形列表

        for i1, bar1 in enumerate(light_bars[:len(light_bars) - 1]):
            score_max = -1
            index_max = -1  # 配对得分最高的灯条索引

            for i2, bar2 in enumerate(light_bars[i1 + 1:], i1 + 1): # 避免重复配对

                # 统一度量衡
                rect1 = sp.standardRect(bar1)
                rect2 = sp.standardRect(bar2)

                angle_abs = self.filterLight(rect1, rect2)

                # 初步筛选成功，通过角度差和白色像素数量占比计算配对得分
                if angle_abs is not None:
                    whites = self.caculateWhites(rect1, rect2, gray_img)
                    score = whites - (angle_abs / self.thresh_angle)

                    # 更新最大得分和配对灯条索引
                    if score > score_max:
                        score_max = score
                        index_max = i2

            if index_max != -1:  # 配对成功
                armors.append(self.rect2Armor(bar1, light_bars[index_max]))
        return armors