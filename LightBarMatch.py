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
    def rectToArmor(self, bar1, bar2):

        # 计算平均角度作为装甲板两边的倾斜角度
        angle1 = bar1[2]
        angle2 = bar2[2]
        armor_angle = (angle1 + angle2) / 2

        line1 = sp.rectToline(bar1)
        line2 = sp.rectToline(bar2)

        armor = (line1[0], line1[1], line2[1], line2[0])

        return np.int32(armor)  # 返回装甲板矩形

    # 灯条配对算法，返回配对成功的矩形（即装甲板）
    def matchLight(self, light_bars, gray_img):

        matched = [[], [], []] # 标记已配对成功的灯条索引和分数，用于去重
        armors = []  # 配对成功的装甲板矩形列表，带有分数
        rects = list(map(sp.standardRect, light_bars))  # 统一旋转矩形

        for i1, rect1 in enumerate(rects[:len(light_bars) - 1]):
            score_max = -1
            index_max = -1  # 配对得分最高的灯条索引

            for i2, rect2 in enumerate(rects[i1 + 1:], i1 + 1): # 避免重复配对
                angle_abs = self.filterLight(rect1, rect2)

                # 初步筛选成功，通过角度差和白色像素数量占比计算配对得分
                if angle_abs is not None:
                    whites = self.caculateWhites(rect1, rect2, gray_img)
                    score = 2 * whites - (angle_abs / self.thresh_angle)

                    # 更新最大得分和配对灯条索引
                    if score > score_max:
                        score_max = score
                        index_max = i2

            if index_max != -1:  # 配对成功
                if i1 not in matched[1] and index_max not in matched[1]:
                    matched[0].append(i1)
                    matched[1].append(index_max)
                    matched[2].append(score_max)
                else:  # 避免重复配对
                    for i, i2 in enumerate(matched[1]):
                        if i2 == index_max:
                            break  # 先找到重复配对的灯条的配对分数和索引
                    if score < score_max:  # 保留得分更高的配对
                        del matched[0][i], matched[1][i], matched[2][i]
                        matched[0].append(i1)
                        matched[1].append(index_max)
                        matched[2].append(score_max)

        if matched[0]:  # 配对成功
            for i1, i2, score in zip(matched[0], matched[1], matched[2]):
                armor = self.rectToArmor(rects[i1], rects[i2])
                armors.append((armor, score))
        return armors