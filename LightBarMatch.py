import cv2
import numpy as np

# 装甲板类
class Armor:
    def __init__(self, points, score, type=None):
        self.points = points  # 装甲板矩形顶点坐标
        self.score = score  # 装甲板得分
        self.type = type    # 装甲板类型

    def drawArmor(self, frame, info=False):
        cv2.drawContours(frame, [self.points], 0, (0, 255, 0), 2)
        if info:
            cv2.putText(frame, str(self.score), self.points[0], 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


class LightBarMatch:
    def __init__(self, thresh_angle=10):
        self.thresh_angle = thresh_angle  # 允许的最大角度差

    # 初步筛选灯条对
    def filterLight(self, light1, light2):
        angle_diff = abs(light1.angle - light2.angle)
        x_diff = abs(light1.pos[0] - light2.pos[0])
        y_diff = abs(light1.pos[1] - light2.pos[1])
        thresh_x = (light1.length + light2.length) * 1.72
        ''' 注：
        1.72 = 230(大装甲板宽度) / 67(灯条高度) / 2(取两个灯条的平均值作为平均宽度)
        '''
        thresh_y = (light1.length + light2.length) / 2

        if angle_diff > self.thresh_angle or \
            x_diff > thresh_x or y_diff > thresh_y:
            return None        # 超过阈值，直接斩杀
        else:
            return angle_diff  # 返回角度差

    # 将配对成功的灯条转换成装甲板矩形
    def rectToArmor(self, light1, light2):

        # 计算平均角度作为装甲板两边的倾斜角度
        line1 = light1.getEndPoints()
        line2 = light2.getEndPoints()

        armor = (line1[0], line1[1], line2[1], line2[0])

        return np.int32(armor)  # 返回装甲板矩形

    # 灯条配对算法，返回配对成功的矩形（即装甲板）
    def matchLight(self, lights):

        matched = [[], [], []] # 标记已配对成功的灯条索引和分数，用于去重
        armors = []  # 配对成功的装甲板列表

        for i1, rect1 in enumerate(lights[: len(lights)-1]):
            score_max = 0
            i2_max = -1  # 和i1灯条配对得分最高的灯条索引

            for i2, rect2 in enumerate(lights[i1 + 1:], i1 + 1): # 避免重复配对
                angle_diff = self.filterLight(rect1, rect2)

                # 初步筛选成功，通过角度差计算配对得分
                if angle_diff is not None:
                    score = 1 - (angle_diff / self.thresh_angle)

                    # 更新最大得分和配对灯条索引
                    if score > score_max:
                        score_max = score
                        i2_max = i2

            if i2_max != -1:  # 配对成功
                if i1 not in matched[1] and i2_max not in matched[1]:
                    matched[0].append(i1)
                    matched[1].append(i2_max)
                    matched[2].append(score_max)
                else:  # 避免重复配对
                    for i, i2 in enumerate(matched[1]):
                        if i2 == i2_max:
                            break  # 先找到重复配对的灯条的配对分数和索引
                    if score < score_max:  # 保留得分更高的配对
                        del matched[0][i], matched[1][i], matched[2][i]
                        matched[0].append(i1)
                        matched[1].append(i2_max)
                        matched[2].append(score_max)

        if matched[0]:  # 配对成功
            for i1, i2, score in zip(matched[0], matched[1], matched[2]):
                armor = self.rectToArmor(lights[i1], lights[i2])
                armor_class = Armor(armor, score)
                armors.append(armor_class)
        return armors