import cv2
import numpy as np

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
        if self.enemy_color == "red":
            # 红方则提取红色通道
            channel = frame[:, :, 2]
        else:  # blue
            channel = frame[:, :, 0]
        
        # 二值化
        _, binary = cv2.threshold(channel, 220, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary
    
    # 统一旋转矩形的w、h、angle
    def standardRect(self, Rect):
        _, (w, h), angle = Rect
        if w > h:
            w, h = h, w
        else:
            angle += 90
        return w, h, angle

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
            w, h, angle = self.standardRect(min_rect)
            aspect_ratio = h / w
            
            # 筛选：长宽比+角度符合要求
            if (
                self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]
                and
                self.angle_range[0] <= angle <= self.angle_range[1]
                ):
                box = cv2.boxPoints(min_rect).astype(np.int32)  # 转换为矩形顶点
                light_bars.append(box)
        
        return light_bars


# 测试
if __name__ == "__main__":
    detector = LightBarDetector(enemy_color="red")

    frame = cv2.imread('video_and_image\\image2.png')
    de_1 = detector.preprocess(frame)

    bars = detector.detect(frame)

    cv2.drawContours(frame, bars, -1, (0, 255, 0), 2)  # 绿色框标记灯条
    cv2.imshow("Light Bar Detection", frame)
    cv2.imshow('1', de_1)

    # cap = cv2.VideoCapture("video_and_image\\test02.mp4")

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
        
    #     light_bars = detector.detect(frame)
        
    #     # 绘制旋转矩形
    #     for rect in light_bars:
    #         box = cv2.boxPoints(rect).astype(np.int32)  # 转换为矩形顶点
    #         cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)  # 绿色框标记灯条
        
    #     cv2.imshow("Light Bar Detection", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()