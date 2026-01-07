import cv2
import numpy as np
import LightBarDetector as LBD
import LightBarMatch as LBM

# 测试
if __name__ == "__main__":
    detector = LBD.LightBarDetector(enemy_color="red")
    matcher = LBM.LightBarMatch()

    frame = cv2.imread('video_and_image\\image.png')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bars = detector.detect(frame)
    for bar in bars:
        box = cv2.boxPoints(bar).astype(np.int32)  # 转换为矩形顶点
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)  # 绿色框标记灯条

    armors = matcher.matchLight(bars, gray_frame)
    for armor in armors:
        box = cv2.boxPoints(armor).astype(np.int32)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    cv2.imshow("Light Bar Detection", frame)

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