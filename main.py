import cv2
import numpy as np
import LightBarDetector as LBD
import LightBarMatch as LBM

# 测试
if __name__ == "__main__":
    detector = LBD.LightBarDetector(enemy_color="red")
    matcher = LBM.LightBarMatch()

    # frame = cv2.imread('video_and_image\\image.png')
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # bars = detector.detect(frame)
    # for bar in bars:
    #     box = cv2.boxPoints(bar).astype(np.int32)  # 转换为矩形顶点
    #     cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)  # 绿色框标记灯条

    # armors = matcher.matchLight(bars, gray_frame)
    # for armor in armors:
    #     box = np.int32(armor)
    #     cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    # cv2.imshow("Light Bar Detection", frame)
    cv2.waitKey(0)

    cap = cv2.VideoCapture("video_and_image\\test01.avi")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bars = detector.detect(frame)
        for bar in bars:
            box = cv2.boxPoints(bar).astype(np.int32)  # 转换为矩形顶点
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)  # 绿色框标记灯条

        armors = matcher.matchLight(bars, gray_frame)
        for armor, score in armors:
            cv2.putText(frame, f"score:{score:.2f}", armor[3], 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.drawContours(frame, [armor], 0, (0, 0, 255), 2)

        cv2.imshow("Light Bar Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()