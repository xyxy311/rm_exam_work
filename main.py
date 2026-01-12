import cv2
import LightBarDetector as LBD
import LightBarMatch as LBM
import DigitalRecognize as DR

hog = cv2.HOGDescriptor((28, 28), (8, 8), (4, 4), (4, 4), 9)
svm = cv2.ml.SVM_load('train/digitRecogniser.xml')

# 测试
if __name__ == "__main__":
    detector = LBD.LightBarDetector(enemy_color="blue")
    matcher = LBM.LightBarMatch()
    recor = DR.DigitalRecognizer(hog, svm)

    # frame = cv2.imread('video_and_image\\image2.png')
    
    # # 调整图像大小
    # ratio = 800 / max(frame.shape[:2])
    # frame = cv2.resize(frame, None, fx=ratio, fy=ratio)

    # cv2.imshow("frame", frame)

    # lights = detector.run(frame)
    # for light in lights:
    #     light.drawLight(frame)
    # armors = matcher.matchLight(lights)
    # for armor in armors:
    #     armor.drawArmor(frame)

    # cv2.imshow("evnetual", frame)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture("video_and_image\\test03.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 调整图像大小
        ratio = 800 / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
        cv2.imshow("frame", frame)
    
        lights = detector.run(frame)
        for light in lights:
            light.drawLight(frame)
        armors = matcher.matchLight(lights)
        for armor in armors:
            armor.drawArmor(frame)
            recor.run(frame, armor, True)
        cv2.imshow("Light Bar Detection", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()

    cv2.destroyAllWindows()