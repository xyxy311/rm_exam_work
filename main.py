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

    # frame = cv2.imread('video_and_image/10.png')
    # if frame is None:
    #     exit("打不开！")
    
    # # 调整图像大小
    # ratio = 800 / max(frame.shape[:2])
    # frame = cv2.resize(frame, None, fx=ratio, fy=ratio)

    # cv2.imshow("frame", frame)

    # lights = detector.run(frame)
    # for i, light in enumerate(lights):
    #     light.drawLight(frame)
    # armors = matcher.matchLight(lights)
    # for armor in armors:
    #     recor.run(frame, armor, True)
    #     armor.drawArmor(frame)
    # cv2.imshow("evnetual", frame)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture("video_and_image\\test03.mp4")
    _, frame = cap.read()
    h, w = frame.shape[:2]
    ratio = 800 / max(h, w)
    frame_width = int(w * ratio)
    frame_height = int(h * ratio)
    # video_writer = cv2.VideoWriter(
    #     "video_and_image/gangle.mp4",                  # 输出视频路径
    #     cv2.VideoWriter_fourcc(*'mp4v'),# 编码器
    #     15,                           # 帧率
    #     (frame_width, frame_height)    # 帧分辨率（宽，高）
    # )

    while cv2.waitKey(10) & 0xFF != ord('q'):
        ret, frame = cap.read()
        if not ret:
            break

        # 调整图像大小
        frame = cv2.resize(frame, (frame_width, frame_height))
        cv2.imshow("frame", frame)
    
        lights = detector.run(frame)
        for light in lights:
            light.drawLight(frame)
        armors = matcher.matchLight(lights)
        for armor in armors:
            recor.run(frame, armor, True)
            armor.drawArmor(frame, False)
        cv2.imshow("Light Bar Detection", frame)
        # video_writer.write(frame)

    cap.release()
    # video_writer.release()
    cv2.destroyAllWindows()