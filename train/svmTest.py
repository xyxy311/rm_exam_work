import cv2
import numpy as np

hog = cv2.HOGDescriptor((28, 28), (8, 8), (4, 4), (4, 4), 9)
svm = cv2.ml.SVM_load('train/digitRecogniser.xml')

# 测试单张图片
def recognize_one_digit(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (28, 28))
    feature = hog.compute(img).flatten()
    features = np.array([feature], dtype=np.float32)

    _, pred = svm.predict(features)

    return pred

pred = recognize_one_digit('video_and_image/d2.png')
print(pred)