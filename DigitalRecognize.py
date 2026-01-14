import cv2
import numpy as np
import support as sp

class DigitalRecognizer:
    def __init__(self, hog, svm):
        self.hog = hog
        self.svm = svm
    
    # 扩展装甲板四边形
    def expandArmor(self, armor, ratio=2):
        pt1, pt2, pt3, pt4 = armor.points
        p1, p2 = sp.expandLine(pt1, pt2, ratio)
        p3, p4 = sp.expandLine(pt3, pt4, ratio)

        # 确保第一个顶点在左上角，其它顶点按逆时针顺序
        if p1[1] > p2[1]:
            p1, p2, p3, p4 = p2, p1, p4, p3
        if p2[0] > p3[0]:
            p1, p2, p3, p4 = p4, p3, p2, p1

        return np.array((p1, p2, p3, p4), dtype=np.int32)

    # 从扩展的装甲板四边形提取数字ROI
    def getROI(self, img, shape):

        # 透视变换
        pts1 = np.float32(shape)
        pts2 = np.float32(((0, 0), (0, 50), (50, 50), (50, 0)))
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # 裁剪
        roi = cv2.warpPerspective(img, M, (50, 50)) 

        return roi
    
    # 处理数字ROI，提取数字，并把数字图像标准化
    def extractDigital(self, roi):

        # 使用单通道，进行阈值处理
        channel = roi[:,:, 1]
        _, binary = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 找到最大的连通域，即为数字
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels < 2:
            return None
        i = np.argmax(stats[1:, 4])
        x, y, w, h = stats[i+1, :4]
        digital = binary[y: y+h, x: x+w]
        
        # 把数字放到28*28的图像中居中显示，和图像边界保持一定的距离
        length = max(w, h)
        ratio = 20 / length
        w_small = int(w * ratio)
        h_small = int(h * ratio)

        digital_small = cv2.resize(digital, (w_small, h_small), 
                                    interpolation=cv2.INTER_NEAREST)
        zone = np.zeros((28, 28), dtype=np.uint8)
        x1 = int(14 - w_small / 2)
        y1 = int(14 - h_small / 2)
        x2 = int(14 + w_small / 2)
        y2 = int(14 + h_small / 2)
        zone[y1: y2, x1: x2] = digital_small

        return zone
    
    # 识别数字，返回整数
    def recognize(self, zone):
        feature = self.hog.compute(zone).flatten()
        features = np.array([feature], dtype=np.float32)
        _, pred = self.svm.predict(features)
        return int(pred[0][0])
    
    # 运行识别流程
    def run(self, frame, armor ,draw=False):
        shape = self.expandArmor(armor)
        roi = self.getROI(frame, shape)
        if roi is None:
            return None
        digital = self.extractDigital(roi)
        pred = self.recognize(digital)
        if draw:
            cv2.putText(frame, str(pred), shape[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
        return pred