import cv2
import numpy as np
import os
from glob import glob

TRAINSET = "train/digital/train_data"
TESTSET = "train/digital/test_data"
SVM_MODEL_PATH = "train/digitRecogniser.xml"
IMG_SIZE = (28, 28)

# 加载数据集（图片），制作成HOG特征
def load_and_extract_hog(path=TRAINSET) -> tuple[np.ndarray, np.ndarray]:

    # 初始化HOG
    hog = cv2.HOGDescriptor((28, 28), (8, 8), (4, 4), (4, 4), 9)
    
    features = []  # 存储HOG特征
    labels = []    # 存储标签
    
    # 遍历每个数字文件夹
    for digit in range(10):
        digit_dir = os.path.join(path, str(digit))
        if not os.path.exists(digit_dir):
            print(f"数字{digit}文件夹不存在：{digit_dir}")
            continue
        
        img_paths = glob(os.path.join(digit_dir, "*"))  # 匹配所有图片
        print(f"处理数字{digit}，共{len(img_paths)}张二值图...")
        
        for img_path in img_paths:
            img_28x28 = cv2.imread(img_path, 0)
            if img_28x28 is None:
                continue
            
            # 提取HOG特征并展平
            hog_feature = hog.compute(img_28x28).flatten()
            features.append(hog_feature)
            labels.append(digit)
    
    # 转换为格式
    features = np.array(features, dtype=np.float32)  # 特征是float32
    labels = np.array(labels, dtype=np.int32).reshape(-1, 1)  # 标签是int32列向量
    
    return features, labels

# 训练SVM模型
def train_svm(features: np.ndarray, labels: np.ndarray):

    # 初始化SVM（分类任务+线性核，适合数字识别）
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-3))
    
    # 训练模型
    print("\n训练中...")
    svm.train(features, cv2.ml.ROW_SAMPLE, labels)
    
    # 保存模型
    svm.save(SVM_MODEL_PATH)
    print(f"模型训练完成！已保存到：{SVM_MODEL_PATH}")
    
    # 测试训练效果
    features_test, labels_test = load_and_extract_hog(TESTSET)
    ret, result = svm.predict(features_test)
    print(f"测试集准确率：{np.mean(result == labels_test):.2f}")

    return svm

# 主函数
if __name__ == "__main__":

    # 加载二值图数据集并提取HOG特征
    features, labels = load_and_extract_hog()
    
    # 训练并保存模型
    svm = train_svm(features, labels)