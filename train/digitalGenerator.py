import cv2
import numpy as np
import os
import random
from pathlib import Path

# 配置参数
CONFIG = {
    "output_dir": "./train/digital",  # 数据集根目录
    "train_num": 1000,
    "test_num": 100,
    "img_size": 28,
    "font_types": [
                    # cv2.FONT_HERSHEY_SIMPLEX,   # 字体类型
                    cv2.FONT_HERSHEY_PLAIN,
                    ],
    "font_scales": [2, 2.1],           # 字体大小
    "rot_angles": [-3, 0, 3],          # 旋转角度
    "thickness": [2, 3]
}

# 2. 创建目录结构
def create_dirs():
    for split in ["train_data", "test_data"]:
        split_dir = Path(CONFIG["output_dir"]) / split
        for digit in range(10):
            (split_dir / str(digit)).mkdir(parents=True, exist_ok=True)
    print("目录结构创建完成")

# 3. 生成单张印刷体数字图片
def generate_digit_img(digit):

    # 创建画布
    img = np.zeros((CONFIG["img_size"], CONFIG["img_size"]), dtype=np.uint8)
    
    # 随机选择字体参数
    font = random.choice(CONFIG["font_types"])
    font_scale = random.choice(CONFIG["font_scales"])
    thickness = random.choice(CONFIG["thickness"])
    
    # 计算文字居中放置的位置
    text = str(digit)
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (CONFIG["img_size"] - text_w) // 2
    y = (CONFIG["img_size"] + text_h) // 2
    
    # 绘制数字
    cv2.putText(img, text, (x, y), font, font_scale, 255, thickness)
    
    # 随机旋转
    rot_angle = random.choice(CONFIG["rot_angles"])
    center = (CONFIG["img_size"]//2, CONFIG["img_size"]//2)
    M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    img = cv2.warpAffine(img, M, (CONFIG["img_size"], CONFIG["img_size"]), borderValue=0)
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    
    return img

# 4. 批量生成数据集
def generate_dataset():
    create_dirs()
    
    # 生成训练集
    print(" 开始生成训练集...")
    train_count = {i:0 for i in range(10)}
    while sum(train_count.values()) < CONFIG["train_num"]:
        digit = random.randint(0, 9)
        if train_count[digit] >= CONFIG["train_num"]//10:
            continue
        # 生成图片
        img = generate_digit_img(digit)
        # 保存图片（命名：digit_count.jpg）
        save_path = Path(CONFIG["output_dir"]) / "train_data" / str(digit) / f"{train_count[digit]}.jpg"
        cv2.imwrite(str(save_path), img)
        train_count[digit] += 1
    
    # 生成测试集
    print("开始生成测试集...")
    test_count = {i:0 for i in range(10)}
    while sum(test_count.values()) < CONFIG["test_num"]:
        digit = random.randint(0, 9)
        if test_count[digit] >= CONFIG["test_num"]//10:
            continue
        img = generate_digit_img(digit)
        save_path = Path(CONFIG["output_dir"]) / "test_data" / str(digit) / f"{test_count[digit]}.jpg"
        cv2.imwrite(str(save_path), img)
        test_count[digit] += 1
    
    print(f"数据集生成完成！")
    print(f"   - 训练集：{CONFIG['train_num']}张（每个数字{CONFIG['train_num']//10}张）")
    print(f"   - 测试集：{CONFIG['test_num']}张（每个数字{CONFIG['test_num']//10}张）")
    print(f"   - 保存路径：{os.path.abspath(CONFIG['output_dir'])}")

# 5. 主函数
if __name__ == "__main__":
    generate_dataset()
    # img = generate_digit_img(4)
    # cv2.imshow('d', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()