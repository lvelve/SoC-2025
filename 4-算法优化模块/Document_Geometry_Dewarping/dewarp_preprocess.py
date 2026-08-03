import cv2
import numpy as np
import os


def extract_edge_features(gray):
    """
    对输入的灰度图像进行二值化并提取边缘特征。

    参数:
        gray: 输入的灰度图像 (numpy array, 单通道)

    返回:
        edge_image: 边缘特征图像 (numpy array, 单通道, 0/255)
    """
    # 使用 OTSU 自动阈值进行二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学闭操作：消除白色区域内的黑色噪点
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 使用 Canny 边缘检测提取边缘特征
    edges = cv2.Canny(binary, 200, 255)

    return edges


def main():
    #对 b_mask 图像进行边缘特征提取
    b_mask_dir = os.path.join("publictest/masks/", "rotate")
    result1_dir = os.path.join("publictest/masks/", "mask_b/rotate")
    os.makedirs(result1_dir, exist_ok=True)

    if not os.path.isdir(b_mask_dir):
        print(f"目录不存在: {b_mask_dir}")
        return

    b_mask_files = [f for f in os.listdir(b_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n共找到 {len(b_mask_files)} 个 b_mask 图像，开始提取边缘特征...")

    edge_count = 0
    for filename in b_mask_files:
        img_path = os.path.join(b_mask_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"无法读取图像: {img_path}，跳过")
            continue

        edge_image = extract_edge_features(image)
        
        # 将 00000_mask.png 格式保存为 00000.png
        save_name = filename.replace("_mask", "")
        save_path_b = os.path.join(result1_dir, save_name)
        cv2.imwrite(save_path_b, edge_image)
        edge_count += 1
        print(f"边缘特征已保存: {save_path_b}")

    print(f"\n边缘特征提取完成！共处理 {edge_count} 张图像，结果保存在 {result1_dir}")


if __name__ == "__main__":
    main()