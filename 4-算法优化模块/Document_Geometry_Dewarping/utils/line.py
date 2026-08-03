import cv2
import numpy as np
from skimage import morphology
from numpy import *

# ---------------------------------------------------------------------------
# 自适应形态学去噪预处理
# ---------------------------------------------------------------------------

# 可调参数（基于 00031 / 00044 样本实测留有余量）
_MIN_COMPONENT_AREA = 50       # 过滤噪声小点的面积阈值
_GAP_RATIO_HIGH = 1.5          # case‑a 判定：行间距 / 行高
_GAP_RATIO_LOW = 0.9           # case‑b 判定
_DENSITY_LOW = 20              # case‑a 判定：组件 / 百万像素
_DENSITY_HIGH = 35             # case‑b 判定


def preprocess_mask(binary, scale=1.0):
    """
    自适应形态学去噪预处理。

    Step 1（必须）: 闭运算(3×3) 去除白色区域内的黑点噪声。
    Step 2（分析）: 在闭运算结果上做轻量连通域分析，
                    计算 Gap/Height ratio（行间距/行高中位数）和
                    density（有效组件数 / 百万像素）。
    Step 3（自适应）:
        • Case a — 行稀疏且高（gap_ratio > 1.5 或 density < 20）:
            开运算核 = 7×7，平滑锯齿边缘。
        • Case b — 行密集（gap_ratio < 0.8 或 density > 40）:
            跳过开运算，避免行间粘连。
        • 默认（中间地带）:
            开运算核 = 5×5（保持原有行为）。

    参数:
        binary : 二值图像 (uint8, 0/255)
        scale  : 当前缩放比例（用于调整面积阈值）

    返回:
        处理后的二值图像 (uint8, 0/255)
    """
    # ---- Step 1: 闭运算去黑点（必须） ----
    close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kern)

    # ---- Step 2: 轻量连通域分析 ----
    min_area = max(10, int(_MIN_COMPONENT_AREA * scale * scale))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    valid_indices = [i for i in range(1, n_labels) if stats[i][4] > min_area]
    n_valid = len(valid_indices)
    h, w = binary.shape[:2]
    density = n_valid / (h * w) * 1e6  # 组件 / 百万像素

    heights = [stats[i][3] for i in valid_indices]
    median_h = float(np.median(heights)) if heights else 0.0

    gap_ratio = 0.0
    if n_valid >= 2 and median_h > 0:
        cy_sorted = sorted([centroids[i][1] for i in valid_indices])
        gaps = [cy_sorted[k + 1] - cy_sorted[k] for k in range(len(cy_sorted) - 1)]
        gap_ratio = float(np.median(gaps)) / median_h

    # ---- Step 3: 自适应开运算 ----
    is_sparse = (gap_ratio > _GAP_RATIO_HIGH) or (density < _DENSITY_LOW)
    is_dense = (gap_ratio < _GAP_RATIO_LOW) or (density > _DENSITY_HIGH)

    if is_sparse:
        # Case a: 稀疏大字 → 7×7 核平滑边缘锯齿
        open_size = 7
    elif is_dense:
        # Case b: 密集小字 → 跳过开运算
        open_size = 0
    else:
        # 默认
        open_size = 5

    if open_size > 0:
        open_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (open_size, open_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kern)
    print(f"open_size = {open_size}") 
    return binary


def line(img, target_size=512):
    """
    在高分辨率下进行 skeletonize 和连通域分析，避免相邻文本行粘连。
    最后将提取的坐标按比例缩放回 target_size 坐标空间。

    缩放逻辑：
        短边 < 2048  → 不缩放
        短边 >= 2048  → 缩小为 1/2
        短边 >= 4096  → 缩小为 1/4

    参数:
        img          : 二值图像（任意尺寸，灰度或二值）
        target_size  : 输出坐标的参考尺寸（默认512，输出坐标在 [0, target_size) 空间）

    返回:
        points: 文本行点集列表，每个元素为 (K, 2) float32 数组，
                第0列为 x（col），第1列为 y（row），坐标范围 [0, target_size)
    """
    img = img.astype(np.uint8)
    h, w = img.shape[:2]

    # 根据短边长度决定缩放比例：过大时缩小以提高处理效率
    short_edge = min(h, w)
    if short_edge >= 4096:
        scale = 0.25
    elif short_edge >= 2048:
        scale = 0.5
    else:
        scale = 1.0

    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_large = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        img_large = img

    _, th = cv2.threshold(img_large, 0, 255, cv2.THRESH_BINARY)

    # 自适应形态学去噪：闭运算去黑点 + 根据文本密度/行高自适应选择开运算策略
    th = preprocess_mask(th, scale)

    th = morphology.skeletonize(th / 255).astype(np.uint8) * 255

    _, labels, stats, _ = cv2.connectedComponentsWithStats(th)

    # 宽度阈值按缩放比例调整（原始阈值为 60，对应 512 空间下的像素数）
    width_threshold = int(60 * scale)

    i = 1
    points = []
    for stat in stats[1:]:
        point = []
        if stat[2] > width_threshold:
            inter = max(1, int(16 * scale))  # 采样间隔按缩放比例调整
            line = np.where(labels.T == i)
            for j in range(int(len(line[0]) / inter)):
                point.append([line[0][inter * j], line[1][inter * j]])
            point.append([line[0][-1], line[1][-1]])
            if point:
                # 将坐标从高分辨率空间缩放回 target_size 空间
                pts = np.array(point).astype(np.float32)
                pts[:, 0] = pts[:, 0] / scale / w * target_size  # x (col)
                pts[:, 1] = pts[:, 1] / scale / h * target_size  # y (row)
                points.append(pts)
        i += 1
    return points


def visualize_components(img, img1, width_threshold=60):
    """对输入图像进行二值化、连通组件分析，仅可视化通过宽度筛选的连通域。

    筛选条件与 line() 一致：只保留宽度 > width_threshold 的连通域。
    在高分辨率下进行 skeletonize 以避免相邻文本行粘连，然后将结果缩放回原始尺寸。

    缩放逻辑（与 line() 一致）：
        短边 < 2048  → 不缩放
        短边 >= 2048  → 缩小为 1/2
        短边 >= 4096  → 缩小为 1/4

    可视化内容：
    1. 通过筛选的连通域用随机颜色绘制
    2. 未通过筛选的连通域不显示（背景为黑色）
    3. 绘制边界框并标注组件ID、宽度

    参数:
        img: 灰度图或二值图
        img1: 用于叠加显示的原始图像（BGR）
        width_threshold: 宽度筛选阈值，默认60（与line()一致）

    返回:
        vis: BGR可视化图像（仅包含筛选后的连通域，尺寸与 img1 一致）
    """
    img = img.astype(np.uint8)
    h, w = img.shape[:2]

    # 根据短边长度决定缩放比例：过大时缩小以提高处理效率（与 line() 一致）
    short_edge = min(h, w)
    if short_edge >= 4096:
        scale = 0.25
    elif short_edge >= 2048:
        scale = 0.5
    else:
        scale = 1.0

    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_large = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        img_large = img

    _, th = cv2.threshold(img_large, 0, 255, cv2.THRESH_BINARY)



    # 自适应形态学去噪：闭运算去黑点 + 根据文本密度/行高自适应选择开运算策略
    th = preprocess_mask(th, scale)

    # 保存平滑后的二值图用于可视化背景，并缩放回原始尺寸
    if scale != 1.0:
        test = cv2.resize(th, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        test = th#.copy()

    th = morphology.skeletonize(th / 255).astype(np.uint8) * 255

    # 将骨架化结果缩放回原始尺寸，用于可视化叠加
    if scale != 1.0:
        th_vis = cv2.resize(th, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        th_vis = th

    # 在原始尺寸上做连通域分析（用于可视化标注）
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th_vis)
    scaled_width_threshold = width_threshold  # 在原始尺寸下使用原始阈值

    # 构建掩码：仅保留宽>width_threshold的连通域
    mask = np.zeros_like(th_vis)
    filtered_count = 0
    filtered_ids = []
    for i in range(1, n_labels):
        x, y, w_c, h_c, area = stats[i]
        if w_c > scaled_width_threshold:
            mask[labels == i] = 255
            filtered_count += 1
            filtered_ids.append(i)

    print(f'连通域总数: {n_labels - 1}, 通过筛选(宽>{scaled_width_threshold}): {filtered_count}')

    # 生成颜色表（仅为筛选后的连通域分配颜色）
    colors = np.zeros((n_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景黑色
    for idx, i in enumerate(filtered_ids):
        hue = (idx * 137.508) % 360  # 黄金角度，颜色均匀分布
        saturation = 200 + np.random.randint(0, 55)
        value = 200 + np.random.randint(0, 55)
        hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors[i] = rgb

    # 创建彩色标签图（未筛选的连通域为黑色）
    vis = colors[labels]
    # 需要在原图上，使用colors进行涂色。
    vis[labels == 0] = [0, 0, 0]  # 背景透明
    # 将彩色标签叠加到原图上，但只在有标签的区域
    mask = (labels != 0).astype(np.uint8) * 255
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    # result = img1.copy() # 彩色图为背景
    result = cv2.merge([test, test, test]) # 文本行特征图（灰度图）为背景
    result = result * (1 - mask_3ch) + vis * mask_3ch
    result = result.astype(np.uint8)
    cv2.imshow("test", result)
    cv2.waitKey(0)

    # 标注信息
    for i in filtered_ids:
        x, y, w, h, area = stats[i]
        # 绘制边界框
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
        # 标注组件ID和宽度 W:{w}
        label_text = f"ID:{i}"
        cv2.putText(vis, label_text, (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # 标记质心
        cx, cy = centroids[i]
        cv2.circle(vis, (int(cx), int(cy)), 2, (255, 255, 255), -1)

    return result
