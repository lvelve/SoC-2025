"""
从预备好的二值特征图像出发，调用 opt 方法建立形变场并矫正弯曲文档。
输入：
  --input   : 待矫正的原始图像
  --boundary: 边界特征图 (b_all.jpg)，512x512 二值图，白色像素为文档边界
  --textline: 文本行特征图 (t.png)，512x512 二值图，白色像素为文本行
  --output  : 矫正结果保存路径
  --debug   : 是否显示形变场可视化（默认 False）
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optim.opt import opt
from utils.line import line as extract_line
from utils.line1 import line1 as extract_line1


# ─────────────────────────────────────────────
# 1b. opt 输入参数可视化（debug 用）
# ─────────────────────────────────────────────

def visualize_opt_inputs(boundary_edges, textline_np, line1_np, n=128, save_path=None):
    """
    可视化传入 opt() 的三类输入参数，便于检查数据是否正确：
      - boundary_edges : (top, right, bottom, left) 四条边完整轮廓点，坐标在 [0, n-1]
      - textline_np    : 水平文本行点集列表，坐标在 [0, 512) 像素空间
      - line1_np       : 垂直文本行点集列表，坐标在 [0, 512) 像素空间

    子图布局：
      左上  — boundary_edges 四条边（网格坐标系，原点在左上，y 向下）
      右上  — 水平文本行 textline_np（512x512 像素坐标系）
      左下  — 垂直文本行 line1_np（512x512 像素坐标系）
      右下  — 三者叠加在同一坐标系（归一化到 [0,1]）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('opt() Input Visualization', fontsize=14)

    # ── 左上：boundary_edges 四条边 ──────────────────────────────
    ax = axes[0, 0]
    ax.set_title('boundary_edges (grid coords [0, n-1])')
    ax.set_xlim(-2, n + 2)
    ax.set_ylim(n + 2, -2)   # y 轴向下
    ax.set_aspect('equal')
    ax.set_xlabel('col (x)'); ax.set_ylabel('row (y)')
    ax.grid(True, alpha=0.3)

    if boundary_edges is not None:
        top, right, bottom, left = boundary_edges
        edge_data = [
            ('top',    top,    'green'),
            ('right',  right,  'blue'),
            ('bottom', bottom, 'orange'),
            ('left',   left,   'purple'),
        ]
        for name, pts, color in edge_data:
            # top/bottom 格式为 (col, row)；left/right 格式为 (row, col)，需交换
            if name in ('left', 'right'):
                cols, rows = pts[:, 1], pts[:, 0]
            else:
                cols, rows = pts[:, 0], pts[:, 1]
            ax.plot(cols, rows, '-', color=color, linewidth=1.5, label=name)
            ax.scatter(cols[::max(1, len(cols)//20)],
                       rows[::max(1, len(rows)//20)],
                       color=color, s=15, zorder=3)
            # 标注起点和终点
            ax.annotate('S', (cols[0],  rows[0]),  fontsize=8, color=color,
                         fontweight='bold')
            ax.annotate('E', (cols[-1], rows[-1]), fontsize=8, color=color,
                         fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        # 打印点数统计
        for name, pts, _ in edge_data:
            print(f"  boundary_edges {name}: {len(pts)} 点")
    else:
        ax.text(0.5, 0.5, 'boundary_edges is None',
                ha='center', va='center', transform=ax.transAxes)

    # ── 右上：水平文本行 textline_np ─────────────────────────────
    ax = axes[0, 1]
    ax.set_title(f'textline_np (horizontal, {len(textline_np)} lines, pixel [0,512])')
    ax.set_xlim(-5, 517)
    ax.set_ylim(517, -5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (col)'); ax.set_ylabel('y (row)')
    ax.grid(True, alpha=0.3)

    cmap_h = plt.cm.get_cmap('tab20', max(len(textline_np), 1))
    for idx, line_pts in enumerate(textline_np):
        color = cmap_h(idx % 20)
        # textline_np 坐标格式：[x, y]（col, row），line.py 对 labels.T 做 np.where，
        # 导致第0列为 x（col），第1列为 y（row）
        if line_pts.ndim == 2 and line_pts.shape[1] == 2:
            xs, ys = line_pts[:, 0], line_pts[:, 1]
        else:
            xs, ys = np.arange(len(line_pts)), line_pts
        ax.plot(xs, ys, '-', color=color, linewidth=1, alpha=0.8)
        ax.scatter(xs[::max(1, len(xs)//10)],
                   ys[::max(1, len(ys)//10)],
                   color=color, s=8, zorder=3)
    if not textline_np:
        ax.text(0.5, 0.5, 'No horizontal textlines',
                ha='center', va='center', transform=ax.transAxes)

    # ── 左下：垂直文本行 line1_np ────────────────────────────────
    ax = axes[1, 0]
    ax.set_title(f'line1_np (vertical, {len(line1_np)} lines, pixel [0,512])')
    ax.set_xlim(-5, 517)
    ax.set_ylim(517, -5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (col)'); ax.set_ylabel('y (row)')
    ax.grid(True, alpha=0.3)

    cmap_v = plt.cm.get_cmap('tab20b', max(len(line1_np), 1))
    for idx, line_pts in enumerate(line1_np):
        color = cmap_v(idx % 20)
        # line1_np 坐标格式同 textline_np：第0列为 x（col），第1列为 y（row）
        if line_pts.ndim == 2 and line_pts.shape[1] == 2:
            xs, ys = line_pts[:, 0], line_pts[:, 1]
        else:
            xs, ys = np.arange(len(line_pts)), line_pts
        ax.plot(xs, ys, '-', color=color, linewidth=1, alpha=0.8)
        ax.scatter(xs[::max(1, len(xs)//10)],
                   ys[::max(1, len(ys)//10)],
                   color=color, s=8, zorder=3)
    if not line1_np:
        ax.text(0.5, 0.5, 'No vertical textlines',
                ha='center', va='center', transform=ax.transAxes)

    # ── 右下：三者叠加（归一化到 [0,1]）────────────────────────
    ax = axes[1, 1]
    ax.set_title('All inputs overlaid (normalized [0,1])')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(1.02, -0.02)
    ax.set_aspect('equal')
    ax.set_xlabel('x (normalized)'); ax.set_ylabel('y (normalized)')
    ax.grid(True, alpha=0.3)

    # boundary_edges（已在 [0, n-1]，归一化除以 n-1）
    # top/bottom 格式为 (col, row)；left/right 格式为 (row, col)，需交换
    if boundary_edges is not None:
        for name, pts, color in edge_data:
            if name in ('left', 'right'):
                cols_n = pts[:, 1] / (n - 1)
                rows_n = pts[:, 0] / (n - 1)
            else:
                cols_n = pts[:, 0] / (n - 1)
                rows_n = pts[:, 1] / (n - 1)
            ax.plot(cols_n, rows_n, '-', color=color, linewidth=2,
                    label=f'boundary/{name}', alpha=0.9)

    # 水平文本行（像素坐标 / 512 归一化，第0列为x，第1列为y）
    for idx, line_pts in enumerate(textline_np):
        color = cmap_h(idx % 20)
        if line_pts.ndim == 2 and line_pts.shape[1] == 2:
            xs, ys = line_pts[:, 0] / 512, line_pts[:, 1] / 512
        else:
            xs = np.arange(len(line_pts)) / 512
            ys = line_pts / 512
        ax.plot(xs, ys, '--', color=color, linewidth=0.8, alpha=0.6,
                label='h-textline' if idx == 0 else '')

    # 垂直文本行（第0列为x，第1列为y）
    for idx, line_pts in enumerate(line1_np):
        color = cmap_v(idx % 20)
        if line_pts.ndim == 2 and line_pts.shape[1] == 2:
            xs, ys = line_pts[:, 0] / 512, line_pts[:, 1] / 512
        else:
            xs = np.arange(len(line_pts)) / 512
            ys = line_pts / 512
        ax.plot(xs, ys, ':', color=color, linewidth=0.8, alpha=0.6,
                label='v-textline' if idx == 0 else '')

    # 图例去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"opt 输入可视化已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 1c. 边缘采样 debug 可视化
# ─────────────────────────────────────────────

def _extract_boundary_edges(b_mask, n=128):
    """
    从边界二值 mask 中提取四条边的采样点（像素坐标）及中间结果。

    返回:
        top_pts, right_pts, bottom_pts, left_pts : 各 (n, 2) float32，xy 像素坐标
        corners : dict，键为 'tl','tr','br','bl'，值为 (2,) int 像素坐标
        contour : (K, 2) 最大轮廓点
        hull    : (M, 2) 凸包点
        H, W    : 原图尺寸
    """
    if len(b_mask.shape) == 3:
        gray = cv2.cvtColor(b_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = b_mask.copy()
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    H, W = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("边界特征图中未找到任何轮廓，请检查边缘mask图。")

    # 取最大轮廓，有利于去掉噪声。但是当轮廓存在断裂时，断裂的部分可能被舍弃。
    # contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    # 合并所有轮廓（假设它们都是边界的一部分）,代替取最大轮廓。如果有噪声可以考虑设置面积过滤。
    all_contour_points = []
    for cnt in contours:
        # 可选：过滤掉太小的噪点轮廓
        # if cv2.contourArea(cnt) < min_area_threshold:
        #     continue
        all_contour_points.extend(cnt.reshape(-1, 2))
    contour = np.array(all_contour_points)

    hull_idx = cv2.convexHull(contour, returnPoints=False).flatten()
    hull = contour[hull_idx]

    cx, cy = hull[:, 0].mean(), hull[:, 1].mean()
    angles = np.arctan2(hull[:, 1] - cy, hull[:, 0] - cx)
    hull = hull[np.argsort(angles)]

    tl = hull[np.argmin(hull[:, 0] + hull[:, 1])]
    tr = hull[np.argmin(-hull[:, 0] + hull[:, 1])]
    br = hull[np.argmin(-hull[:, 0] - hull[:, 1])]
    bl = hull[np.argmin(hull[:, 0] - hull[:, 1])]

    def sample_edge(p1, p2, num):
        xs = np.linspace(p1[0], p2[0], num)
        ys = np.linspace(p1[1], p2[1], num)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    top_pts    = sample_edge(tl, tr, n)
    right_pts  = sample_edge(tr, br, n)
    bottom_pts = sample_edge(br, bl, n)
    left_pts   = sample_edge(bl, tl, n)

    corners = {'tl': tl, 'tr': tr, 'br': br, 'bl': bl}
    return top_pts, right_pts, bottom_pts, left_pts, corners, contour, hull, H, W

def extract_boundary_edges_full(b_mask, n=128):
    """
    从边界二值 mask 中提取四条边的**完整轮廓点**（不做线性采样），
    完整保留边缘的弯曲特征，供 opt() 的 boundary_edges 参数直接使用。

    修改：直接从 contour 上采样点，不进行折线拟合插值。

    返回:
        (top, right, bottom, left) 四元组，每项为 (K, 2) float32 numpy 数组，
        坐标已归一化到 [0, n-1] 的网格坐标系，列顺序为 (row, col)，
        与 opt.py 中 grid() 函数的期望一致。
    """
    if len(b_mask.shape) == 3:
        gray = cv2.cvtColor(b_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = b_mask.copy()
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    H, W = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("边界特征图中未找到任何轮廓，请检查 b_all.jpg")

    # 合并所有轮廓（假设它们都是边界的一部分）
    all_contour_points = []
    min_area_threshold = 10  # 过滤小噪点
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_threshold:
            all_contour_points.extend(cnt.reshape(-1, 2))
    contour = np.array(all_contour_points)

    # 用凸包找四个角点
    hull_idx = cv2.convexHull(contour, returnPoints=False).flatten()
    hull = contour[hull_idx]
    cx, cy = hull[:, 0].mean(), hull[:, 1].mean()
    angles = np.arctan2(hull[:, 1] - cy, hull[:, 0] - cx)
    hull = hull[np.argsort(angles)]

    # 角点识别
    tl = hull[np.argmin(hull[:, 0] + hull[:, 1])]
    tr = hull[np.argmax(hull[:, 0] - hull[:, 1])]
    br = hull[np.argmax(hull[:, 0] + hull[:, 1])]
    bl = hull[np.argmax(hull[:, 1] - hull[:, 0])]

    # 在轮廓点序列中找各角点最近的索引
    def nearest_idx(contour, pt):
        dists = np.sum((contour - pt) ** 2, axis=1)
        return int(np.argmin(dists))

    # 提取坐标并转换为 float32
    tl_f = tl.astype(np.float32)
    tr_f = tr.astype(np.float32)
    br_f = br.astype(np.float32)
    bl_f = bl.astype(np.float32)

    # 计算斜率（提前检查垂直情况）
    delta1 = tl_f[0] - br_f[0]
    delta2 = bl_f[0] - tr_f[0]

    # 避免除零，使用条件判断
    if abs(delta1) < 1e-6:
        k1 = np.inf
        f1 = lambda x: np.where(x == tl_f[0], tl_f[1], np.nan)
    else:
        k1 = (tl_f[1] - br_f[1]) / delta1
        f1 = lambda x: k1 * (x - tl_f[0]) + tl_f[1]

    if abs(delta2) < 1e-6:
        k2 = np.inf
        f2 = lambda x: np.where(x == bl_f[0], bl_f[1], np.nan)
    else:
        k2 = (bl_f[1] - tr_f[1]) / delta2
        f2 = lambda x: k2 * (x - bl_f[0]) + bl_f[1]

    # 批量处理所有点
    pts_array = np.array(contour).squeeze()  # shape: (n, 2)
    x_vals = pts_array[:, 0]
    y_vals = pts_array[:, 1]

    # 计算分组条件
    cond1 = y_vals < f1(x_vals)
    cond2 = y_vals < f2(x_vals)

    # 根据条件组合分组。排序
    top_raw    = np.array(sorted(pts_array[(cond1) & (cond2)], key=lambda p: p[0])) # tl → tr（上边缘）col 递增
    bottom_raw = np.array(sorted(pts_array[(~cond1) & (~cond2)], key=lambda p: p[0]))  # br → bl（下边缘）col 递增
    left_raw   = np.array(sorted(pts_array[(~cond1) & (cond2)], key=lambda p: p[1]))   # bl → tl（左边缘）row 递增
    right_raw  = np.array(sorted(pts_array[(cond1) & (~cond2)], key=lambda p: p[1]))   # tr → br（右边缘）row 递增



    # 可选的均匀采样：如果点数过多或过少，可以均匀采样到 n 个点
    def uniform_sample_from_contour(pts, num):
        """
        从轮廓点序列中均匀采样 num 个点（索引均匀采样，不插值）
        """
        if len(pts) <= num:
            # 点数不足，直接返回全部
            return pts.astype(np.float32)
        
        # 按索引均匀采样
        indices = np.linspace(0, len(pts) - 1, num, dtype=int)
        return pts[indices].astype(np.float32)
    
    # 可选：统一采样到 n 个点（如果需要固定点数）
    # top_raw = uniform_sample_from_contour(top_raw, n)
    # right_raw = uniform_sample_from_contour(right_raw, n)
    # bottom_raw = uniform_sample_from_contour(bottom_raw, n)
    # left_raw = uniform_sample_from_contour(left_raw, n)

    # 坐标归一化
    def norm_col(pts): 
        return np.clip(pts[:, 0].astype(np.float32) / (W - 1) * (n - 1), 0, n - 1)
    def norm_row(pts): 
        return np.clip(pts[:, 1].astype(np.float32) / (H - 1) * (n - 1), 0, n - 1)

    top    = np.stack([norm_col(top_raw),    norm_row(top_raw)],    axis=1)
    bottom = np.stack([norm_col(bottom_raw), norm_row(bottom_raw)], axis=1)
    # left/right：(row, col) 格式
    left   = np.stack([norm_row(left_raw),   norm_col(left_raw)],   axis=1)
    right  = np.stack([norm_row(right_raw),  norm_col(right_raw)],  axis=1)

    print(f"  原始轮廓切片点数: top={len(top_raw)}, right={len(right_raw)}, "
          f"bottom={len(bottom_raw)}, left={len(left_raw)}")
    print(f"  归一化后点数: top={len(top)}, right={len(right)}, "
          f"bottom={len(bottom)}, left={len(left)}")

    # vis1 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # sides = {'top': top_raw, 'bottom': bottom_raw, 'left': left_raw, 'right': right_raw}
    # colors = {'top':    (0, 255, 0),    # 绿色
    #           'bottom': (0, 165, 255),  # 橙色
    #           'left':   (255, 0, 255),  # 紫色
    #           'right':  (255, 0, 0)}    # 蓝色
    # for name, pt in sides.items():
    #     for p in pt:
    #         cv2.circle(vis1, tuple(p.astype(int)), 5, colors[name], -1)
    #     cv2.putText(vis1, name, tuple(pt[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 10, colors[name], 2)
    # cv2.namedWindow("side", cv2.WINDOW_NORMAL)
    # cv2.imshow('side', vis1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # exit()

    # 可选：可视化调试（注释掉原版的 cv2 显示）
    # vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # corners = {'TL': tl, 'TR': tr, 'BR': br, 'BL': bl}
    # corner_colors = {'TL': (0, 0, 255), 'TR': (0, 0, 255), 
    #                  'BR': (0, 255, 0), 'BL': (0, 255, 0)}
    # for name, pt in corners.items():
    #     cv2.circle(vis, tuple(pt.astype(int)), 8, corner_colors[name], -1)
    #     cv2.putText(vis, name, (pt[0]+5, pt[1]-5), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 15, corner_colors[name], 2)
    # cv2.namedWindow("Contours, Convex Hull & Corners", cv2.WINDOW_NORMAL)
    # cv2.imshow('Contours, Convex Hull & Corners', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return top, right, bottom, left

# ─────────────────────────────────────────────
# 2. 图像矫正（grid_sample）
# ─────────────────────────────────────────────

def unwarp(img, grid, blur_ksize=1):
    """
    img  : HxWx3 numpy uint8
    grid : (n, n, 2) torch tensor，值域 [-1, 1]
    """
    h, w = img.shape[:2]
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).double()

    bm = grid.detach().numpy()
    bm0 = cv2.blur(bm[:, :, 0], (blur_ksize, blur_ksize))
    bm1 = cv2.blur(bm[:, :, 1], (blur_ksize, blur_ksize))
    bm0 = cv2.resize(bm0, (w, h))
    bm1 = cv2.resize(bm1, (w, h))
    bm = np.stack([bm0, bm1], axis=-1)
    bm = torch.from_numpy(bm[np.newaxis]).double()

    res = F.grid_sample(input=img_t, grid=bm, align_corners=True)
    return res[0].numpy().transpose(1, 2, 0)


# ─────────────────────────────────────────────
# 3. 形变场可视化（debug 用）
# 3.1 形变场热力图
def visualize_grid(grid, img=None, step=8, save_path=None):
    """
    可视化 opt 输出的形变场热力图，X和Y方向。

    grid     : (n, n, 2) torch tensor 或 numpy，值域 [-1, 1]，最后一维 (x, y)
    img      : 可选，叠加显示的背景图（HxWx3 numpy）
    step     : 网格线采样间隔
    save_path: 若指定则保存图片，否则 plt.show()
    """
    if isinstance(grid, torch.Tensor):
        g = grid.detach().numpy()
    else:
        g = grid.copy()

    n = g.shape[0]
    # 映射到像素坐标（假设显示尺寸为 n x n）
    gx = (g[:, :, 0] + 1) / 2 * (n - 1)  # x 方向
    gy = (g[:, :, 1] + 1) / 2 * (n - 1)  # y 方向

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 子图1：x 分量热力图
    ax = axes[0]
    im = ax.imshow(gx, cmap='jet', origin='upper')
    plt.colorbar(im, ax=ax)
    ax.set_title('Grid X component')

    # 子图2：y 分量热力图
    ax = axes[1]
    im = ax.imshow(gy, cmap='jet', origin='upper')
    plt.colorbar(im, ax=ax)
    ax.set_title('Grid Y component')

    # 子图3：网格线可视化
    # ax = axes[2]
    # if img is not None:
    #     disp = cv2.resize(img, (n, n))
    #     ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    # else:
    #     ax.set_xlim(0, n - 1)
    #     ax.set_ylim(n - 1, 0)
    #     ax.set_facecolor('#f0f0f0')

    # # 画水平网格线（沿列方向）
    # for i in range(0, n, step):
    #     ax.plot(gx[i, :], gy[i, :], 'b-', linewidth=0.6, alpha=0.7)
    # # 画垂直网格线（沿行方向）
    # for j in range(0, n, step):
    #     ax.plot(gx[:, j], gy[:, j], 'r-', linewidth=0.6, alpha=0.7)

    # ax.set_title('Deformation grid (blue=row, red=col)')
    # ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"形变场可视化已保存: {save_path}")
    else:
        plt.show()
    plt.close()

# 3.2 形变场网格线叠加在原图上显示
def visualize_grid_on_image(grid, img, step=8, save_path=None):
    """
    将形变场网格线叠加在原图上显示。

    grid     : (n, n, 2) torch tensor 或 numpy，值域 [-1, 1]，最后一维 (x, y)
    img      : HxWx3 numpy uint8，原始图像
    step     : 网格线采样间隔
    save_path: 若指定则保存图片，否则 plt.show()
    """
    if isinstance(grid, torch.Tensor):
        g = grid.detach().numpy()
    else:
        g = grid.copy()

    n = g.shape[0]
    H, W = img.shape[:2]

    # 将 grid 坐标从 [-1, 1] 映射到原图像素坐标
    # grid 的 x 分量映射到列方向 [0, W-1]，y 分量映射到行方向 [0, H-1]
    gx = (g[:, :, 0] + 1) / 2 * (W - 1)  # x → 列
    gy = (g[:, :, 1] + 1) / 2 * (H - 1)  # y → 行

    # 原图保持原始尺寸显示
    disp_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title('Deformation Grid on Original Image', fontsize=14, fontweight='bold')
    ax.imshow(disp_rgb)
    ax.set_aspect('equal')

    # 画水平网格线（沿行方向）
    for i in range(0, n, step):
        ax.plot(gx[i, :], gy[i, :], 'b-', linewidth=0.5, alpha=0.6)
    # 画垂直网格线（沿列方向）
    for j in range(0, n, step):
        ax.plot(gx[:, j], gy[:, j], 'r-', linewidth=0.5, alpha=0.6)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1, alpha=0.6, label='Horizontal (row)'),
        Line2D([0], [0], color='red', linewidth=1, alpha=0.6, label='Vertical (col)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"形变场叠加原图可视化已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# 3.3 变形网格（grid）与均匀网格（uv）的映射关系可视化
def visualize_grid_mapping(grid1, n=128, sample_step=8, show_arrows=True, 
                          show_grids=True, arrow_alpha=0.3, figsize=(12, 10)):
    """
    可视化变形网格与均匀网格的映射关系
    
    参数:
    - grid1: 形状为(n, n, 2)的变形场，坐标范围可超出[0,1]
    - n: 网格分辨率，默认128
    - sample_step: 采样步长，控制显示的映射点密度，越大越稀疏
    - show_arrows: 是否显示映射箭头
    - show_grids: 是否显示网格线
    - arrow_alpha: 箭头透明度
    - figsize: 图像大小
    """
    
    # 创建均匀网格点
    y_coords = np.linspace(0, 1, n)
    x_coords = np.linspace(0, 1, n)
    uniform_grid_y, uniform_grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
    uniform_grid = np.stack([uniform_grid_x, uniform_grid_y], axis=-1)
    
    # 获取变形网格点（从grid1中）
    deformed_grid = grid1.numpy() if torch.is_tensor(grid1) else grid1
    
    # 根据 deformed_grid 的实际取值范围计算坐标轴边界
    # 同时考虑 uniform_grid [0,1] 和 deformed_grid 的范围，并留一定边距
    margin = 0.05
    x_min = min(0, np.min(deformed_grid[:, :, 0])) - margin
    x_max = max(1, np.max(deformed_grid[:, :, 0])) + margin
    y_min = min(0, np.min(deformed_grid[:, :, 1])) - margin
    y_max = max(1, np.max(deformed_grid[:, :, 1])) + margin
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ============ 左图：网格对比 ============
    ax1 = axes[0]
    ax1.set_title('Grid Comparison (Uniform vs Deformed)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    if show_grids:
        # 绘制均匀网格（蓝色虚线）
        for i in range(0, n, sample_step):
            # 水平线
            ax1.plot(uniform_grid[i, :, 0], uniform_grid[i, :, 1], 
                    'b--', alpha=0.3, linewidth=0.5)
            # 竖直线
            ax1.plot(uniform_grid[:, i, 0], uniform_grid[:, i, 1], 
                    'b--', alpha=0.3, linewidth=0.5)
        
        # 绘制变形网格（红色实线）
        for i in range(0, n, sample_step):
            # 水平线
            ax1.plot(deformed_grid[i, :, 0], deformed_grid[i, :, 1], 
                    'r-', alpha=0.5, linewidth=1)
            # 竖直线
            ax1.plot(deformed_grid[:, i, 0], deformed_grid[:, i, 1], 
                    'r-', alpha=0.5, linewidth=1)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='--', alpha=0.5, label='Uniform Grid'),
        Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Deformed Grid')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # ============ 右图：映射关系 ============
    ax2 = axes[1]
    ax2.set_title('Mapping Relationships', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 采样点
    sample_indices = np.arange(0, n, sample_step)
    
    if show_arrows:
        # 绘制映射箭头
        for i in sample_indices:
            for j in sample_indices:
                start_point = uniform_grid[i, j]
                end_point = deformed_grid[i, j]
                
                # 计算位移向量
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                
                # 只有当位移足够大时才画箭头（避免混乱）
                if np.sqrt(dx**2 + dy**2) > 0.001:
                    ax2.arrow(start_point[0], start_point[1], 
                             dx, dy,
                             head_width=0.008, head_length=0.012,
                             fc='green', ec='green', 
                             alpha=arrow_alpha, linewidth=0.5)
    
    # 绘制均匀网格点（蓝色圆点）
    for i in sample_indices:
        for j in sample_indices:
            ax2.plot(uniform_grid[i, j, 0], uniform_grid[i, j, 1], 
                    'bo', markersize=3, alpha=0.6, label='Uniform' if i==sample_indices[0] and j==sample_indices[0] else '')
    
    # 绘制变形网格点（红色三角）
    for i in sample_indices:
        for j in sample_indices:
            ax2.plot(deformed_grid[i, j, 0], deformed_grid[i, j, 1], 
                    'r^', markersize=3, alpha=0.6, label='Deformed' if i==sample_indices[0] and j==sample_indices[0] else '')
    
    # 添加图例
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig, axes


def visualize_mapping_detail(grid1, center_y=0.5, center_x=0.5, 
                            window_size=0.2, n=128, figsize=(10, 10)):
    """
    可视化局部区域的详细映射关系
    
    参数:
    - grid1: 变形场
    - center_y, center_x: 局部区域的中心坐标（归一化坐标）
    - window_size: 窗口大小（归一化坐标单位）
    - n: 网格分辨率
    """
    
    # 创建均匀网格
    y_coords = np.linspace(0, 1, n)
    x_coords = np.linspace(0, 1, n)
    uniform_grid_y, uniform_grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
    uniform_grid = np.stack([uniform_grid_x, uniform_grid_y], axis=-1)
    
    deformed_grid = grid1.numpy() if torch.is_tensor(grid1) else grid1
    
    # 确定显示区域
    y_min = max(0, center_y - window_size/2)
    y_max = min(1, center_y + window_size/2)
    x_min = max(0, center_x - window_size/2)
    x_max = min(1, center_x + window_size/2)
    
    # 找到对应的网格索引
    y_idx_min = int(y_min * (n-1))
    y_idx_max = int(y_max * (n-1))
    x_idx_min = int(x_min * (n-1))
    x_idx_max = int(x_max * (n-1))
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(f'Detailed Mapping (Region: [{x_min:.2f}, {x_max:.2f}] × [{y_min:.2f}, {y_max:.2f}])', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制均匀网格（蓝色虚线）
    step = max(1, (y_idx_max - y_idx_min) // 10)
    for i in range(y_idx_min, y_idx_max, step):
        ax.plot(uniform_grid[i, x_idx_min:x_idx_max, 0], 
               uniform_grid[i, x_idx_min:x_idx_max, 1], 
               'b--', alpha=0.3, linewidth=0.5)
    for j in range(x_idx_min, x_idx_max, step):
        ax.plot(uniform_grid[y_idx_min:y_idx_max, j, 0], 
               uniform_grid[y_idx_min:y_idx_max, j, 1], 
               'b--', alpha=0.3, linewidth=0.5)
    
    # 绘制变形网格（红色实线）
    for i in range(y_idx_min, y_idx_max, step):
        ax.plot(deformed_grid[i, x_idx_min:x_idx_max, 0], 
               deformed_grid[i, x_idx_min:x_idx_max, 1], 
               'r-', alpha=0.5, linewidth=1)
    for j in range(x_idx_min, x_idx_max, step):
        ax.plot(deformed_grid[y_idx_min:y_idx_max, j, 0], 
               deformed_grid[y_idx_min:y_idx_max, j, 1], 
               'r-', alpha=0.5, linewidth=1)
    
    # 采样更密的点显示映射
    sample_step = max(1, (y_idx_max - y_idx_min) // 8)
    for i in range(y_idx_min, y_idx_max, sample_step):
        for j in range(x_idx_min, x_idx_max, sample_step):
            start_point = uniform_grid[i, j]
            end_point = deformed_grid[i, j]
            
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            
            # 绘制点
            ax.plot(start_point[0], start_point[1], 'bo', markersize=4, alpha=0.7)
            ax.plot(end_point[0], end_point[1], 'r^', markersize=4, alpha=0.7)
            
            # 绘制箭头
            if np.sqrt(dx**2 + dy**2) > 0.0001:
                # 放大箭头以便在局部区域看清
                scale = 1.0
                ax.arrow(start_point[0], start_point[1], 
                        dx * scale, dy * scale,
                        head_width=0.003, head_length=0.005,
                        fc='green', ec='green', 
                        alpha=0.6, linewidth=0.5)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='Uniform Grid Points'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
               markersize=8, label='Deformed Grid Points'),
        Line2D([0], [0], color='green', linewidth=1, label='Mapping Direction')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax


def create_comparison_figure(grid1, n=128):
    """
    创建一个综合对比图，展示不同密度的映射关系
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 不同的采样步长
    sample_steps = [32, 16, 8, 4]
    plot_configs = [
        (axes[0,0], 32, 'Coarse Sampling (step=32)'),
        (axes[0,1], 16, 'Medium Sampling (step=16)'),
        (axes[1,0], 8, 'Fine Sampling (step=8)'),
        (axes[1,1], 4, 'Very Fine Sampling (step=4)'),
    ]
    
    y_coords = np.linspace(0, 1, n)
    x_coords = np.linspace(0, 1, n)
    uniform_grid_y, uniform_grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
    uniform_grid = np.stack([uniform_grid_x, uniform_grid_y], axis=-1)
    
    deformed_grid = grid1.numpy() if torch.is_tensor(grid1) else grid1
    
    for ax, step, title in plot_configs:
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # 绘制网格线
        for i in range(0, n, step):
            ax.plot(deformed_grid[i, :, 0], deformed_grid[i, :, 1], 
                   'gray', alpha=0.3, linewidth=0.5)
            ax.plot(deformed_grid[:, i, 0], deformed_grid[:, i, 1], 
                   'gray', alpha=0.3, linewidth=0.5)
        
        # 绘制映射箭头
        for i in range(0, n, step):
            for j in range(0, n, step):
                start = uniform_grid[i, j]
                end = deformed_grid[i, j]
                dx, dy = end[0] - start[0], end[1] - start[1]
                
                if np.sqrt(dx**2 + dy**2) > 0.001:
                    ax.arrow(start[0], start[1], dx, dy,
                           head_width=0.005, head_length=0.008,
                           fc='blue', ec='blue', alpha=0.5, linewidth=0.3)
    
    # 右下角：变形强度热图
    ax_heatmap = axes[1, 2]
    ax_heatmap.set_title('Deformation Magnitude', fontsize=12, fontweight='bold')
    
    # 计算变形强度
    displacement = deformed_grid - uniform_grid
    magnitude = np.sqrt(displacement[:,:,0]**2 + displacement[:,:,1]**2)
    
    im = ax_heatmap.imshow(magnitude, origin='lower', extent=[0, 1, 0, 1], 
                          cmap='hot', aspect='auto')
    ax_heatmap.set_xlabel('X coordinate')
    ax_heatmap.set_ylabel('Y coordinate')
    plt.colorbar(im, ax=ax_heatmap, label='Displacement Magnitude')
    
    # 左上角可以留空或显示其他信息
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, 'Grid Deformation\nVisualization', 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=axes[0, 2].transAxes)
    
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────────

def dewarp(input_path, boundary_path, textline_path, output_path, debug=False):
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法读取原始图像: {input_path}")

    b_mask = cv2.imread(boundary_path)
    if b_mask is None:
        raise FileNotFoundError(f"无法读取边界特征图: {boundary_path}")

    t_mask = cv2.imread(textline_path)
    if t_mask is None:
        raise FileNotFoundError(f"无法读取文本行特征图: {textline_path}")

    print("提取完整边缘点（保留弯曲特征）...")
    boundary_edges = extract_boundary_edges_full(b_mask, n=128)

    print("提取文本行 ...")
    # t.png → 灰度 → 二值 → line()
    if len(t_mask.shape) == 3:
        t_gray = cv2.cvtColor(t_mask, cv2.COLOR_BGR2GRAY)
    else:
        t_gray = t_mask.copy()
    _, t_bin = cv2.threshold(t_gray, 10, 255, cv2.THRESH_BINARY)

    # 确保尺寸为 512x512（line/line1 内部假设 512x512）
    # 不再缩放到 512x512，保持原始分辨率传入 line()，
    # line() 内部会自动放大到足够分辨率进行 skeletonize，
    # 最终坐标缩放回 512 空间，与下游 line1() / opt() 兼容。

    img_name = os.path.basename(input_path)
    textline_np = extract_line(t_bin)
    # cv2.imshow("t_bin",t_bin)
    # cv2.waitKey()
    print(f"  水平文本行数: {len(textline_np)}")

    # line1 内部会写文件，需要目录存在
    os.makedirs("result/vertical_line", exist_ok=True)
    line1_np = extract_line1(textline_np, img_name)
    print(f"  垂直文本行数: {len(line1_np)}")

    if debug:
        opt_input_path = os.path.splitext(output_path)[0] + "_opt_inputs.png"
        visualize_opt_inputs(boundary_edges, textline_np, line1_np,
                             n=128, save_path=opt_input_path)

    grid = opt(None, textline_np, line1_np, boundary_edges=boundary_edges)
    print("opt 完成")

    if debug:
        debug_path = os.path.splitext(output_path)[0] + "_grid.png"
        visualize_grid(grid, img=img, step=1, save_path=debug_path)

        # 形变场叠加在原图上显示
        grid_on_img_path = os.path.splitext(output_path)[0] + "_grid_on_image.png"
        visualize_grid_on_image(grid, img, step=1, save_path=grid_on_img_path)

        # 可视化网格映射对比，减小sample_step可进行密集采样，查看详细形变
        fig, axes = visualize_grid_mapping(grid, sample_step=4)
        grid_mapping_path = os.path.splitext(output_path)[0] + "_grid_mapping.png"
        fig.savefig(grid_mapping_path, dpi=150)
        print(f"网格映射可视化已保存: {grid_mapping_path}")
        plt.close(fig)

        # 查看特定区域的映射（可调整center_y, center_x, window_size查看不同区域）
        fig, ax = visualize_mapping_detail(grid, center_y=0.5, center_x=0.5, window_size=0.1)
        mapping_detail_path = os.path.splitext(output_path)[0] + "_mapping_detail.png"
        fig.savefig(mapping_detail_path, dpi=150)
        print(f"局部映射详细可视化已保存: {mapping_detail_path}")
        plt.close(fig)

        # 综合对比图（不同采样步长 + 变形强度热图）
        fig = create_comparison_figure(grid, n=128)
        comparison_path = os.path.splitext(output_path)[0] + "_comparison.png"
        fig.savefig(comparison_path, dpi=150)
        print(f"综合对比图已保存: {comparison_path}")
        plt.close(fig)
        
    print("矫正图像 ...")
    result = unwarp(img, grid)
    result = np.clip(result, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"矫正结果已保存: {output_path}")


def get_args():
    parser = argparse.ArgumentParser(description="从特征图像矫正弯曲文档")
    parser.add_argument("--input",    "-i", required=False,  help="待矫正原始图像路径")
    parser.add_argument("--boundary", "-b", default="./DocHV/test5_b.png", help="边界特征图路径")
    parser.add_argument("--textline", "-t", default="./DocHV/test5_t.png",     help="文本行特征图路径")
    parser.add_argument("--output",   "-o", default="./DocHV/dewarped.png", help="输出路径")
    parser.add_argument("--debug",    "-d", action="store_true", help="保存边界采样和形变场可视化图")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.input is None:
        args.input = "./DocHV/test5.jpg"
    args.debug = True
    dewarp(
        input_path=args.input,
        boundary_path=args.boundary,
        textline_path=args.textline,
        output_path=args.output,
        debug=args.debug,
    )
