## 三个组件的完整数据流

### 一、`geom_textures`（带纹理内容的文档图像）

**数据来源文件夹：**
- `{path}/{name}/img_geom/` → 几何体图像（弯曲空白文档照片）
- `{path}/textures/` → 纹理图像（平坦的文档内容页面）
- `{path}/{name}/uvmap/` → UV 映射场（.mat 文件）

**经过函数：`apply_texture(geom_path, texture, uv_path)`**（第100-155行）

处理流程：

```
纹理(texture)  ──→  F.grid_sample(texture, uvmap)  ──→  warped_texture
                                                          │
几何体(geom)    ─────────────────────────────────────→     │
                                                          │
         warped_texture * 0.75 * mask_border + geom * (1 - 0.75 * mask_border)
                                                          │
                                                          ↓
                                                     geom_textures
```

具体步骤：
1. 读取几何体 `geom = cv2.imread(geom_path)` 和 UV 映射 `uvmap`
2. 将纹理通过 UV 映射变形：`warped_texture = F.grid_sample(texture, uvmap)` — UV 映射告诉每个像素"去纹理图的哪里取值"
3. 生成文档边缘蒙版 `mask`：灰度值为 0.5 的像素视为背景，`mask = 1 - all(warped_texture == 0.5)`
4. 生成侵蚀蒙版 `mask_small = binary_erosion(mask)` — 去掉边缘 1 像素避免伪影
5. 模糊纹理后与几何体相乘：`geom_textures = (blur_texture * geom / 255) * 0.75 * mask + geom * (1 - 0.75 * mask)`

---

### 二、`background`（可能经过颜色迁移的背景图）

**数据来源文件夹：**
- `{path}/backgrounds/` → 原始背景图片
- `{path}/{name}/img_geom/` → 几何体（用于计算颜色信息）
- `{path}/{name}/uvmap/` → UV 映射（用于颜色迁移时变形颜色网格）
- 程序内部生成的 `create_color_grid()` → 辅助颜色采样网格

**经过函数：`apply_background(geom_path, background_path, uv_path, geom_textures, mask_small, color_transfer)`**（第158-208行）

处理流程：

```
背景(backgrounds/) ──────────────────────────────────→ background
                                                              │
几何体(img_geom/) ──→ 采样前景均值 ──→ color_transfer_fn() ──┤
UV映射(uvmap/)   ──→ 变形color_grid ──→ 采样文档色调 ────────┘
                                                              │
                                                         颜色迁移后的background
                                                         （或原始background）
```

颜色迁移步骤（如果启用）：
1. `create_color_grid()` 生成 3×3 彩色方块图像
2. `apply_texture(geom_path, color_grid, uv_path)` 将颜色网格通过 UV 映射贴到弯曲文档上
3. `color_transfer_fn(geom, background, mask_small, warped_color_grid)`：
   - 从文档区域采样每通道均值偏移 → 微调背景色偏
   - 从颜色网格采样全局亮度 → 调整背景亮度

---

### 三、`mask`（文档与背景的混合蒙版）

**数据来源文件夹：**
- `{path}/{name}/uvmap/` → UV 映射场

**经过函数：`apply_background()` 内部**（第176-189行）

处理流程：

```
UV映射(uvmap/) ──→ F.grid_sample(white_image, uvmap) ──→ warped_white
                                                            │
                         gaussian_filter(sigma=0.75)  ←────┘
                                                            │
                         grey_dilation(3×3) × 2       ←────┘
                                                            │
                                                            ↓
                                                           mask
```

具体步骤：
1. 创建一个 1000×1000 的白色图像，**四周边缘设为0**（`white[:, :, 0]=0, white[:, :, -1]=0, ...`）
2. 通过 UV 映射将白色图像变形到文档的弯曲形状上：`warped_white = F.grid_sample(white, uvmap)`
3. 高斯模糊平滑边缘：`gaussian_filter(sigma=0.75)`
4. 灰度膨胀两次（`grey_dilation`）扩大蒙版区域
5. 结果 `mask` 值在 [0,1] 之间，文档中心为1（显示文档），边缘渐变为0（过渡到背景）

---

### 最终合成

```python
# create_final.py 第206行
result = geom_textures * mask + (1 - mask) * background
```

| 组件 | 数据来源 | 核心函数 |
|------|---------|---------|
| `geom_textures` | `img_geom/` + `textures/` + `uvmap/` | `apply_texture()` |
| `background` | `backgrounds/` + `img_geom/` + `uvmap/` + `color_grid` | `apply_background()` → `color_transfer_fn()` |
| `mask` | `uvmap/`（+ 内部创建的 white image） | `apply_background()` 内部，经 `F.grid_sample` + `gaussian_filter` + `grey_dilation` |