import torch
import numpy as np
import math
from typing import List, Tuple, Union, Optional


def affine_transform(image: torch.Tensor,
                     matrix: torch.Tensor,
                     translation: List[float],
                     mode: str = 'bilinear') -> torch.Tensor:
    """
    仿射变换 - 使用反向映射确保完整覆盖（修正版）

    参数:
        image: 输入图像张量 (D, H, W)
        matrix: 3x3变换矩阵
        translation: 平移向量 [tx, ty, tz]
        mode: 插值模式 ('nearest', 'bilinear')

    返回:
        torch.Tensor: 变换后的图像
    """
    # 参数验证
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be a 3x3 tensor")
    if len(translation) != 3:
        raise ValueError("translation must be a list of 3 elements")
    if mode not in ['nearest', 'bilinear']:
        raise ValueError("mode must be 'nearest' or 'bilinear'")

    D, H, W = image.shape

    # 创建输出图像
    transformed = torch.zeros_like(image)

    # 将平移转换为张量（确保设备一致性）
    translation_tensor = torch.tensor(translation, dtype=torch.float32, device=image.device)

    # 计算变换矩阵的逆（用于反向映射）
    try:
        inv_matrix = torch.inverse(matrix)
    except RuntimeError:
        raise ValueError("变换矩阵不可逆，无法进行仿射变换")

    # 使用反向映射：遍历输出图像的每个体素
    for z_out in range(D):
        for y_out in range(H):
            for x_out in range(W):
                # 输出坐标
                out_coord = torch.tensor([x_out, y_out, z_out], dtype=torch.float32, device=image.device)

                # 计算输入坐标：input_coord = inv_matrix * (out_coord - translation)
                in_coord = inv_matrix @ (out_coord - translation_tensor)
                in_x, in_y, in_z = in_coord[0].item(), in_coord[1].item(), in_coord[2].item()

                # 边界检查
                if (0 <= in_x < W and 0 <= in_y < H and 0 <= in_z < D):
                    if mode == 'nearest':
                        # 最近邻插值
                        in_x_int = int(round(in_x))
                        in_y_int = int(round(in_y))
                        in_z_int = int(round(in_z))

                        # 安全钳制坐标到有效范围
                        in_x_int = max(0, min(W - 1, in_x_int))
                        in_y_int = max(0, min(H - 1, in_y_int))
                        in_z_int = max(0, min(D - 1, in_z_int))

                        transformed[z_out, y_out, x_out] = image[in_z_int, in_y_int, in_x_int]
                    elif mode == 'bilinear':
                        # 双线性插值
                        transformed[z_out, y_out, x_out] = bilinear_interpolation_3d(
                            image, in_x, in_y, in_z
                        )

    return transformed


def affine_transform_vectorized(image: torch.Tensor,
                                matrix: torch.Tensor,
                                translation: Union[List[float], Tuple[float, float, float]],
                                mode: str = 'bilinear') -> torch.Tensor:
    """
    向量化版本的仿射变换（性能优化）

    参数:
        image: 输入图像张量 (D, H, W)
        matrix: 3x3变换矩阵
        translation: 平移向量 [tx, ty, tz]
        mode: 插值模式 ('nearest', 'bilinear')

    返回:
        torch.Tensor: 变换后的图像
    """
    # 参数验证
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be a 3x3 tensor")
    if len(translation) != 3:
        raise ValueError("translation must be a list/tuple of 3 elements")

    D, H, W = image.shape

    # 创建坐标网格
    z_coords, y_coords, x_coords = torch.meshgrid(
        torch.arange(D, device=image.device),
        torch.arange(H, device=image.device),
        torch.arange(W, device=image.device),
        indexing='ij'
    )

    # 展平坐标
    coords = torch.stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()], dim=1).float()

    # 应用变换
    translation_tensor = torch.tensor(translation, dtype=torch.float32, device=image.device)
    new_coords = (matrix @ coords.T).T + translation_tensor

    # 重塑回图像形状
    new_coords = new_coords.reshape(D, H, W, 3)

    # 使用grid_sample进行高效插值
    image_4d = image.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度 (1, 1, D, H, W)
    grid = new_coords.unsqueeze(0)  # 添加批次维度 (1, D, H, W, 3)

    # 归一化坐标到[-1, 1]范围
    grid_normalized = torch.stack([
        2.0 * grid[..., 0] / (W - 1) - 1,
        2.0 * grid[..., 1] / (H - 1) - 1,
        2.0 * grid[..., 2] / (D - 1) - 1
    ], dim=-1)

    # 应用网格采样
    warped = torch.nn.functional.grid_sample(
        image_4d, grid_normalized,
        mode=mode, align_corners=True, padding_mode='zeros'
    )

    return warped.squeeze(0).squeeze(0)


def bilinear_interpolation_3d(image: torch.Tensor, x: float, y: float, z: float) -> float:
    """
    3D双线性插值（修正版）

    参数:
        image: 输入图像张量
        x, y, z: 浮点坐标

    返回:
        float: 插值结果
    """
    D, H, W = image.shape

    # 计算整数坐标
    x0, y0, z0 = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    x1, y1, z1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1), min(z0 + 1, D - 1)

    # 计算权重
    xd = x - x0
    yd = y - y0
    zd = z - z0

    # 获取八个角点的值
    c000 = image[z0, y0, x0].item()
    c001 = image[z0, y0, x1].item()
    c010 = image[z0, y1, x0].item()
    c011 = image[z0, y1, x1].item()
    c100 = image[z1, y0, x0].item()
    c101 = image[z1, y0, x1].item()
    c110 = image[z1, y1, x0].item()
    c111 = image[z1, y1, x1].item()  # 修正索引错误

    # 三线性插值
    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    return c0 * (1 - zd) + c1 * zd


def resize_image(image: torch.Tensor, new_shape: Tuple[int, int, int], mode: str = 'trilinear') -> torch.Tensor:
    """
    调整图像尺寸

    参数:
        image: 输入图像张量
        new_shape: 新形状 (D, H, W)
        mode: 插值模式 ('nearest', 'trilinear')

    返回:
        torch.Tensor: 调整尺寸后的图像
    """
    # 参数验证
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if len(new_shape) != 3:
        raise ValueError("new_shape must be a tuple of 3 integers")
    if mode not in ['nearest', 'trilinear']:
        raise ValueError("mode must be 'nearest' or 'trilinear'")

    # 添加批次和通道维度
    image_4d = image.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # 使用PyTorch插值函数
    resized = torch.nn.functional.interpolate(
        image_4d,
        size=new_shape,
        mode=mode,
        align_corners=True
    )

    # 移除批次和通道维度
    return resized.squeeze(0).squeeze(0)


def rotate_image(image: torch.Tensor, angles: List[float], center: Optional[List[float]] = None) -> torch.Tensor:
    """
    旋转图像（修正版）- 添加旋转中心处理

    参数:
        image: 输入图像张量
        angles: 旋转角度 [rx, ry, rz]（弧度）
        center: 旋转中心，默认为图像物理中心

    返回:
        torch.Tensor: 旋转后的图像
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if len(angles) != 3:
        raise ValueError("angles must be a list of 3 values")

    D, H, W = image.shape

    # 默认旋转中心 - 使用图像物理中心而非索引中心
    if center is None:
        # 使用图像物理中心 (x_center, y_center, z_center)
        center = [(W-1)/2, (H-1)/2, (D-1)/2]
    elif len(center) != 3:
        raise ValueError("center must be a list of 3 values")

    # 创建旋转矩阵
    R = create_rotation_matrix(angles)

    # 修正：正确的旋转中心处理
    # 计算平移向量：先平移到中心，旋转后再平移回来
    cx, cy, cz = center
    translation = [
        cx - (R[0, 0] * cx + R[0, 1] * cy + R[0, 2] * cz),
        cy - (R[1, 0] * cx + R[1, 1] * cy + R[1, 2] * cz),
        cz - (R[2, 0] * cx + R[2, 1] * cy + R[2, 2] * cz)
    ]

    # 应用旋转
    return affine_transform(image, R, translation)


def translate_image(image: torch.Tensor, translation: List[float]) -> torch.Tensor:
    """
    平移图像 - 修复版本（正确坐标计算）

    参数:
        image: 输入图像张量
        translation: 平移向量 [tx, ty, tz]

    返回:
        torch.Tensor: 平移后的图像
    """
    # 参数验证
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if len(translation) != 3:
        raise ValueError("translation must be a list of 3 values")

    # 单位矩阵作为变换矩阵
    identity_matrix = torch.eye(3, device=image.device)

    return affine_transform(image, identity_matrix, translation)


def create_rotation_matrix(angles: List[float]) -> torch.Tensor:
    """
    从欧拉角创建旋转矩阵 - 修复版本（使用X-Y-Z顺序）

    参数:
        angles: 旋转角度 [rx, ry, rz]（弧度）

    返回:
        torch.Tensor: 3x3旋转矩阵
    """
    if len(angles) != 3:
        raise ValueError("angles must be a list of 3 values")

    rx, ry, rz = angles

    # 创建单个轴的旋转矩阵
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ], dtype=torch.float32)

    Ry = torch.tensor([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ], dtype=torch.float32)

    Rz = torch.tensor([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # 组合旋转矩阵 (X-Y-Z顺序)
    return Rz @ Ry @ Rx


def create_affine_matrix(rotation: torch.Tensor,
                         scale: List[float],
                         translation: List[float]) -> torch.Tensor:
    """
    创建仿射变换矩阵

    参数:
        rotation: 3x3旋转矩阵
        scale: 缩放向量 [sx, sy, sz]
        translation: 平移向量 [tx, ty, tz]

    返回:
        torch.Tensor: 4x4仿射变换矩阵
    """
    # 参数验证
    if not isinstance(rotation, torch.Tensor) or rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 tensor")
    if len(scale) != 3:
        raise ValueError("scale must be a list of 3 values")
    if len(translation) != 3:
        raise ValueError("translation must be a list of 3 values")

    # 创建缩放矩阵
    S = torch.diag(torch.tensor(scale, dtype=torch.float32))

    # 组合旋转和缩放
    RS = rotation @ S

    # 创建4x4仿射矩阵
    affine = torch.eye(4, dtype=torch.float32)
    affine[:3, :3] = RS
    affine[:3, 3] = torch.tensor(translation, dtype=torch.float32)

    return affine