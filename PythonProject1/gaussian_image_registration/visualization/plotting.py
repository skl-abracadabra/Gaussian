import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict, Any
import math


def plot_image_3d(image: torch.Tensor,
                  title: str = "3D Image",
                  cmap: str = 'viridis',
                  alpha: float = 0.7,
                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    绘制3D图像的体积渲染

    参数:
        image: 3D图像张量 (D, H, W)
        title: 图像标题
        cmap: 颜色映射
        alpha: 透明度
        figsize: 图像大小

    返回:
        plt.Figure: matplotlib图形对象
    """
    # 转换为numpy数组
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    D, H, W = image_np.shape

    # 创建3D图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 创建坐标网格
    x, y, z = np.mgrid[0:W, 0:H, 0:D]

    # 设置阈值，只显示强度较高的体素
    threshold = np.percentile(image_np, 80)
    mask = image_np > threshold

    # 绘制散点图表示体素
    ax.scatter(x[mask], y[mask], z[mask],
               c=image_np[mask],
               cmap=cmap,
               alpha=alpha,
               s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    return fig


def plot_image_slices(image: torch.Tensor,
                      title: str = "Image Slices",
                      slice_indices: Optional[List[int]] = None,
                      cmap: str = 'gray',
                      figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    显示3D图像的三个正交切片（轴向、冠状、矢状）

    参数:
        image: 3D图像张量 (D, H, W)
        title: 图像标题
        slice_indices: 切片索引 [axial, coronal, sagittal]
        cmap: 颜色映射
        figsize: 图像大小

    返回:
        plt.Figure: matplotlib图形对象
    """
    # 转换为numpy数组
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    D, H, W = image_np.shape

    # 默认切片位置（中间切片）
    if slice_indices is None:
        slice_indices = [D // 2, H // 2, W // 2]

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 轴向切片 (XY平面)
    axes[0].imshow(image_np[slice_indices[0], :, :], cmap=cmap)
    axes[0].set_title(f'Axial Slice (Z={slice_indices[0]})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # 冠状切片 (XZ平面)
    axes[1].imshow(image_np[:, slice_indices[1], :], cmap=cmap)
    axes[1].set_title(f'Coronal Slice (Y={slice_indices[1]})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')

    # 矢状切片 (YZ平面)
    axes[2].imshow(image_np[:, :, slice_indices[2]], cmap=cmap)
    axes[2].set_title(f'Sagittal Slice (X={slice_indices[2]})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def plot_registration_result(fixed_image: torch.Tensor,
                             moving_image: torch.Tensor,
                             registered_image: torch.Tensor,
                             titles: List[str] = None,
                             cmap: str = 'gray',
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    显示配准结果对比（固定图像、移动图像、配准后图像）

    参数:
        fixed_image: 固定图像
        moving_image: 移动图像
        registered_image: 配准后的图像
        titles: 各子图标题
        cmap: 颜色映射
        figsize: 图像大小

    返回:
        plt.Figure: matplotlib图形对象
    """
    if titles is None:
        titles = ['Fixed Image', 'Moving Image', 'Registered Image', 'Difference']

    # 转换为numpy数组
    fixed_np = fixed_image.detach().cpu().numpy() if isinstance(fixed_image, torch.Tensor) else fixed_image
    moving_np = moving_image.detach().cpu().numpy() if isinstance(moving_image, torch.Tensor) else moving_image
    registered_np = registered_image.detach().cpu().numpy() if isinstance(registered_image,
                                                                          torch.Tensor) else registered_image

    # 计算差异图像
    difference_np = np.abs(fixed_np - registered_np)

    # 获取中间切片
    slice_idx = fixed_np.shape[0] // 2

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    images = [
        fixed_np[slice_idx, :, :],
        moving_np[slice_idx, :, :],
        registered_np[slice_idx, :, :],
        difference_np[slice_idx, :, :]
    ]

    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Image Registration Results')
    plt.tight_layout()

    return fig


def plot_optimization_progress(loss_history: List[float],
                               parameter_history: Optional[Dict[str, List[float]]] = None,
                               title: str = "Optimization Progress",
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    绘制优化过程监控图（损失曲线和参数变化）

    参数:
        loss_history: 损失历史
        parameter_history: 参数变化历史
        title: 图像标题
        figsize: 图像大小

    返回:
        plt.Figure: matplotlib图形对象
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 绘制损失曲线
    axes[0].plot(loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Convergence')
    axes[0].grid(True)

    # 绘制参数变化（如果提供）
    if parameter_history is not None:
        for param_name, param_values in parameter_history.items():
            axes[1].plot(param_values, label=param_name)

        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Parameter Value')
        axes[1].set_title('Parameter Evolution')
        axes[1].legend()
        axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def plot_transformation_field(transformation: torch.Tensor,
                              title: str = "Transformation Field",
                              slice_index: int = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    绘制变换场（位移场或变形场）

    参数:
        transformation: 变换场张量 (3, D, H, W) 或 (2, H, W)
        title: 图像标题
        slice_index: 切片索引（3D时使用）
        figsize: 图像大小

    返回:
        plt.Figure: matplotlib图形对象
    """
    if isinstance(transformation, torch.Tensor):
        transform_np = transformation.detach().cpu().numpy()
    else:
        transform_np = transformation

    fig = plt.figure(figsize=figsize)

    if transform_np.ndim == 4:  # 3D变换场
        if slice_index is None:
            slice_index = transform_np.shape[1] // 2

        # 获取指定切片的变换场
        u = transform_np[0, slice_index, :, :]  # X方向位移
        v = transform_np[1, slice_index, :, :]  # Y方向位移

        # 创建网格
        Y, X = np.mgrid[0:u.shape[0], 0:u.shape[1]]

        plt.quiver(X, Y, u, v, scale=1.0, scale_units='xy')
        plt.title(f'{title} (Slice Z={slice_index})')

    else:  # 2D变换场
        u = transform_np[0, :, :]  # X方向位移
        v = transform_np[1, :, :]  # Y方向位移

        # 创建网格
        Y, X = np.mgrid[0:u.shape[0], 0:u.shape[1]]

        plt.quiver(X, Y, u, v, scale=1.0, scale_units='xy')
        plt.title(title)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    return fig


def plot_intensity_distribution(fixed_image: torch.Tensor,
                                moving_image: torch.Tensor,
                                registered_image: torch.Tensor,
                                title: str = "Intensity Distribution",
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制图像强度分布直方图对比

    参数:
        fixed_image: 固定图像
        moving_image: 移动图像
        registered_image: 配准后的图像
        title: 图像标题
        figsize: 图像大小

    返回:
        plt.Figure: matplotlib图形对象
    """
    # 转换为numpy数组并展平
    fixed_flat = fixed_image.detach().cpu().numpy().flatten() if isinstance(fixed_image,
                                                                            torch.Tensor) else fixed_image.flatten()
    moving_flat = moving_image.detach().cpu().numpy().flatten() if isinstance(moving_image,
                                                                              torch.Tensor) else moving_image.flatten()
    registered_flat = registered_image.detach().cpu().numpy().flatten() if isinstance(registered_image,
                                                                                      torch.Tensor) else registered_image.flatten()

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制直方图
    ax.hist(fixed_flat, bins=50, alpha=0.7, label='Fixed', density=True)
    ax.hist(moving_flat, bins=50, alpha=0.7, label='Moving', density=True)
    ax.hist(registered_flat, bins=50, alpha=0.7, label='Registered', density=True)

    ax.set_xlabel('Intensity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_multimodal_comparison(images: List[torch.Tensor],
                               titles: List[str],
                               overlay: bool = False,
                               cmap: str = 'jet',
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    多模态图像对比显示

    参数:
        images: 图像列表
        titles: 标题列表
        overlay: 是否叠加显示
        cmap: 颜色映射
        figsize: 图像大小

    返回:
        plt.Figure: matplotlib图形对象
    """
    n_images = len(images)

    if overlay and n_images == 2:
        # 叠加显示两个图像
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 转换为numpy数组
        img1 = images[0].detach().cpu().numpy() if isinstance(images[0], torch.Tensor) else images[0]
        img2 = images[1].detach().cpu().numpy() if isinstance(images[1], torch.Tensor) else images[1]

        slice_idx = img1.shape[0] // 2

        # 原始图像
        axes[0].imshow(img1[slice_idx, :, :], cmap='gray')
        axes[0].set_title(titles[0])
        axes[0].axis('off')

        # 叠加图像
        axes[1].imshow(img1[slice_idx, :, :], cmap='gray')
        im = axes[1].imshow(img2[slice_idx, :, :], cmap=cmap, alpha=0.5)
        axes[1].set_title(f'{titles[0]} + {titles[1]}')
        axes[1].axis('off')

        plt.colorbar(im, ax=axes[1])

    else:
        # 并排显示多个图像
        fig, axes = plt.subplots(1, n_images, figsize=figsize)

        for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
            img_np = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
            slice_idx = img_np.shape[0] // 2

            im = ax.imshow(img_np[slice_idx, :, :], cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax)

    plt.suptitle('Multimodal Image Comparison')
    plt.tight_layout()

    return fig


def create_registration_animation(fixed_image: torch.Tensor,
                                  moving_images: List[torch.Tensor],
                                  titles: List[str],
                                  interval: int = 200,
                                  figsize: Tuple[int, int] = (8, 6)) -> animation.FuncAnimation:
    """
    创建配准过程动画

    参数:
        fixed_image: 固定图像
        moving_images: 移动图像序列（配准过程中的中间结果）
        titles: 动画帧标题
        interval: 帧间隔（毫秒）
        figsize: 图像大小

    返回:
        animation.FuncAnimation: 动画对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 转换为numpy数组
    fixed_np = fixed_image.detach().cpu().numpy() if isinstance(fixed_image, torch.Tensor) else fixed_image
    moving_sequence = [img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img for img in moving_images]

    slice_idx = fixed_np.shape[0] // 2

    def animate(frame):
        ax.clear()

        # 显示固定图像作为背景
        ax.imshow(fixed_np[slice_idx, :, :], cmap='gray', alpha=0.7)

        # 显示当前帧的移动图像
        im = ax.imshow(moving_sequence[frame][slice_idx, :, :], cmap='jet', alpha=0.5)

        ax.set_title(titles[frame] if frame < len(titles) else f'Frame {frame}')
        ax.axis('off')

        return im,

    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=len(moving_sequence),
                                   interval=interval, blit=True)

    plt.tight_layout()

    return anim


# 工具函数
def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """
    保存图形到文件

    参数:
        fig: matplotlib图形对象
        filename: 文件名
        dpi: 分辨率
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {filename}")


def set_style(style: str = 'default') -> None:
    """
    设置绘图样式

    参数:
        style: 样式名称 ('default', 'dark', 'scientific')
    """
    if style == 'dark':
        plt.style.use('dark_background')
    elif style == 'scientific':
        plt.style.use('seaborn-whitegrid')
    else:
        plt.style.use('default')