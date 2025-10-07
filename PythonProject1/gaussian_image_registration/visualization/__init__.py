"""
高斯图像配准可视化模块

提供图像配准过程中的各种可视化功能，包括：
- 图像显示和对比
- 配准过程可视化
- 变换效果展示
- 优化过程监控
"""

from .plotting import (
    plot_image_3d,
    plot_image_slices,
    plot_registration_result,
    plot_optimization_progress,
    plot_transformation_field,
    plot_intensity_distribution,
    plot_multimodal_comparison,
    create_registration_animation
)

__all__ = [
    'plot_image_3d',
    'plot_image_slices',
    'plot_registration_result',
    'plot_optimization_progress',
    'plot_transformation_field',
    'plot_intensity_distribution',
    'plot_multimodal_comparison',
    'create_registration_animation'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "高斯图像配准团队"
__description__ = "高斯图像配准可视化工具"