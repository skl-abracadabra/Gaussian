"""
高斯图像配准工具模块
提供优化算法、图像变换和数学工具函数
"""

from .optimization import (
    ncc_loss,
    gradient_descent,
    adam_optimizer,
    early_stopping,
    compute_gradient_norm,
    learning_rate_scheduler
)

from .transformations import (
    affine_transform,
    resize_image,
    rotate_image,
    translate_image,
    bilinear_interpolation_3d,
    create_rotation_matrix,
    create_affine_matrix
)

__all__ = [
    # 优化函数
    'ncc_loss',
    'gradient_descent',
    'adam_optimizer',
    'early_stopping',
    'compute_gradient_norm',
    'learning_rate_scheduler',

    # 变换函数
    'affine_transform',
    'resize_image',
    'rotate_image',
    'translate_image',
    'bilinear_interpolation_3d',
    'create_rotation_matrix',
    'create_affine_matrix'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "高斯配准团队"
__description__ = "高斯图像配准工具函数库"