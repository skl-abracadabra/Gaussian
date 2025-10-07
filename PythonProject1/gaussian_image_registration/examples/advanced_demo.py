import torch
from .._01_core.registration import GaussianDIR
from .._02_io.image_loaders import load_image
from .._04_visualization.plotting import plot_registration_results, plot_dvf


def advanced_demo(fixed_path, moving_path):
    """高级演示（使用真实数据）"""
    print("高斯配准高级演示")

    # 加载图像
    fixed = load_image(fixed_path)
    moving = load_image(moving_path)

    if fixed is None or moving is None:
        print("图像加载失败")
        return None, None

    # 初始化配准器
    registrar = GaussianDIR(fixed.shape)

    # 运行配准
    dvf = registrar.optimize(fixed, moving, iterations=50)
    warped = registrar.warp_image(moving, dvf)

    # 显示结果
    plot_registration_results(fixed, moving, warped)
    plot_dvf(dvf)

    return dvf, warped