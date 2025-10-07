import torch
from .._01_core.registration import GaussianDIR
from .._02_io.file_handlers import create_simulated_brain
from .._04_visualization.plotting import plot_registration_results


def basic_demo():
    """基础演示"""
    print("高斯配准基础演示")

    # 创建测试数据
    fixed = create_simulated_brain((64, 64, 64))
    moving = torch.roll(fixed, shifts=(5, 3, 2), dims=(0, 1, 2))

    # 初始化配准器
    registrar = GaussianDIR(fixed.shape, num_primitives=50)

    # 运行配准
    dvf = registrar.optimize(fixed, moving, iterations=20)
    warped = registrar.warp_image(moving, dvf)

    # 显示结果
    plot_registration_results(fixed, moving, warped)

    return dvf, warped