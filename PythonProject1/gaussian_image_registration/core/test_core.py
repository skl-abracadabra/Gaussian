import sys
import os
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from gaussian_image_registration.core.primitives import GaussianPrimitive
from gaussian_image_registration.core.registration import GaussianDIR


def test_gaussian_primitives():
    """测试高斯基元类"""
    print("测试高斯基元...")

    # 创建测试高斯参数
    mu = torch.tensor([10.0, 20.0, 30.0])
    scale = torch.tensor([0.1, 0.2, 0.3])
    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 单位四元数
    translation = torch.tensor([1.0, 2.0, 3.0])

    # 创建高斯基元
    gaussian = GaussianPrimitive(mu, scale, rotation, translation)

    # 测试协方差矩阵计算
    cov_matrix = gaussian.get_covariance_matrix()
    print(f"协方差矩阵形状: {cov_matrix.shape}")
    print(f"协方差矩阵:\n{cov_matrix}")

    # 测试可训练参数
    params = gaussian.get_trainable_parameters()
    print(f"可训练参数数量: {len(params)}")

    # 测试参数梯度
    for param in params:
        print(f"参数形状: {param.shape}, 需要梯度: {param.requires_grad}")

    print("高斯基元测试通过!\n")


def test_gaussian_registration():
    """测试高斯配准器"""
    print("测试高斯配准器...")

    # 创建小尺寸测试图像
    image_shape = (10, 10, 10)
    registrar = GaussianDIR(image_shape, num_primitives=8, k_neighbors=2)

    print(f"配准器初始化: {registrar}")

    # 创建测试图像
    fixed = torch.randn(image_shape)
    moving = torch.roll(fixed, shifts=(2, 1, 1), dims=(0, 1, 2))

    print(f"固定图像形状: {fixed.shape}, 强度范围: {fixed.min():.2f} - {fixed.max():.2f}")
    print(f"移动图像形状: {moving.shape}, 强度范围: {moving.min():.2f} - {moving.max():.2f}")

    # 测试DVF计算
    dvf = registrar.compute_dvf(batch_size=100)
    print(f"DVF形状: {dvf.shape}")

    # 测试图像变形
    warped = registrar.warp_image(moving, dvf)
    print(f"变形后图像形状: {warped.shape}")

    # 测试NCC计算
    ncc_value = registrar.ncc(fixed, warped)
    print(f"初始NCC: {ncc_value:.4f}")

    print("高斯配准器测试通过!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("高斯配准核心模块测试")
    print("=" * 50)

    test_gaussian_primitives()
    test_gaussian_registration()

    print("所有核心模块测试完成!")