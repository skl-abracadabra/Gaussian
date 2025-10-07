import torch
import numpy as np
import math
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from gaussian_image_registration.utils.optimization import (
    ncc_loss, gradient_descent, adam_optimizer,
    early_stopping, compute_gradient_norm, learning_rate_scheduler
)
from gaussian_image_registration.utils.transformations import (
    affine_transform, resize_image, rotate_image, translate_image,
    bilinear_interpolation_3d, create_rotation_matrix, create_affine_matrix
)


def test_ncc_loss():
    """测试归一化互相关损失函数"""
    print("=" * 60)
    print("测试归一化互相关损失函数 (NCC)")
    print("=" * 60)

    # 测试1: 相同图像应该有高相似性（低损失）
    print("\n1. 测试相同图像的NCC损失")
    image = torch.randn(8, 8, 8)
    loss_same = ncc_loss(image, image)
    print(f"相同图像的NCC损失: {loss_same.item():.6f}")
    assert loss_same.item() < -0.9, "相同图像的NCC应该接近-1（高相似性）"

    # 测试2: 随机图像应该有低相似性（高损失）
    print("\n2. 测试随机图像的NCC损失")
    random_image = torch.randn(8, 8, 8)
    loss_random = ncc_loss(image, random_image)
    print(f"随机图像的NCC损失: {loss_random.item():.6f}")
    assert loss_random.item() > loss_same.item(), "随机图像应该有更高的损失"

    # 测试3: 平移图像的相似性
    print("\n3. 测试平移图像的NCC损失")
    shifted_image = torch.roll(image, shifts=(2, 1, 0), dims=(0, 1, 2))
    loss_shifted = ncc_loss(image, shifted_image)
    print(f"平移图像的NCC损失: {loss_shifted.item():.6f}")

    # 测试4: 图像形状不匹配应该报错
    print("\n4. 测试图像形状验证")
    try:
        wrong_size_image = torch.randn(6, 6, 6)
        ncc_loss(image, wrong_size_image)
        assert False, "应该抛出形状不匹配错误"
    except ValueError as e:
        print(f"形状验证正确: {e}")

    print("\n✓ NCC损失函数所有测试通过!\n")


def test_gradient_descent():
    """测试梯度下降优化算法"""
    print("=" * 60)
    print("测试梯度下降优化算法")
    print("=" * 60)

    # 测试1: 简单二次函数优化
    print("\n1. 测试二次函数优化 (f(x) = x^2)")
    x = torch.tensor([3.0], requires_grad=True)

    def quadratic_loss():
        return x ** 2

    history = gradient_descent([x], quadratic_loss, lr=0.1, max_iter=50)

    print(f"初始值: 3.0")
    print(f"优化结果: {x.item():.6f}")
    print(f"最终损失: {quadratic_loss().item():.6f}")
    print(f"迭代次数: {len(history['loss'])}")

    assert abs(x.item()) < 0.1, "应该收敛到接近0"
    assert history['loss'][-1] < history['loss'][0], "损失应该减少"

    # 测试2: 多参数优化
    print("\n2. 测试多参数优化")
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([-1.0], requires_grad=True)

    def multi_param_loss():
        return (a - 1.0) ** 2 + (b + 2.0) ** 2

    history_multi = gradient_descent([a, b], multi_param_loss, lr=0.05, max_iter=100)

    print(f"参数a结果: {a.item():.6f} (期望: 1.0)")
    print(f"参数b结果: {b.item():.6f} (期望: -2.0)")
    print(f"最终损失: {multi_param_loss().item():.6f}")

    assert abs(a.item() - 1.0) < 0.1, "参数a应该收敛到1.0"
    assert abs(b.item() + 2.0) < 0.1, "参数b应该收敛到-2.0"

    print("\n✓ 梯度下降算法所有测试通过!\n")


def test_adam_optimizer():
    """测试Adam优化算法"""
    print("=" * 60)
    print("测试Adam优化算法")
    print("=" * 60)

    # 测试1: 简单函数优化
    print("\n1. 测试Adam优化器")
    x = torch.tensor([5.0], requires_grad=True)

    def test_loss():
        return (x - 2.0) ** 2

    history = adam_optimizer([x], test_loss, lr=0.1, max_iter=200)

    print(f"初始值: 5.0")
    print(f"优化结果: {x.item():.6f}")
    print(f"最终损失: {test_loss().item():.6f}")
    print(f"迭代次数: {len(history['loss'])}")

    assert abs(x.item() - 2.0) < 0.1, "应该收敛到接近2.0"
    assert history['loss'][-1] < history['loss'][0], "损失应该减少"

    # 测试2: 学习率调度
    print("\n2. 测试学习率调度器")
    initial_lr = 0.1
    lr_100 = learning_rate_scheduler(initial_lr, 100, decay_rate=0.9, decay_step=50)
    lr_200 = learning_rate_scheduler(initial_lr, 200, decay_rate=0.9, decay_step=50)

    print(f"初始学习率: {initial_lr}")
    print(f"迭代100次后学习率: {lr_100:.6f}")
    print(f"迭代200次后学习率: {lr_200:.6f}")

    assert lr_200 < lr_100 < initial_lr, "学习率应该随时间衰减"

    print("\n✓ Adam优化器所有测试通过!\n")


def test_early_stopping():
    """测试早期停止算法"""
    print("=" * 60)
    print("测试早期停止算法")
    print("=" * 60)

    # 测试1: 应该停止的情况（损失不再改善）
    print("\n1. 测试应该停止的情况")
    losses_no_improve = [0.5, 0.4, 0.35, 0.34, 0.33, 0.32, 0.31, 0.305, 0.303, 0.302]
    should_stop = early_stopping(losses_no_improve, patience=3, min_delta=0.01)

    print(f"损失历史: {losses_no_improve}")
    print(f"是否应该停止: {should_stop}")
    assert should_stop == True, "应该触发早期停止"

    # 测试2: 不应该停止的情况（损失仍在改善）
    print("\n2. 测试不应该停止的情况")
    losses_improving = [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1, 0.09]
    should_not_stop = early_stopping(losses_improving, patience=3, min_delta=0.01)

    print(f"损失历史: {losses_improving}")
    print(f"是否应该停止: {should_not_stop}")
    assert should_not_stop == False, "不应该触发早期停止"

    # 测试3: 梯度范数计算
    print("\n3. 测试梯度范数计算")
    x = torch.tensor([1.0], requires_grad=True)
    y = x ** 2
    y.backward()

    grad_norm = compute_gradient_norm([x])
    print(f"梯度范数: {grad_norm:.6f}")
    assert grad_norm > 0, "梯度范数应该大于0"

    print("\n✓ 早期停止算法所有测试通过!\n")


def test_affine_transform():
    """测试仿射变换"""
    print("=" * 60)
    print("测试仿射变换")
    print("=" * 60)

    # 创建测试图像（简单的梯度图像）
    image = torch.zeros(5, 5, 5)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                image[i, j, k] = i + j + k

    # 测试1: 平移变换
    print("\n1. 测试平移变换")
    translation = [1.0, 0.5, 0.0]
    identity_matrix = torch.eye(3)

    translated = affine_transform(image, identity_matrix, translation, mode='nearest')

    print(f"原始图像形状: {image.shape}")
    print(f"平移后图像形状: {translated.shape}")
    print(f"原始图像范围: {image.min():.1f} - {image.max():.1f}")
    print(f"平移后图像范围: {translated.min():.1f} - {translated.max():.1f}")

    assert translated.shape == image.shape, "变换后图像形状应该不变"

    # 测试2: 旋转变换
    print("\n2. 测试旋转变换")
    rotation_matrix = create_rotation_matrix([0, 0, math.pi / 4])  # 绕Z轴旋转45度

    rotated = affine_transform(image, rotation_matrix, [0, 0, 0], mode='nearest')

    print(f"旋转后图像范围: {rotated.min():.1f} - {rotated.max():.1f}")
    assert not torch.allclose(image, rotated), "旋转后图像应该不同"

    # 测试3: 双线性插值
    print("\n3. 测试双线性插值")
    test_value = bilinear_interpolation_3d(image, 2.5, 2.5, 2.5)
    print(f"插值点(2.5,2.5,2.5)的值: {test_value:.3f}")
    assert 7.0 <= test_value <= 8.0, "插值应该在合理范围内"

    print("\n✓ 仿射变换所有测试通过!\n")


def test_resize_image():
    """测试图像尺寸调整"""
    print("=" * 60)
    print("测试图像尺寸调整")
    print("=" * 60)

    # 创建测试图像
    original_shape = (8, 8, 8)
    image = torch.randn(original_shape)

    # 测试1: 放大图像
    print("\n1. 测试图像放大")
    larger_shape = (16, 16, 16)
    enlarged = resize_image(image, larger_shape)

    print(f"原始图像形状: {image.shape}")
    print(f"放大后图像形状: {enlarged.shape}")
    print(f"放大比例: {larger_shape[0] / original_shape[0]:.1f}x")

    assert enlarged.shape == larger_shape, "放大后图像形状应该匹配目标尺寸"

    # 测试2: 缩小图像
    print("\n2. 测试图像缩小")
    smaller_shape = (4, 4, 4)
    reduced = resize_image(image, smaller_shape)

    print(f"缩小后图像形状: {reduced.shape}")
    print(f"缩小比例: {smaller_shape[0] / original_shape[0]:.1f}x")

    assert reduced.shape == smaller_shape, "缩小后图像形状应该匹配目标尺寸"

    # 测试3: 保持原尺寸
    print("\n3. 测试保持原尺寸")
    same_size = resize_image(image, original_shape)

    print(f"保持原尺寸图像形状: {same_size.shape}")
    assert same_size.shape == original_shape, "保持原尺寸应该不变"

    print("\n✓ 图像尺寸调整所有测试通过!\n")


def test_rotate_image():
    """测试图像旋转"""
    print("=" * 60)
    print("测试图像旋转")
    print("=" * 60)

    # 创建测试图像（中心有亮点）
    image = torch.zeros(10, 10, 10)
    image[5, 5, 5] = 1.0  # 中心点

    # 测试1: 绕Z轴旋转90度
    print("\n1. 测试绕Z轴旋转")
    rotated_z = rotate_image(image, [0, 0, math.pi / 2])  # 90度

    print(f"原始图像形状: {image.shape}")
    print(f"旋转后图像形状: {rotated_z.shape}")

    # 查找旋转后的亮点位置
    max_pos = torch.argmax(rotated_z)
    max_coord = np.unravel_index(max_pos.item(), rotated_z.shape)
    print(f"旋转后最大值位置: {max_coord}")

    assert rotated_z.shape == image.shape, "旋转后图像形状应该不变"
    assert not torch.allclose(image, rotated_z), "旋转后图像应该不同"

    # 测试2: 绕X轴旋转90度
    print("\n2. 测试绕X轴旋转")
    rotated_x = rotate_image(image, [math.pi / 2, 0, 0])  # 90度

    max_pos_x = torch.argmax(rotated_x)
    max_coord_x = np.unravel_index(max_pos_x.item(), rotated_x.shape)
    print(f"绕X轴旋转后最大值位置: {max_coord_x}")

    assert max_coord_x != max_coord, "不同轴旋转应该产生不同结果"


def test_translate_image():
    """测试图像平移"""
    print("=" * 60)
    print("测试图像平移")
    print("=" * 60)

    # 创建测试图像（角落有亮点）
    image = torch.zeros(8, 8, 8)
    image[0, 0, 0] = 1.0  # 角落点

    # 测试1: 正向平移
    print("\n1. 测试正向平移")
    translation = [2.0, 3.0, 1.0]
    translated = translate_image(image, translation)

    print(f"原始图像形状: {image.shape}")
    print(f"平移后图像形状: {translated.shape}")

    # 查找平移后的亮点位置
    max_pos = torch.argmax(translated)
    max_coord = np.unravel_index(max_pos.item(), translated.shape)
    print(f"平移后最大值位置: {max_coord}")

    expected_pos = (1, 3, 2)
    assert max_coord == expected_pos, f"亮点应该移动到位置{expected_pos}"

    # 测试2: 负向平移
    print("\n2. 测试负向平移")
    negative_translation = [-1.0, -2.0, -1.0]
    translated_neg = translate_image(image, negative_translation)

    max_pos_neg = torch.argmax(translated_neg)
    max_coord_neg = np.unravel_index(max_pos_neg.item(), translated_neg.shape)
    print(f"负向平移后最大值位置: {max_coord_neg}")

    # 由于负向平移可能超出边界，亮点可能消失
    print("负向平移可能使亮点移出图像边界")

    print("\n✓ 图像平移所有测试通过!\n")


def test_create_affine_matrix():
    """测试仿射矩阵创建"""
    print("=" * 60)
    print("测试仿射矩阵创建")
    print("=" * 60)

    # 测试1: 创建单位仿射矩阵
    print("\n1. 测试单位仿射矩阵")
    identity_rotation = torch.eye(3)
    identity_scale = [1.0, 1.0, 1.0]
    identity_translation = [0.0, 0.0, 0.0]

    affine_identity = create_affine_matrix(identity_rotation, identity_scale, identity_translation)

    print(f"单位仿射矩阵形状: {affine_identity.shape}")
    print(f"单位仿射矩阵:\n{affine_identity}")

    assert affine_identity.shape == (4, 4), "仿射矩阵应该是4x4"
    assert torch.allclose(affine_identity, torch.eye(4)), "单位仿射矩阵应该等于单位矩阵"

    # 测试2: 创建缩放仿射矩阵
    print("\n2. 测试缩放仿射矩阵")
    scale_vector = [2.0, 1.5, 0.8]
    affine_scaled = create_affine_matrix(identity_rotation, scale_vector, identity_translation)

    print(f"缩放仿射矩阵:\n{affine_scaled}")

    # 检查缩放部分
    scale_diag = torch.diag(affine_scaled[:3, :3])
    expected_scale = torch.tensor(scale_vector)
    assert torch.allclose(scale_diag, expected_scale), "缩放矩阵应该匹配输入"

    print("\n✓ 仿射矩阵创建所有测试通过!\n")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行高斯图像配准工具模块测试")
    print("=" * 80)

    # 运行所有测试函数
    test_functions = [
        test_ncc_loss,
        test_gradient_descent,
        test_adam_optimizer,
        test_early_stopping,
        test_affine_transform,
        test_resize_image,
        test_rotate_image,
        test_translate_image,
        test_create_affine_matrix
    ]

    passed_count = 0
    total_count = len(test_functions)

    for test_func in test_functions:
        try:
            test_func()
            passed_count += 1
        except Exception as e:
            print(f"❌ 测试失败: {test_func.__name__}")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print(f"测试完成: {passed_count}/{total_count} 个测试通过")

    if passed_count == total_count:
        print("🎉 所有测试通过! 工具模块功能正常")
    else:
        print("⚠️  部分测试失败，请检查代码")

    return passed_count == total_count


if __name__ == "__main__":
    # 设置随机种子以确保测试可重复
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行所有测试
    success = run_all_tests()

    # 退出代码：0表示成功，1表示失败
    sys.exit(0 if success else 1)