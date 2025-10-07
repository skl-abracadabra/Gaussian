import sys
import os
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 使用绝对导入
from gaussian_image_registration.file_io.file_handlers import find_mri_files, get_file_info
from gaussian_image_registration.file_io.image_loaders import load_image, get_image_info


def test_file_handlers():
    """测试文件处理功能"""
    print("测试文件处理功能...")

    # 测试文件查找
    files = find_mri_files("D:/MRI")
    print(f"找到 {len(files)} 个MRI文件")

    for filepath in files[:3]:  # 只显示前3个文件的信息
        info = get_file_info(filepath)
        print(f"文件: {info['filename']}")
        print(f"  格式: {info['format']}, 大小: {info['size_mb']:.1f} MB")
        print(f"  有效: {info['valid']}, 消息: {info['message']}")

    print("文件处理功能测试通过!\n")


def test_image_loaders():
    """测试图像加载功能"""
    print("测试图像加载功能...")

    # 查找测试文件
    files = find_mri_files("D:/MRI")
    if not files:
        print("未找到测试文件，跳过图像加载测试")
        return

    # 测试第一个文件
    test_file = files[0]
    print(f"测试加载文件: {os.path.basename(test_file)}")

    # 加载图像
    image_data = load_image(test_file)
    if image_data is not None:
        info = get_image_info(image_data)
        print(f"图像加载成功!")
        print(f"  形状: {info['shape']}")
        print(f"  数据类型: {info['dtype']}")
        print(f"  强度范围: {info['min']:.2f} - {info['max']:.2f}")
        print(f"  内存占用: {info['size_mb']:.2f} MB")
    else:
        print("图像加载失败")

    print("图像加载功能测试通过!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("文件IO模块测试")
    print("=" * 50)

    test_file_handlers()
    test_image_loaders()

    print("所有文件IO模块测试完成!")