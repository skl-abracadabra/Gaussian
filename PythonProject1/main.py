import torch
from gaussian_image_registration.core.registration import GaussianDIR
from gaussian_image_registration.file_io.file_handlers import find_mri_files
from gaussian_image_registration.file_io.image_loaders import load_image
from gaussian_image_registration.visualization.plotting import plot_registration_result


def main():
    """主程序"""
    print("高斯可变形图像配准")

    # 查找MRI文件
    files = find_mri_files("D:/MRI")

    if files:
        print("找到的MRI文件:")
        for i, f in enumerate(files):
            print(f"{i + 1}. {f}")

        # 选择文件
        fixed_idx = int(input("选择固定图像编号: ")) - 1
        moving_idx = int(input("选择移动图像编号: ")) - 1

        fixed_path = files[fixed_idx]
        moving_path = files[moving_idx]

        # 加载图像
        fixed = load_image(fixed_path)
        moving = load_image(moving_path)


    # 运行配准
    registrar = GaussianDIR(fixed.shape)
    dvf = registrar.optimize(fixed, moving, iterations=30)
    warped = registrar.warp_image(moving, dvf)

    # 显示结果
    plot_registration_result(fixed, moving, warped)

    print("配准完成")


if __name__ == "__main__":
    main()