import nibabel as nib
import numpy as np
import torch
import os


def load_image(filepath, normalize=True, dtype=torch.float32):
    """加载医学图像文件"""
    try:
        # 检查文件是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 根据文件扩展名选择加载方法
        if filepath.endswith(('.nii', '.nii.gz')):
            img = nib.load(filepath)
        elif filepath.endswith('.mnc'):
            # MNC文件需要特殊处理
            img = load_mnc_file(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {filepath}")

        # 获取图像数据
        data = img.get_fdata()

        # 转换为PyTorch张量
        tensor_data = torch.tensor(data, dtype=dtype)

        # 归一化处理
        if normalize:
            tensor_data = normalize_image(tensor_data)

        return tensor_data

    except Exception as e:
        print(f"加载图像失败 {filepath}: {e}")
        return None


def load_mnc_file(filepath):
    """专门处理MNC文件加载"""
    try:
        # 尝试使用nibabel加载MNC
        img = nib.load(filepath)
        return img
    except Exception as e:
        print(f"标准MNC加载失败，尝试备选方法: {e}")
        # 这里可以添加其他MNC加载方法
        raise e


def save_image(data, filepath, affine=None, dtype=np.float32):
    """保存图像数据到文件"""
    try:
        # 转换数据格式
        if isinstance(data, torch.Tensor):
            data_np = data.detach().numpy().astype(dtype)
        else:
            data_np = data.astype(dtype)

        # 默认仿射矩阵
        if affine is None:
            affine = np.eye(4)

        # 创建NIfTI图像
        img = nib.Nifti1Image(data_np, affine)

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存文件
        nib.save(img, filepath)
        print(f"图像已保存: {filepath}")

    except Exception as e:
        print(f"保存图像失败 {filepath}: {e}")


def normalize_image(image_tensor, method='minmax'):
    """图像归一化"""
    if method == 'minmax':
        # 最小-最大归一化到[0,1]
        min_val = torch.min(image_tensor)
        max_val = torch.max(image_tensor)
        if max_val > min_val:
            normalized = (image_tensor - min_val) / (max_val - min_val)
        else:
            normalized = torch.zeros_like(image_tensor)

    elif method == 'zscore':
        # Z-score归一化（均值0，标准差1）
        mean_val = torch.mean(image_tensor)
        std_val = torch.std(image_tensor)
        if std_val > 0:
            normalized = (image_tensor - mean_val) / std_val
        else:
            normalized = torch.zeros_like(image_tensor)

    else:
        raise ValueError(f"不支持的归一化方法: {method}")

    return normalized


def get_image_info(image_tensor):
    """获取图像基本信息"""
    info = {
        'shape': image_tensor.shape,
        'dtype': str(image_tensor.dtype),
        'min': float(torch.min(image_tensor)),
        'max': float(torch.max(image_tensor)),
        'mean': float(torch.mean(image_tensor)),
        'std': float(torch.std(image_tensor)),
        'size_mb': image_tensor.element_size() * image_tensor.nelement() / (1024 * 1024)
    }
    return info