import os
import glob
import nibabel as nib


def find_mri_files(directory="D:/MRI", file_extensions=None):
    """查找指定目录中的医学图像文件"""
    if file_extensions is None:
        file_extensions = ['.mnc', '.nii', '.nii.gz', '.img', '.hdr']

    if not os.path.exists(directory):
        print(f"警告: 目录不存在 - {directory}")
        return []

    found_files = []
    for ext in file_extensions:
        pattern = os.path.join(directory, f"*{ext}")
        files = glob.glob(pattern)
        found_files.extend(files)

    # 去重和排序
    found_files = sorted(list(set(found_files)))
    return found_files


def validate_file_path(filepath):
    """验证文件路径是否存在且可读"""
    if not os.path.exists(filepath):
        return False, f"文件不存在: {filepath}"

    if not os.path.isfile(filepath):
        return False, f"路径不是文件: {filepath}"

    try:
        # 尝试读取文件头信息
        if filepath.endswith(('.nii', '.nii.gz', '.mnc')):
            nib.load(filepath)
        return True, "文件有效"
    except Exception as e:
        return False, f"文件读取失败: {e}"


def get_file_info(filepath):
    """获取医学图像文件的基本信息"""
    info = {
        'filename': os.path.basename(filepath),
        'filepath': filepath,
        'size_bytes': 0,
        'size_mb': 0,
        'format': 'unknown',
        'valid': False,
        'message': ''
    }

    if not os.path.exists(filepath):
        info['message'] = f"文件不存在: {filepath}"
        return info

    try:
        info['size_bytes'] = os.path.getsize(filepath)
        info['size_mb'] = info['size_bytes'] / (1024 * 1024)

        # 检测文件格式
        if filepath.endswith('.mnc'):
            info['format'] = 'MNC'
        elif filepath.endswith('.nii'):
            info['format'] = 'NIfTI'
        elif filepath.endswith('.nii.gz'):
            info['format'] = 'NIfTI GZ'
        elif filepath.endswith('.img'):
            info['format'] = 'Analyze IMG'
        elif filepath.endswith('.hdr'):
            info['format'] = 'Analyze HDR'

        # 验证文件可读性
        valid, message = validate_file_path(filepath)
        info['valid'] = valid
        info['message'] = message

    except Exception as e:
        info['message'] = f"获取文件信息失败: {e}"

    return info


def list_mri_files(directory="D:/MRI", detailed=False):
    """列出目录中的所有MRI文件"""
    files = find_mri_files(directory)

    if not detailed:
        return files

    # 返回详细信息
    file_info_list = []
    for filepath in files:
        info = get_file_info(filepath)
        file_info_list.append(info)

    return file_info_list