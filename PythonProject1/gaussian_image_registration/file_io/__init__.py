"""
高斯图像配准文件IO模块
处理医学图像的加载、保存和文件管理
"""

from .file_handlers import find_mri_files, validate_file_path, get_file_info, list_mri_files
from .image_loaders import load_image, save_image, normalize_image, get_image_info

# 注意：不要从test_io导入任何内容，因为它是测试文件
# from .test_io import download_from_url, upload_to_server  # 删除这行

__all__ = [
    'find_mri_files',
    'validate_file_path',
    'get_file_info',
    'list_mri_files',
    'load_image',
    'save_image',
    'normalize_image',
    'get_image_info'
]