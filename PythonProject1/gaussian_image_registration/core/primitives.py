import torch
import numpy as np


class GaussianPrimitive:
    """高斯基元定义"""

    def __init__(self, mu, scale, rotation, translation):
        """
        初始化高斯基元

        参数:
            mu: 中心位置 [x, y, z]
            scale: 缩放向量 [sx, sy, sz]
            rotation: 旋转四元数 [qw, qx, qy, qz]
            translation: 平移向量 [tx, ty, tz]
        """
        # 确保所有参数都是叶子张量
        self.mu = mu.clone().detach()
        self.mu.requires_grad = True

        self.scale = scale.clone().detach()
        self.scale.requires_grad = True

        self.rotation = rotation.clone().detach()
        self.rotation.requires_grad = True

        self.translation = translation.clone().detach()
        self.translation.requires_grad = True

    def get_covariance_matrix(self):
        """计算协方差矩阵 Σ = R S S^T R^T"""
        R = self._quaternion_to_matrix(self.rotation)
        S = torch.diag(self.scale)
        return R @ S @ S.T @ R.T

    def _quaternion_to_matrix(self, q):
        """四元数转旋转矩阵 - 修复梯度警告"""
        # 分离张量进行计算，避免梯度警告
        qw, qx, qy, qz = q[0].item(), q[1].item(), q[2].item(), q[3].item()

        # 直接使用标量计算旋转矩阵元素
        r00 = 1.0 - 2.0 * qy ** 2 - 2.0 * qz ** 2
        r01 = 2.0 * qx * qy - 2.0 * qz * qw
        r02 = 2.0 * qx * qz + 2.0 * qy * qw

        r10 = 2.0 * qx * qy + 2.0 * qz * qw
        r11 = 1.0 - 2.0 * qx ** 2 - 2.0 * qz ** 2
        r12 = 2.0 * qy * qz - 2.0 * qx * qw

        r20 = 2.0 * qx * qz - 2.0 * qy * qw
        r21 = 2.0 * qy * qz + 2.0 * qx * qw
        r22 = 1.0 - 2.0 * qx ** 2 - 2.0 * qy ** 2

        # 创建旋转矩阵张量
        R = torch.tensor([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ], dtype=torch.float32)

        return R

    def get_trainable_parameters(self):
        """获取所有可训练参数"""
        return [self.mu, self.scale, self.rotation, self.translation]

    def __repr__(self):
        return f"GaussianPrimitive(mu={self.mu.detach().numpy()}, scale={self.scale.detach().numpy()})"