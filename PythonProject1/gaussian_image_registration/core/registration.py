import torch
import numpy as np
from scipy.spatial import cKDTree
from .primitives import GaussianPrimitive


class GaussianDIR:
    """高斯可变形图像配准"""

    def __init__(self, image_shape, num_primitives=100, k_neighbors=3):
        """
        初始化高斯配准器

        参数:
            image_shape: 图像形状 (D, H, W)
            num_primitives: 高斯基元数量
            k_neighbors: 最近邻高斯数量
        """
        self.image_shape = image_shape
        self.num_primitives = num_primitives
        self.k_neighbors = k_neighbors
        self.gaussians = self._initialize_gaussians()
        self._build_kdtree()

    def _initialize_gaussians(self):
        """在图像空间内均匀初始化高斯基元"""
        # 计算网格大小
        grid_size = max(1, int(np.ceil(self.num_primitives ** (1 / 3))))
        gaussians = []

        # 创建均匀网格点
        x_coords = np.linspace(0, self.image_shape[0] - 1, grid_size)
        y_coords = np.linspace(0, self.image_shape[1] - 1, grid_size)
        z_coords = np.linspace(0, self.image_shape[2] - 1, grid_size)

        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                for k, z in enumerate(z_coords):
                    if len(gaussians) >= self.num_primitives:
                        break

                    # 创建高斯参数
                    mu = torch.tensor([x, y, z], dtype=torch.float32)
                    scale = torch.ones(3, dtype=torch.float32) * 0.1
                    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # 单位四元数
                    translation = torch.zeros(3, dtype=torch.float32)

                    # 创建高斯基元
                    gaussian = GaussianPrimitive(mu, scale, rotation, translation)
                    gaussians.append(gaussian)

        print(f"初始化了 {len(gaussians)} 个高斯基元")
        return gaussians

    def _build_kdtree(self):
        """构建KDTree用于快速近邻搜索"""
        positions = np.array([g.mu.detach().numpy() for g in self.gaussians])
        self.kdtree = cKDTree(positions)

    def compute_dvf(self, batch_size=1000):
        """计算变形向量场(DVF)"""
        D, H, W = self.image_shape
        dvf = torch.zeros((D, H, W, 3), dtype=torch.float32)
        total_voxels = D * H * W

        print(f"开始计算DVF，图像尺寸: {self.image_shape}，总体素数: {total_voxels}")

        for start in range(0, total_voxels, batch_size):
            end = min(start + batch_size, total_voxels)
            batch_indices = np.arange(start, end)
            batch_coords = np.unravel_index(batch_indices, (D, H, W))

            for idx, (z, y, x) in enumerate(zip(*batch_coords)):
                voxel_pos = torch.tensor([x, y, z], dtype=torch.float32)
                new_pos = self._transform_voxel(voxel_pos)
                dvf[z, y, x] = new_pos - voxel_pos

            if end % (batch_size * 10) == 0 or end == total_voxels:
                print(f"处理进度: {end}/{total_voxels} ({end / total_voxels * 100:.1f}%)")

        return dvf

    def _transform_voxel(self, voxel_pos):
        """变换单个体素位置"""
        # 查询最近邻高斯
        voxel_np = voxel_pos.detach().numpy().reshape(1, -1)
        distances, indices = self.kdtree.query(voxel_np, k=self.k_neighbors)

        # 计算混合权重
        weights = torch.softmax(torch.tensor(-distances, dtype=torch.float32), dim=1)

        transformed_pos = torch.zeros(3, dtype=torch.float32)

        for i, g_idx in enumerate(indices[0]):
            g = self.gaussians[g_idx]
            R = g._quaternion_to_matrix(g.rotation)
            relative_pos = voxel_pos - g.mu
            local_transform = R @ relative_pos + g.mu + g.translation
            transformed_pos += weights[0][i] * local_transform

        return transformed_pos

    def warp_image(self, image, dvf):
        """使用DVF变形图像"""
        D, H, W = image.shape
        warped = torch.zeros_like(image)

        print(f"开始图像变形，图像尺寸: {image.shape}")

        for z in range(D):
            for y in range(H):
                for x in range(W):
                    dx, dy, dz = dvf[z, y, x]
                    new_x = x + dx.item()
                    new_y = y + dy.item()
                    new_z = z + dz.item()

                    # 边界检查
                    if (0 <= new_x < W and 0 <= new_y < H and 0 <= new_z < D):
                        # 最近邻插值（简化版）
                        warped[z, y, x] = image[int(new_z), int(new_y), int(new_x)]

            if z % (D // 10) == 0 or z == D - 1:
                print(f"变形进度: {z + 1}/{D} ({(z + 1) / D * 100:.1f}%)")

        return warped

    def get_trainable_parameters(self):
        """获取所有可训练参数"""
        params = []
        for g in self.gaussians:
            params.extend(g.get_trainable_parameters())
        return params

    def optimize(self, fixed, moving, iterations=30, lr=0.01):
        """优化高斯参数以最大化图像相似性"""
        params = self.get_trainable_parameters()
        optimizer = torch.optim.Adam(params, lr=lr)

        best_loss = float('inf')
        best_dvf = None

        print(f"开始优化，迭代次数: {iterations}")

        for iter in range(iterations):
            optimizer.zero_grad()

            # 计算当前DVF
            dvf = self.compute_dvf()

            # 变形移动图像
            warped = self.warp_image(moving, dvf)

            # 计算相似性损失（负NCC）
            loss = -self.ncc(fixed, warped)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 更新KDTree（高斯位置可能已改变）
            self._build_kdtree()

            print(f"迭代 {iter + 1}/{iterations}: 损失 = {loss.item():.4f}")

            # 保存最佳结果
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_dvf = dvf.clone()

            # 早期停止检查
            if -loss.item() > 0.9:  # NCC > 0.9
                print("早期停止: NCC > 0.9 已达到")
                break

        return best_dvf if best_dvf is not None else dvf

    def ncc(self, I, J):
        """归一化互相关(NCC)相似性度量"""
        I_mean = torch.mean(I)
        J_mean = torch.mean(J)
        I_std = torch.std(I)
        J_std = torch.std(J)

        ncc_value = torch.mean((I - I_mean) * (J - J_mean)) / (I_std * J_std + 1e-8)
        return ncc_value

    def __repr__(self):
        return f"GaussianDIR(image_shape={self.image_shape}, num_primitives={self.num_primitives})"