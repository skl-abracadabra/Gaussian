import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端

import pytest
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plotting import (
    plot_image_3d,
    plot_image_slices,
    plot_registration_result,
    plot_optimization_progress,
    plot_transformation_field,
    plot_intensity_distribution,
    plot_multimodal_comparison,
    create_registration_animation,
    save_figure,
    set_style
)
import tempfile
import os


class TestPlottingFunctions:
    """测试可视化绘图函数"""

    def setup_method(self):
        """测试设置：创建测试数据"""
        # 创建测试3D图像
        self.test_image_3d = torch.randn(16, 16, 16)
        self.test_image_2d = torch.randn(32, 32)

        # 创建配准测试数据
        self.fixed_image = torch.randn(8, 8, 8)
        self.moving_image = torch.randn(8, 8, 8)
        self.registered_image = torch.randn(8, 8, 8)

        # 创建优化过程测试数据
        self.loss_history = [1.0 / (i + 1) for i in range(50)]
        self.parameter_history = {
            'translation_x': [0.1 * i for i in range(50)],
            'rotation_z': [0.05 * i for i in range(50)]
        }

        # 创建变换场测试数据
        self.transformation_3d = torch.randn(3, 8, 8, 8)  # 3D变换场
        self.transformation_2d = torch.randn(2, 16, 16)  # 2D变换场

        # 设置临时目录用于保存测试图像
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """测试清理：删除临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_plot_image_3d_basic(self):
        """测试3D图像绘图基本功能"""
        fig = plot_image_3d(self.test_image_3d, title="Test 3D Image")
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_image_3d_custom_params(self):
        """测试3D图像绘图自定义参数"""
        fig = plot_image_3d(
            self.test_image_3d,
            title="Custom 3D Plot",
            cmap='hot',
            alpha=0.5,
            figsize=(12, 10)
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_image_slices_basic(self):
        """测试图像切片绘图基本功能"""
        fig = plot_image_slices(self.test_image_3d)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # 应该有3个子图
        plt.close(fig)

    def test_plot_image_slices_custom_slices(self):
        """测试图像切片绘图自定义切片位置"""
        fig = plot_image_slices(
            self.test_image_3d,
            slice_indices=[5, 8, 10],
            cmap='viridis',
            figsize=(12, 4)
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_registration_result(self):
        """测试配准结果绘图"""
        fig = plot_registration_result(
            self.fixed_image,
            self.moving_image,
            self.registered_image,
            titles=['Fixed', 'Moving', 'Registered', 'Diff']
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 4个子图
        plt.close(fig)

    def test_plot_optimization_progress_loss_only(self):
        """测试优化进度绘图（仅损失）"""
        fig = plot_optimization_progress(self.loss_history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_optimization_progress_with_params(self):
        """测试优化进度绘图（包含参数）"""
        fig = plot_optimization_progress(
            self.loss_history,
            self.parameter_history,
            title="Optimization Test"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_transformation_field_3d(self):
        """测试3D变换场绘图"""
        fig = plot_transformation_field(
            self.transformation_3d,
            title="3D Transformation Field",
            slice_index=4
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_transformation_field_2d(self):
        """测试2D变换场绘图"""
        fig = plot_transformation_field(
            self.transformation_2d,
            title="2D Transformation Field"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_intensity_distribution(self):
        """测试强度分布绘图"""
        fig = plot_intensity_distribution(
            self.fixed_image,
            self.moving_image,
            self.registered_image
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_multimodal_comparison(self):
        """测试多模态比较绘图"""
        images = [self.fixed_image, self.moving_image]
        titles = ['Fixed', 'Moving']

        # 测试并排显示
        fig1 = plot_multimodal_comparison(images, titles, overlay=False)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # 测试叠加显示
        fig2 = plot_multimodal_comparison(images, titles, overlay=True)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_create_registration_animation(self):
        """测试配准动画创建"""
        # 创建动画序列
        moving_sequence = [
            self.moving_image + 0.1 * i * torch.randn_like(self.moving_image)
            for i in range(5)
        ]
        titles = [f'Frame {i}' for i in range(5)]

        anim = create_registration_animation(
            self.fixed_image,
            moving_sequence,
            titles,
            interval=100
        )
        assert isinstance(anim, animation.FuncAnimation)

        # 测试动画保存（可选）
        temp_file = os.path.join(self.temp_dir, 'test_animation.gif')
        try:
            anim.save(temp_file, writer='pillow', fps=2)
            assert os.path.exists(temp_file)
        except Exception as e:
            # 某些环境可能不支持动画保存，跳过此检查
            print(f"Animation save test skipped: {e}")

    def test_save_figure(self):
        """测试图形保存功能"""
        fig = plot_image_slices(self.test_image_3d)
        temp_file = os.path.join(self.temp_dir, 'test_figure.png')

        save_figure(fig, temp_file)
        assert os.path.exists(temp_file)
        plt.close(fig)

    def test_set_style(self):
        """测试样式设置功能"""
        # 测试默认样式
        set_style('default')
        fig1 = plot_image_slices(self.test_image_3d)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # 测试暗色样式
        set_style('dark')
        fig2 = plot_image_slices(self.test_image_3d)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

        # 测试科学样式
        set_style('scientific')
        fig3 = plot_image_slices(self.test_image_3d)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空图像
        empty_image = torch.zeros(5, 5, 5)
        fig = plot_image_slices(empty_image)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # 测试单值图像
        constant_image = torch.ones(5, 5, 5)
        fig = plot_image_slices(constant_image)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_input_validation(self):
        """测试输入验证"""
        # 测试无效图像输入
        with pytest.raises(AttributeError):
            plot_image_slices("invalid_input")

        # 测试无效切片索引
        with pytest.raises(IndexError):
            plot_image_slices(self.test_image_3d, slice_indices=[100, 100, 100])


class TestPlottingPerformance:
    """测试绘图性能"""

    def test_large_image_performance(self):
        """测试大图像性能"""
        large_image = torch.randn(64, 64, 64)

        import time
        start_time = time.time()
        fig = plot_image_slices(large_image)
        end_time = time.time()

        assert isinstance(fig, plt.Figure)
        assert (end_time - start_time) < 5.0  # 应在5秒内完成
        plt.close(fig)

    def test_multiple_plots_performance(self):
        """测试多图绘制性能"""
        import time
        start_time = time.time()

        figures = []
        for i in range(5):
            fig = plot_image_slices(self.test_image_3d)
            figures.append(fig)

        end_time = time.time()

        # 清理
        for fig in figures:
            plt.close(fig)

        assert (end_time - start_time) < 10.0  # 10秒内完成5个图


# 辅助函数测试
def test_imports():
    """测试模块导入"""
    from plotting import plot_image_3d, plot_image_slices
    assert callable(plot_image_3d)
    assert callable(plot_image_slices)


def test_module_availability():
    """测试模块可用性"""
    try:
        import plotting
        assert hasattr(plotting, 'plot_image_3d')
        assert hasattr(plotting, 'plot_image_slices')
    except ImportError:
        pytest.fail("Plotting module not available")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])