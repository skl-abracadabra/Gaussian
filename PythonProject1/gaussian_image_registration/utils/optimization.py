import torch
import numpy as np
from typing import List, Callable, Dict


def ncc_loss(fixed: torch.Tensor, moving: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    归一化互相关损失函数 (Negative Normalized Cross-Correlation)

    参数:
        fixed: 固定图像张量
        moving: 移动图像张量
        eps: 数值稳定性常数

    返回:
        torch.Tensor: 负NCC损失值（最小化此值相当于最大化图像相似性）
    """
    if fixed.shape != moving.shape:
        raise ValueError(f"图像形状不匹配: fixed {fixed.shape}, moving {moving.shape}")

    # 计算均值和标准差
    fixed_mean = torch.mean(fixed)
    moving_mean = torch.mean(moving)
    fixed_std = torch.std(fixed)
    moving_std = torch.std(moving)

    # 计算NCC
    numerator = torch.mean((fixed - fixed_mean) * (moving - moving_mean))
    denominator = fixed_std * moving_std + eps

    ncc = numerator / denominator
    return -ncc  # 返回负值以便最小化


def gradient_descent(params: List[torch.Tensor],
                     loss_fn: Callable[[], torch.Tensor],
                     lr: float = 0.01,
                     max_iter: int = 100,
                     tolerance: float = 1e-6) -> Dict:
    """
    梯度下降优化算法

    参数:
        params: 可训练参数列表
        loss_fn: 损失函数
        lr: 学习率
        max_iter: 最大迭代次数
        tolerance: 收敛容忍度

    返回:
        dict: 优化历史信息
    """
    history = {'loss': [], 'params': [], 'gradient_norms': []}

    for iter in range(max_iter):
        # 清零梯度
        for param in params:
            if param.grad is not None:
                param.grad.zero_()

        # 计算损失
        loss = loss_fn()

        # 反向传播
        loss.backward()

        # 更新参数
        with torch.no_grad():
            for param in params:
                if param.grad is not None:
                    param -= lr * param.grad

        # 记录历史
        history['loss'].append(loss.item())
        history['params'].append([p.detach().clone() for p in params])
        history['gradient_norms'].append(compute_gradient_norm(params))

        # 打印进度
        if iter % 10 == 0 or iter == max_iter - 1:
            print(
                f"迭代 {iter + 1}/{max_iter}: 损失 = {loss.item():.6f}, 梯度范数 = {history['gradient_norms'][-1]:.6f}")

        # 收敛检查
        if iter > 0 and abs(history['loss'][-1] - history['loss'][-2]) < tolerance:
            print(f"梯度下降收敛于迭代 {iter + 1}")
            break

    return history


def adam_optimizer(params: List[torch.Tensor],
                   loss_fn: Callable[[], torch.Tensor],
                   lr: float = 0.001,
                   max_iter: int = 100,
                   beta1: float = 0.9,
                   beta2: float = 0.999,
                   eps: float = 1e-8) -> Dict:
    """
    Adam优化算法 (Adaptive Moment Estimation) - 修复版本

    参数:
        params: 可训练参数列表
        loss_fn: 损失函数
        lr: 学习率
        max_iter: 最大迭代次数
        beta1: 一阶矩衰减率
        beta2: 二阶矩衰减率
        eps: 数值稳定性常数

    返回:
        dict: 优化历史
    """
    # 初始化矩估计
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]

    history = {'loss': [], 'params': []}

    for t in range(1, max_iter + 1):
        # 清零梯度
        for param in params:
            if param.grad is not None:
                param.grad.zero_()

        # 计算损失
        loss = loss_fn()
        loss.backward()

        with torch.no_grad():
            for i, param in enumerate(params):
                if param.grad is not None:
                    # 更新一阶矩估计
                    m[i] = beta1 * m[i] + (1 - beta1) * param.grad
                    # 更新二阶矩估计
                    v[i] = beta2 * v[i] + (1 - beta2) * torch.pow(param.grad, 2)

                    # 偏差校正
                    m_hat = m[i] / (1 - beta1 ** t)
                    v_hat = v[i] / (1 - beta2 ** t)

                    # 更新参数
                    param -= lr * m_hat / (torch.sqrt(v_hat) + eps)

        # 记录历史
        history['loss'].append(loss.item())
        history['params'].append([p.detach().clone() for p in params])

        # 打印进度
        if t % 10 == 0 or t == max_iter:
            print(f"Adam 迭代 {t}/{max_iter}: 损失 = {loss.item():.6f}")

    return history


def early_stopping(loss_history: List[float], patience: int = 10, min_delta: float = 1e-4) -> bool:
    """
    早期停止算法 - 修复版本

    参数:
        loss_history: 损失历史列表
        patience: 容忍轮数
        min_delta: 最小改进量

    返回:
        bool: 是否应该停止
    """
    if len(loss_history) < patience + 1:
        return False

    # 检查最近patience轮是否有显著改善
    recent_losses = loss_history[-patience:]
    best_recent = min(recent_losses)
    best_previous = min(loss_history[:-patience])

    # 如果最近patience轮没有超过历史最佳（考虑min_delta）
    if best_recent > best_previous - min_delta:
        return True

    return False


def compute_gradient_norm(params: List[torch.Tensor]) -> float:
    """
    计算参数梯度的L2范数

    参数:
        params: 参数列表

    返回:
        float: 梯度范数
    """
    total_norm = 0.0
    for param in params:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    return total_norm ** 0.5


def learning_rate_scheduler(initial_lr: float, iteration: int, decay_rate: float = 0.95,
                            decay_step: int = 100) -> float:
    """
    指数衰减学习率调度器

    参数:
        initial_lr: 初始学习率
        iteration: 当前迭代次数
        decay_rate: 衰减率
        decay_step: 衰减步长

    返回:
        float: 调整后的学习率
    """
    return initial_lr * (decay_rate ** (iteration // decay_step))