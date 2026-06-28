"""ValueNorm - 价值/回报归一化 (PopArt 式, MAPPO 同款).

动机 (见 CLAUDE.md "vf_explained_var≈0" 现象): 本环境奖励被一坨近乎恒定的
基础利润地板 (~328±8) 主导, critic 若直接回归原始 returns, 其价值头被迫去
表示那个大常数, 可解释残差方差被淹没 -> vf_explained_var≈0, advantage 无信号.

做法: critic 输出【归一化】价值 (~N(0,1)); 用去偏 running mean/var 把地板搬到
归一化层里. GAE/bootstrap 在原始奖励空间进行 (denormalize critic 输出), critic
损失在归一化空间算 (normalize returns 作目标). 这样价值头只需拟合标准化残差,
可解释结构浮出, advantage 获得信号.
"""
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """逐 agent 去偏 running mean/var 归一化 (Welford-EMA, 与 MAPPO ValueNorm 一致)."""

    def __init__(self, num_agents: int, beta: float = 0.99999, eps: float = 1e-5):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(num_agents))
        self.register_buffer("running_mean_sq", torch.zeros(num_agents))
        self.register_buffer("debiasing_term", torch.zeros(1))

    def _stats(self):
        d = self.debiasing_term.clamp(min=self.eps)
        mean = self.running_mean / d
        var = (self.running_mean_sq / d - mean ** 2).clamp(min=1e-2)
        return mean, var

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """x: (..., num_agents) 原始 returns. 去偏 EMA 更新, 早期估计即无偏."""
        x = x.reshape(-1, x.shape[-1])
        batch_mean = x.mean(dim=0)
        batch_sq = (x ** 2).mean(dim=0)
        w = 1.0 - self.beta
        self.running_mean.mul_(self.beta).add_(batch_mean * w)
        self.running_mean_sq.mul_(self.beta).add_(batch_sq * w)
        self.debiasing_term.mul_(self.beta).add_(w)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._stats()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._stats()
        return x * torch.sqrt(var) + mean
