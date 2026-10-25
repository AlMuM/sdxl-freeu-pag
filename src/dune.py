"""DUNE（Lim et al., arXiv:2607.09753）检测阶段在 SDXL 上的复现。

论文没有公开代码，以下按论文公式实现，含糊之处的取舍都写在注释里：

    z_t   = -h_t / sigma_t                       h_t: U-Net mid block 输出（SDXL 为 1280×32×32）
    Δ     = log|z_t / z̄_{t+1}|                    逐元素
    M     = Δ > λ，λ = Δ 的第 p 分位数            【取舍】每张图在全部 C×H×W 上算一个阈值
    n_t   = |h_t 的逐通道均值|                     形状 C×1×1
    ĥ_t   = (1 − M) ⊙ h_t + M ⊙ κ·n_t·h_t
    z̄_t   = γ·z̄_{t+1} + (1 − γ)·z_t               【取舍】用未抑制的 z_t 更新，首步 z̄ = z
    sigma_t = sqrt(1 − ᾱ_t)                        【取舍】论文未写明，取 VP 噪声标准差

只在前 end_frac 比例的步数内、跳过前 warmup 步后生效（论文：前 40%，SDXL 预热 3 步）。
CFG / PAG 会把多份预测拼在同一个 batch 里，每一份各自维护 EMA、各自算阈值。
"""
from __future__ import annotations

import torch


class Dune:
    def __init__(self, pipe):
        self.pipe = pipe
        self.cfg = None
        self.num_steps = 0
        self.reset()
        pipe.unet.register_forward_pre_hook(self._before_unet, with_kwargs=True)
        pipe.unet.mid_block.register_forward_hook(self._after_mid)

    def configure(self, cfg: dict | None, num_steps: int):
        """cfg 为 None 时关闭；否则需要 kappa、gamma、p、end_frac、warmup。"""
        self.cfg = cfg
        self.num_steps = num_steps
        self.reset()

    def reset(self):
        """每张图生成前调用：清空 EMA 和步数计数。"""
        self.step = -1
        self.sigma = None
        self.ema = None
        self.stats = []

    def _before_unet(self, module, args, kwargs):
        if not self.cfg:
            return
        self.step += 1  # CFG/PAG 的多份预测在同一次 U-Net 调用里，所以一次调用 = 一步
        t = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        t = int(torch.as_tensor(t).flatten()[0].round().item())
        acp = self.pipe.scheduler.alphas_cumprod[t].item()
        self.sigma = (1.0 - acp) ** 0.5

    def _after_mid(self, module, inputs, h):
        if not self.cfg:
            return None
        c = self.cfg
        hf = h.float()
        z = -hf / self.sigma
        if self.ema is None:
            self.ema = z
            return None
        active = c["warmup"] <= self.step < int(c["end_frac"] * self.num_steps)
        out = None
        if active:
            eps = 1e-6
            delta = torch.log(z.abs() + eps) - torch.log(self.ema.abs() + eps)
            flat = delta.flatten(1)
            k = max(1, int(round(c["p"] * flat.shape[1])))
            lam = flat.kthvalue(k, dim=1).values.view(-1, 1, 1, 1)
            mask = delta > lam
            n_t = hf.mean(dim=(2, 3), keepdim=True).abs()
            factor = c["kappa"] * n_t
            new = torch.where(mask, factor * hf, hf)
            self.stats.append({
                "step": self.step,
                "masked": mask.float().mean().item(),
                "factor_mean": factor.mean().item(),
                "factor_max": factor.max().item(),
                "shrunk": (factor < 1).float().mean().item(),  # 被压小（而非放大）的通道比例
            })
            out = new.to(h.dtype)
        self.ema = c["gamma"] * self.ema + (1 - c["gamma"]) * z
        return out
