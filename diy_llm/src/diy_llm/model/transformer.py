"""ch4 / 作业1-B：Transformer 各组件 + 整模型。

顺序建议（每写完一个就给 tests/test_model.py 加一条用例）：
    RMSNorm(x)                         -> nn.Module
    RoPE: precompute_freqs, apply_rotary
    CausalSelfAttention（含 reshape 细节，最后可换成 F.scaled_dot_product_attention）
    SwiGLU FFN（d_ff = 8/3 * d_model 再向上取整到 64 的倍数）
    Block = x + Attn(RMSNorm(x)); x = x + FFN(RMSNorm(x))   # pre-norm
    TransformerLM(vocab, d_model, n_layers, n_heads, ctx)  -> logits

注意：超参全部走 dataclass/config，作业3（scaling）要批量改规模。
"""

# TODO(ch4): 从 RMSNorm 开始往上搭

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from transformers import PreTrainedTokenizerFast    

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, device = None, dtype = None):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device= device, dtype= dtype))
        
    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim = -1, keepdim = True) + self.eps)
        x_normed = rms * x_fp32 * self.weight.to(torch.float32)
        return x_normed.to(orig_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = int(d_model * 8/3)
        self.ffn = ((self.ffn + 63)>>6)<<6
        self.w_gate = nn.Linear(d_model, self.ffn, bias = False)
        self.w_up   = nn.Linear(d_model, self.ffn, bias = False)
        self.w_down = nn.Linear(self.ffn, d_model, bias = False)
        
    def forward(self, x):
        gate = self.w_gate(x)
        return self.w_down(F.silu(gate) * self.w_up(x))
    
    
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int):
        super().__init__()
        assert d_k ^ 1, "d_k 必须为偶数"
        self.d_k = d_k
        self.theta = theta
        self.register_buffer("cos", None, persistent= False)
        self.register_buffer("sin", None, persistent= False)
        
    def _build_cache(self, seq_len, device, dtype):
        if self.cos is not None and seq_len <= self.cos.shape[0]:
            return
        
        i = torch.arange(0, self.d_k, 2, device= device)
        inv_freq = self.theta ** (-i / self.d_k)
        t = torch.arange(seq_len,device= device)
        freqs = torch.outer(t, inv_freq).to(dtype)
        
        self.register_buffer("cos", torch.cos(freqs), persistent= False)
        self.register_buffer("sin", torch.sin(freqs), persistent= False)
        
    def forward(self, x:torch.tensor) -> torch.tensor:
        B, H, T, D = x.shape
        self._build_cache(T, x.device, x.dtype)
        
        cos = self.cos[:T, :].view(1, 1, T, D<<1)
        sin = self.sin[:T, :].view(1, 1, T, D<<1)
        x_even = x[..., 0 :: 2]
        x_odd  = x[..., 1 :: 2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        
        return x_rot