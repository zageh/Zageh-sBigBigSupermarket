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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        self.rope = RotaryPositionEmbedding(10000.0, self.d_k)
        
    def forward(self, x, mask):
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1,2)
        k = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1,2)
        v = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1,2)
        
        q = self.rope(q)
        k = self.rope(k)
        
        if self.flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask= mask,
                dropout_p= 0,
                is_causal= True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                mask = mask.expand(B, self.num_heads, T, T)
                scores = scores.masked_fill(~mask, float('-inf'))
                
            attn = torch.softmax(scores, dim = -1)
            out = attn @ v
        out = out.transpose(1,2).contiguous().view(B,T,-1)
        return self.w_o(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x, mask = None):
        x = x + self.attention(self.norm1, mask)
        x = x + self.ffn(self.norm2(x))
        return x
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len
        
        self.config = {
            "vocab_size": vocab_size,
            "context_length": max_seq_len,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_model": d_model,
        }
        
    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding(idx)
        mask = torch.tril(torch.ones((B, 1, T, T), device= idx.device, dtype= torch.bool))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, :-self.max_seq_len]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(1)))
                logits[logits < v[:,[-1]]] =  -float('Inf')
            
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
            
        return idx
    
class SimpleTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding = "utf-8") as f:
            self.token_to_id = json.load(f)
            
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.unk_id = self.token_to_id.get("<unk>", 0)
        
    def encode(self, text):
        return [self.token_to_id.get(c, self.unk_id) for c in text]
    
    def decode(self, ids):
        return "".join([self.id_to_token.get(i, "<unk>") for i in ids])