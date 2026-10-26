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
