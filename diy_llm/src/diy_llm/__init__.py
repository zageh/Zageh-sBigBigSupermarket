"""diy-llm 自学实现包。

按章节依次填充：
    tokenizer/   ch2   BPE 训练与编解码（作业1-A）
    model/       ch4   RMSNorm / RoPE / Attention / SwiGLU / Transformer（作业1-B）
    optimizer.py ch3/4 AdamW + warmup + cosine
    data.py      ch11  分块读文本、打包 .bin、memmap Dataset（作业4）
    train.py     ch4/13 训练循环、验证、ckpt
    generate.py  ch10  KV cache 与采样
    kernels/     ch7   Triton FlashAttention-2（作业2）
    parallel/    ch8   DDP / 分片优化器（作业2）
    alignment/   ch13/14 SFT 与 GRPO（作业5）
"""

__version__ = "0.1.0"
