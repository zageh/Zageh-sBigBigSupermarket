"""ch2 / 作业1-A：BPE 分词器（自己实现，不调用 tokenizers 的 train）。

计划实现：
    train_bpe(input_path, vocab_size, special_tokens) -> (vocab, merges)
        1. 用 GPT-2 正则预切分成 pre-token
        2. 统计词频（Counter[tuple[bytes, ...]]）
        3. 迭代：统计相邻 pair 频次 -> 合并最高频 -> 更新词表
    BPETokenizer.encode(text) -> list[int]
    BPETokenizer.decode(ids) -> str        # 必须 round-trip 完全一致
    save/from_files 兼容 tokenizers 的 tokenizer.json 格式

验收：在 TinyStories 10MB 上训 10k 词表；与 tokenizers 库训练结果对比合并顺序一致。
参考（卡住 30 分钟后再看）：ref/diy-llm/coursework/assignment1-basics/CS336_Assignment1_BPE.ipynb
"""

# TODO(ch2): 先写 pre_tokenize + 训练循环，再加 encode/decode
