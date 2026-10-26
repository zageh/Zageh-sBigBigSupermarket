"""ch2：分块读取大语料 + 训练 BPE，并监控内存与吞吐。

用法（计划）：
    python -m diy_llm.tokenizer.train_bpe --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --out data/bpe_tokenizer/tokenizer.json
"""

# TODO(ch2): 分块读（避免一次性读入）、psutil 打印 peak RSS、tokens/s
