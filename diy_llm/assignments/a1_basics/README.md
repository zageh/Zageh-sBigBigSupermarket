# 作业1：手搓大模型（basics）

上游：`https://datawhalechina.github.io/diy-llm/coursework/assignment1-basics/`（本地在 `ref/diy-llm/coursework/assignment1-basics/`）

## 分工：代码在 src，证据在这里

| 作业部分 | 实现位置 | 交付物 |
| --- | --- | --- |
| BPE 分词器 + 训练 | `src/diy_llm/tokenizer/` | `writeup.md` 的 1.x 题 |
| Transformer 架构 | `src/diy_llm/model/transformer.py` | 架构说明 + 形状推演 |
| 优化器 / 训练循环 | `src/diy_llm/optimizer.py`, `src/diy_llm/train.py` | 训练曲线 |
| 文本生成 | `src/diy_llm/generate.py` | 生成样例若干 |
| 3 个消融 | `ablations/`（no RMSNorm / post-norm / SiLU） | PPL 对比表 + 曲线 |

## 运行

```bash
bash scripts/fetch_data.sh
python -m diy_llm.tokenizer.train_bpe --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000
python -m diy_llm.data --input data/TinyStoriesV2-GPT4-train.txt   # 打包 train.bin
python -m diy_llm.train --config configs/small.yaml
```

## 验收

- [ ] val loss 稳定下降，PPL 曲线存 `figures/`
- [ ] 消融表：baseline vs no-RMSNorm vs post-norm vs SiLU，至少 3 个 seed 或说明只跑 1 个的理由
- [ ] 生成样例：同一 prompt 在不同 temperature/top-p 下的对比
- [ ] `writeup.md` 回答完讲义里的书面题
