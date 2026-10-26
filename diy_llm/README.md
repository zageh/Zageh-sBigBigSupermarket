# diy-llm 学习工作区

跟着 [Datawhale diy-llm](https://datawhalechina.github.io/diy-llm/)（CS336 中文版）从零手搓一个 LLM。
本仓库是**我自己的代码/实验/笔记**，与上游课程仓库解耦：上游只读放在 `ref/`，答案型代码不进主线。

- 课程在线文档：https://datawhalechina.github.io/diy-llm/
- 上游仓库：https://github.com/datawhalechina/diy-llm
- 原版课程：https://cs336.stanford.edu/

---

## 0. 约定

| 目录 | 放什么 | 是否入库 |
| --- | --- | --- |
| `src/diy_llm/` | **自己写的可复用实现**（分词器、Transformer、优化器、训练循环…），随课程推进不断长大，最后就是"我的 LLM 包" | ✅ |
| `experiments/chXX_*/` | 每章的一次性验证脚本 / notebook / 小实验（对应文档里的小练习） | ✅ |
| `assignments/aN_*/` | 6 个**作业的交付物**：消融、结果图、writeup、code.zip | ✅（结果大文件除外） |
| `tests/` | 自己写的单测（形状、round-trip、mask 正确性） | ✅ |
| `notes/` | 每章笔记，命名 `chXX_topic.md` | ✅ |
| `data/` | 语料、tokenizer.json、data.bin | ❌ |
| `ckpt/` | 模型权重 | ❌ |
| `outputs/` | 训练日志、PPL 图、评测 jsonl | ❌ |
| `ref/` | `git clone` 的上游仓库，**只读，禁止直接改动** | ❌ |

> 原则：**先自己写，卡 30 分钟再看 `ref/`**。上游 `coursework/` 里是完整参考答案，直接抄会废掉这门课。

---

## 1. 项目结构

```text
diy_llm/
├── README.md                  # 本文件：路线 + 进度
├── pyproject.toml             # 本地包 diy_llm，pip install -e .
├── .gitignore
├── src/diy_llm/               # ★ 主线代码，所有章节的实现都沉淀在这
│   ├── tokenizer/             #   ch2  作业1-A：BPE 训练 + 编解码
│   │   ├── bpe.py
│   │   └── train_bpe.py
│   ├── model/                 #   ch4  作业1-B：RMSNorm / RoPE / Attention / SwiGLU / Block
│   │   └── transformer.py
│   ├── optimizer.py           #   ch3/ch4：AdamW + warmup + cosine
│   ├── data.py                #   ch11：分块读文本、token 打包 .bin、memmap Dataset
│   ├── train.py               #   ch4/ch13：训练循环、验证、ckpt、曲线
│   ├── generate.py            #   ch10：KV cache、temperature/top-k/top-p 采样
│   ├── kernels/               #   ch7  作业2：Triton FlashAttention-2
│   ├── parallel/              #   ch8  作业2：DDP / 分片优化器
│   └── alignment/             #   ch13/14 作业5：SFT、GRPO
├── experiments/               # 按章的小练习和基准脚本
│   ├── ch01_wandb/            #   ch1：wandb 入门
│   ├── ch02_tokenizer/        #   ch2：分词器对比 demo（原 tokenizer.py 已移入）
│   ├── ch03_pytorch/          #   ch3：训练原语、FLOPs/显存核算
│   ├── ch05_moe/              #   ch5：MoE 路由玩具实现
│   ├── ch07_triton/           #   ch7：kernel 正确性测试 + 计时
│   ├── ch09_scaling/          #   ch9：拟合 scaling law、isoFLOP 曲线
│   ├── ch10_inference/        #   ch10：推理优化对比
│   ├── ch11_data/             #   ch11：过滤/去重流水线（作业4 主体）
│   └── ch12_eval/             #   ch12：lm-evaluation-harness / evalscope
├── assignments/               # ★ 6 个作业的交付物
│   ├── a1_basics/             #   分词器 + 架构 + 训练 + 3 个消融
│   ├── a2_systems/            #   profiling + Triton FA2 + 分布式
│   ├── a3_scaling/            #   多规模训练 + 拟合 + 预测
│   ├── a4_data/               #   Common Crawl → 预训练集
│   ├── a5_alignment/          #   SFT / EI / GRPO
│   └── a6_evaluation/         #   多维度评测报告
├── tests/                     # pytest 单测
├── notes/                     # 每章笔记
├── scripts/                   # 一键脚本（装环境、下数据）
├── data/  ckpt/  outputs/     # 运行时产物（git 忽略）
└── ref/diy-llm/               # 上游仓库 clone（只读）
```

**为什么这样分？** 课程章节（`docs/`）和作业（`coursework/`）是两条线：一章文档可能对应半个作业，一个作业会横跨好几章。所以：
- **能力**沉淀在 `src/diy_llm/`（复用它，不复制粘贴）；
- **证据**沉淀在 `assignments/aN/`（writeup + 图 + 结果）；
- **学习痕迹**沉淀在 `experiments/` + `notes/`。

---

## 2. 路线图（建议 10 周，每周 8–10h）

### 阶段 0：环境 + 跑通终点（1 天）

```bash
# 1) 上游参考代码（只读）
git clone --depth 1 https://github.com/datawhalechina/diy-llm.git ref/diy-llm

# 2) 环境（conda 或 venv 均可，Python 3.11）
conda create -n diy-llm python=3.11 -y && conda activate diy-llm
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50 系需要 cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3) 先看"终点"长什么样：跑一遍参考答案（小 epoch）
cd ref/diy-llm/coursework/assignment1-basics && python train.py --epochs 1
```

验收：`torch.cuda.is_available()` 为 `True`，且能看到参考实现的训练日志和 `ckpt/`。
（WSL2 下如果报 `CUDA error 304`，在普通终端而不是受限沙箱里重试；仍失败先单独修驱动，别往下走。）

### 阶段 1（第 1–2 周）ch1 工具 + ch2 分词器 → 作业1-A

| 做什么 | 代码写在哪 | 验收标准 |
| --- | --- | --- |
| ch1：wandb 记录一次 toy 训练 | `experiments/ch01_wandb/train_toy.py` | wandb 上能看到 loss 曲线 |
| ch2：手写 BPE 训练（不调 `tokenizers` 的 `train`） | `src/diy_llm/tokenizer/bpe.py` | 在 TinyStories 10MB 上训出 10k 词表，`encode→decode` 完全还原，与 `tokenizers` 库结果对比一致 |
| ch2：分块读大文件 + 内存/吞吐监控 | `src/diy_llm/tokenizer/train_bpe.py` | 打印 peak RSS 与 tokens/s |
| 作业1-A：BPE 相关问答 | `assignments/a1_basics/writeup.md` | 回答完 notebook 里的思考题 |

笔记：`notes/ch02_tokenizer.md`（BPE 合并规则、`Ġ` 的由来、压缩率对比表）。

### 阶段 2（第 2–4 周）ch3 资源核算 + ch4 架构 → 作业1-B

| 做什么 | 代码写在哪 | 验收标准 |
| --- | --- | --- |
| ch3：FLOPs / 显存手算，和实测对齐 | `experiments/ch03_pytorch/` | 预测值与 `torch.cuda.max_memory_allocated()` 同量级 |
| RMSNorm / RoPE / SwiGLU / Attention / Block | `src/diy_llm/model/transformer.py` | `tests/test_model.py`：形状、causal mask、RoPE 相对位置性质 |
| AdamW + warmup + cosine | `src/diy_llm/optimizer.py` | 和 `torch.optim.AdamW` 数值对齐（小规模） |
| 训练循环（grad clip、验证、ckpt、曲线） | `src/diy_llm/train.py` | val loss 单调下降，画出 PPL 曲线 |
| 采样生成 | `src/diy_llm/generate.py` | 输入 prompt 能续写出通顺英文小故事 |
| 3 个消融：去 RMSNorm / post-norm / SiLU 替 SwiGLU | `assignments/a1_basics/ablations/` | 3 组 PPL 对比表 + 曲线，写进 writeup |

**作业1 完成标志**：`assignments/a1_basics/` 下有 `writeup.md`（含 PPL 表、消融结论、生成样例）+ 架构图。
这份代码后面作业 3/5 直接复用 —— 所以别写死超参，做成 config/dataclass。

### 阶段 3（第 4–6 周）ch6/7/8 系统 → 作业2

- `src/diy_llm/kernels/flash_attn_triton.py`：Triton 版 FlashAttention-2（重点，8GB 卡可以跑）
- `experiments/ch07_triton/bench.py`：对比 naive / `F.scaled_dot_product_attention` / 自己写的，记录显存与延迟
- `src/diy_llm/parallel/`：DDP（flat / bucketed）、分片优化器（ZeRO-1 思想）
- 单卡机器：DDP 用 **2 进程 CPU + gloo** 跑通逻辑；真多卡跑一次 benchmark（可租 2×4090 一两小时）
- 自测：`ref/diy-llm/coursework/assignment2-systems/tests/` 的 `adapters.py` 填自己的实现跑 pytest

### 阶段 4（第 6–7 周）ch9 Scaling Laws → 作业3

- `experiments/ch09_scaling/`：5M/10M/20M/40M 四个规模 × 不同 token 预算，记录 loss
- 拟合 `L(N,D) = E + A/N^α + B/D^β`，画 isoFLOP 曲线，预测更大规模并验证
- 8GB 建议：`d_model ≤ 256`、`ctx 256`，每个 run ≤ 30 min，共约 16 个 run

### 阶段 5（第 7–8 周）ch11 数据工程 → 作业4

- `experiments/ch11_data/`：Common Crawl 抽取 → 语言过滤 → 质量过滤（Gopher 规则）→ MinHash 去重 → 打 `.bin`
- 纯 CPU，和显卡无关；重点是**每条规则都要有前后对比统计**
- 交付：`assignments/a4_data/report.md`（去重率、过滤率、样本抽查）

### 阶段 6（第 8–9 周）ch13 训练流程 + ch14 RLVR → 作业5

- `src/diy_llm/alignment/sft.py`：`tokenize_prompt_and_output`、masked loss、packing
- `src/diy_llm/alignment/grpo.py`：group-normalized reward、KL、裁剪、长度归一化
- **8GB 现实做法**：`Qwen2.5-0.5B-Instruct` + LoRA(peft) + GSM8K 子集；GRPO 只跑 1–2 组消融
- 想要完整曲线就租 1×A100/4090（几小时），用上游 README 里的 wandb 报告对照

### 阶段 7（第 9–10 周）ch10 推理 + ch12 评估 → 作业6

- `experiments/ch10_inference/`：KV cache 有无、batch、量化（int8/int4）的延迟-显存-质量三角
- `experiments/ch12_eval/`：`lm-evaluation-harness` + `evalscope` 跑自己训的小模型 / Qwen，出多维度报告
- 交付：`assignments/a6_evaluation/report.md`（语言理解 / 常识 / 代码 / 数学 四类指标表 + 误差分析）

### 选修（有余力再看）

ch5 MoE（`experiments/ch05_moe/` 写个 top-k 路由 + 负载均衡 loss）、ch15 多模态、ch16 扩展阅读。

---

## 3. 每周节奏

1. 读对应章节文档（1–2h），把不懂的点写进 `notes/chXX_*.md` 的"问题"小节；
2. 写 `src/` 里的实现（3–5h），先让 `tests/` 绿；
3. 跑 `experiments/` 验证（1–2h），**存图到 `outputs/`**；
4. 更新 `assignments/aN/writeup.md`（1h）—— 这步别省，期末汇报/给学长看的就是它。

---

## 4. 提交规范（每个作业）

```text
assignments/aN_xxx/
├── README.md      # 一句话：这作业做了什么、怎么跑
├── writeup.md     # 必答题 + 实验结论 + 图（图放 figures/）
├── figures/
└── code/          # 可选：交付用快照（或直接说明引用 src/ 哪些模块）
```

课程要求：`writeup.pdf`（书面题）+ `code.zip`（实现）。本地用 `writeup.md` 即可，需要交时再导 PDF。

---

## 5. 硬件现实（RTX 5060 Laptop 8GB）

| 作业 | 8GB 能不能做 | 备注 |
| --- | --- | --- |
| 作业1 手搓模型 | ✅ 完全够 | `d_model 256 / 6 层 / ctx 128`，约 10M 参数 |
| 作业2 系统 | 🟡 部分 | Triton FA2 能跑；DDP 单卡用 CPU 2 进程模拟，多卡上云 |
| 作业3 Scaling | 🟡 慢但能 | 小模型 + 短上下文，注意总时长 |
| 作业4 数据 | ✅ CPU 任务 | 主要吃内存和磁盘 |
| 作业5 对齐 | ❌ 完整版 / ✅ 缩水版 | 0.5B + LoRA 本地；完整 SFT/GRPO 需 40–80GB |
| 作业6 评估 | ✅ | 小模型/API 评测，别本地跑大模型 |

---

## 6. 进度

- [ ] 阶段 0 环境 + 跑通参考实现
- [ ] ch1 wandb
- [ ] ch2 分词器（作业1-A）
- [ ] ch3 资源核算
- [ ] ch4 架构 + 训练（作业1-B）
- [ ] 作业1 消融 + writeup
- [ ] ch6/7/8 系统（作业2）
- [ ] ch9 Scaling（作业3）
- [ ] ch11 数据（作业4）
- [ ] ch13/14 对齐（作业5）
- [ ] ch10/12 推理评估（作业6）
- [ ] 选修 ch5 MoE / ch15 多模态
