#!/usr/bin/env bash
# 下载 TinyStories 训练语料到 data/（需要已 pip install huggingface_hub 或使用 hf 命令）
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

if command -v hf >/dev/null 2>&1; then
  hf download roneneldan/TinyStories TinyStoriesV2-GPT4-train.txt --repo-type dataset --local-dir data
else
  python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"]:
    print(hf_hub_download("roneneldan/TinyStories", f, repo_type="dataset", local_dir="data"))
PY
fi
ls -lh data
