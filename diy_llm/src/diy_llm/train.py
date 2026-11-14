"""ch4/ch13：训练循环（grad clip、验证、ckpt、曲线、wandb）。"""

# TODO(ch4): 最小闭环 train(cfg) -> 保存 ckpt/epoch_N.pt + outputs/ppl.png
import argparse
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import psutil  # 需要pip install psutil
import torch
from diy_llm.model.transformer import *
# 训练前环境初始化（防止OMP冲突）
import os
from transformers import PreTrainedTokenizerFast

import wandb

# 允许重复的OpenMP runtime（防止libiomp5md.dll冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 限制OpenMP线程数，防止多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


wandb.init(
    project= "diy_llm",
    entity= "zhaangjunhao-northeastern-university-china--org",
    name="first_self_train_model",
    config={
        "model": "self_train_model",
        "dataset_tag": "raw",
        "batch_size": 64,
        "max_examples": "1000",
        "seed": 2026,
    }
)

def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        progress = (t - T_w) / (T_w - T_c)
        cosine_out = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine_out * (alpha_max - alpha_min)
    else:
        return alpha_min