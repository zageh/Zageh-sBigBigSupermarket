"""ch11 / 作业4：文本 -> token bin 打包、memmap 读取、随机采样 batch。

计划：
    tokenize_file_to_bin(txt_path, tokenizer, out_bin, eot_id)
    MemmapDataset(x: np.uint16, ctx_len)  -> 随机起点切 (x, y)
    get_batch(...)  # 训练时用
"""

# TODO(ch11): 先写 bin 打包，再写 Dataset
