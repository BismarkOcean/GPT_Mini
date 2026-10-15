# GPT Mini

基于 Python 和 PyTorch 从零实现的小型 Decoder-only Transformer，采用字符级编码与自回归建模。

目前已实现数据处理与 GPT 前向计算。训练循环、评估、检查点和文本生成仍在开发中，暂未提供训练权重或生成效果。

## 已实现功能

- 字符级 tokenizer、滑动窗口 Dataset 与 DataLoader 数据示例。
- 单头与多头因果自注意力，显式实现缩放、因果掩码和注意力计算。
- Pre-LN Transformer block：残差连接、LayerNorm 与 GELU 前馈网络。
- Token embedding、可学习的位置 embedding、多层 block 与词表输出投影。
- 覆盖数据窗口、因果性、输出形状和梯度连通性的 pytest 用例。

模型使用 PyTorch 基础层与 autograd，自行组装网络，不依赖现成 GPT 或预训练权重。

## 模型结构

```text
Token IDs [B, T]
  → Token embedding + Position embedding [B, T, C]
  → N × Pre-LN Transformer block [B, T, C]
  → LayerNorm → 词表投影
Logits [B, T, V]
```

`B` 为批大小，`T` 为序列长度，`C` 为隐藏维度，`V` 为词表大小。
每个位置只能关注自身及之前的 token；输出为未经 softmax 的 logits。

## 目录

```text
src/gpt_mini/
  data/           字符编码、窗口与 Dataset
  model/          注意力、Transformer block、GPT
  training/       预留：损失与训练
  evaluation/     预留：验证损失与困惑度
  generation/     预留：自回归生成与采样
scripts/          环境诊断、数据示例
tests/            pytest 用例
data/             本地语料（不上传）
artifacts/        本地权重与实验产物（不上传）
```

## 安装与运行

需要 Python 3.11 或更高版本。以下以 Windows、Python 3.13 和 CPU 版 PyTorch 为例，在项目根目录执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[train,dev]"
```

如需 CUDA，请用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 对应的安装命令替换上述 PyTorch 安装步骤，并使用虚拟环境中的 Python 执行。

运行内置数据示例和现有测试：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe -m pytest
```

数据示例使用内联短文本，无需下载语料。模型调用示例：

```python
import torch
from gpt_mini.model.GPT import GPT

model = GPT(
    vocab_size=7,
    max_seq_len=8,
    embed_dim=16,
    num_heads=4,
    num_layers=2,
)
ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
logits = model(ids)
print(logits.shape)  # torch.Size([1, 4, 7])
```

该示例展示随机初始化模型的接口；当前尚无端到端训练或 CUDA 模型训练结果。
