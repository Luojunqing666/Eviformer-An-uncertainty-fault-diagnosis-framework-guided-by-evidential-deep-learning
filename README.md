# Eviformer

**An Uncertainty Fault Diagnosis Framework Guided by Evidential Deep Learning**

[![Paper](https://img.shields.io/badge/Paper-Engineering%20Applications%20of%20AI-blue)](https://doi.org/10.1016/j.engappai.2025.112328)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Jingjie Luo, Fucai Li*, Xiaolei Xu, Wenqiang Zhao, Dongqing Zhang
>
> *Engineering Applications of Artificial Intelligence*, Volume 161, 2025, 112328

---

## Overview

Eviformer is a fault diagnosis framework that integrates **Evidential Deep Learning (EDL)** with Transformer-based architectures to provide both classification predictions and uncertainty quantification for mechanical fault diagnosis. The framework enables reliable decision-making by quantifying epistemic uncertainty through Dirichlet distribution parameterization.

<div align="center">
<img src="figures/overall_procedure.png" width="700" />
<p><em>The overall procedure of the proposed method</em></p>
</div>

<div align="center">
<img src="figures/distribution_measurer.png" width="600" />
<p><em>Schematic diagram of the Distribution Measurer</em></p>
</div>

## Key Features

- **Evidential Deep Learning**: Quantifies prediction uncertainty via Dirichlet distribution parameters
- **Multiple Loss Functions**: Supports EDL-MSE, EDL-Digamma, and EDL-Log loss variants
- **Flexible Architectures**: Includes 1D-ViT, MC-SwinT, and LeNet backbones for vibration signals
- **Modular Design**: Clean separation of models, datasets, and training logic

## Project Structure

```
Eviformer/
├── main.py                  # Entry point (train / test)
├── train.py                 # Training loop
├── requirements.txt         # Dependencies
├── models/
│   ├── __init__.py
│   ├── losses.py            # EDL loss functions (MSE, Digamma, Log)
│   ├── vit.py               # 1D Vision Transformer
│   ├── mcswint.py           # MC-Swin Transformer
│   └── lenet.py             # 1D LeNet baseline
├── datasets/
│   ├── __init__.py
│   ├── cwru.py              # CWRU bearing dataset loader
│   ├── sequence_dataset.py  # Generic sequence dataset
│   └── transforms.py        # Data augmentation transforms
├── utils/
│   └── __init__.py          # Helpers (device, one-hot encoding)
├── results/                 # Saved checkpoints (gitignored)
└── figures/                 # Paper figures
```

## Installation

```bash
git clone https://github.com/Luojunqing666/Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning.git
cd Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning/Eviformer
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- NumPy >= 1.22
- SciPy >= 1.7
- scikit-learn >= 1.0

## Usage

### Training with Evidential Deep Learning

```bash
# Train MC-SwinT with EDL Digamma loss (recommended)
python main.py --train --model mcswint --uncertainty --digamma \
    --data_dir /path/to/CWRU --num_classes 10 --epochs 100

# Train 1D-ViT with EDL Log loss
python main.py --train --model vit --uncertainty --log \
    --data_dir /path/to/CWRU --num_classes 10 --epochs 100

# Train LeNet with EDL MSE loss
python main.py --train --model lenet --uncertainty --mse \
    --data_dir /path/to/CWRU --num_classes 10 --epochs 100
```

### Training without Uncertainty (Standard Cross-Entropy)

```bash
python main.py --train --model mcswint \
    --data_dir /path/to/CWRU --num_classes 10 --epochs 100
```

### Testing

```bash
python main.py --test --model mcswint --uncertainty --digamma \
    --data_dir /path/to/CWRU --num_classes 10
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `mcswint` | Model architecture: `lenet`, `vit`, `mcswint` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--num_classes` | `10` | Number of fault classes |
| `--uncertainty` | `False` | Enable evidential deep learning |
| `--digamma` | - | Use EDL Digamma loss |
| `--log` | - | Use EDL Log loss |
| `--mse` | - | Use EDL MSE loss |
| `--data_dir` | - | Path to dataset root directory |
| `--normalize` | `0-1` | Normalization: `0-1`, `-1-1`, `mean-std` |

## Dataset

This code uses the [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter) by default. Any publicly available bearing or gear vibration signal dataset can be used with this framework by implementing a custom data loader following the pattern in `datasets/cwru.py`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{luo2025eviformer,
  title     = {Eviformer: An uncertainty fault diagnosis framework guided by evidential deep learning},
  author    = {Luo, Jingjie and Li, Fucai and Xu, Xiaolei and Zhao, Wenqiang and Zhang, Dongqing},
  journal   = {Engineering Applications of Artificial Intelligence},
  volume    = {161},
  pages     = {112328},
  year      = {2025},
  doi       = {10.1016/j.engappai.2025.112328},
  publisher = {Elsevier}
}
```

## Contact

- luojingjie@hnu.edu.cn
- luojingjie@sjtu.edu.cn

---

# Eviformer

**基于证据深度学习引导的不确定性故障诊断框架**

[![论文](https://img.shields.io/badge/论文-Engineering%20Applications%20of%20AI-blue)](https://doi.org/10.1016/j.engappai.2025.112328)
[![许可证](https://img.shields.io/badge/许可证-MIT-green.svg)](LICENSE)

> 罗景杰, 李富才*, 徐晓磊, 赵文强, 张东青
>
> *Engineering Applications of Artificial Intelligence*, 第161卷, 2025, 112328

---

## 概述

Eviformer 是一个将**证据深度学习（EDL）**与 Transformer 架构相结合的故障诊断框架，能够同时提供分类预测和不确定性量化，用于机械故障诊断。该框架通过 Dirichlet 分布参数化来量化认知不确定性，从而实现可靠的决策支持。

<div align="center">
<img src="figures/overall_procedure.png" width="700" />
<p><em>所提方法的整体流程</em></p>
</div>

<div align="center">
<img src="figures/distribution_measurer.png" width="600" />
<p><em>分布度量器示意图</em></p>
</div>

## 主要特点

- **证据深度学习**：通过 Dirichlet 分布参数量化预测不确定性
- **多种损失函数**：支持 EDL-MSE、EDL-Digamma 和 EDL-Log 损失变体
- **灵活的网络架构**：包含 1D-ViT、MC-SwinT 和 LeNet 骨干网络，适用于振动信号
- **模块化设计**：模型、数据集和训练逻辑清晰分离

## 项目结构

```
Eviformer/
├── main.py                  # 入口文件（训练/测试）
├── train.py                 # 训练循环
├── requirements.txt         # 依赖包
├── models/
│   ├── __init__.py
│   ├── losses.py            # EDL 损失函数（MSE、Digamma、Log）
│   ├── vit.py               # 一维 Vision Transformer
│   ├── mcswint.py           # MC-Swin Transformer
│   └── lenet.py             # 一维 LeNet 基线模型
├── datasets/
│   ├── __init__.py
│   ├── cwru.py              # CWRU 轴承数据集加载器
│   ├── sequence_dataset.py  # 通用序列数据集
│   └── transforms.py        # 数据增强变换
├── utils/
│   └── __init__.py          # 工具函数（设备选择、独热编码）
├── results/                 # 保存的模型检查点（已忽略）
└── figures/                 # 论文图片
```

## 安装

```bash
git clone https://github.com/Luojunqing666/Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning.git
cd Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning/Eviformer
pip install -r requirements.txt
```

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- NumPy >= 1.22
- SciPy >= 1.7
- scikit-learn >= 1.0

## 使用方法

### 使用证据深度学习进行训练

```bash
# 使用 MC-SwinT + EDL Digamma 损失训练（推荐）
python main.py --train --model mcswint --uncertainty --digamma \
    --data_dir /path/to/CWRU --num_classes 10 --epochs 100

# 使用 1D-ViT + EDL Log 损失训练
python main.py --train --model vit --uncertainty --log \
    --data_dir /path/to/CWRU --num_classes 10 --epochs 100

# 使用 LeNet + EDL MSE 损失训练
python main.py --train --model lenet --uncertainty --mse \
    --data_dir /path/to/CWRU --num_classes 10 --epochs 100
```

### 不使用不确定性的标准训练（交叉熵损失）

```bash
python main.py --train --model mcswint \
    --data_dir /path/to/CWRU --num_classes 10 --epochs 100
```

### 测试

```bash
python main.py --test --model mcswint --uncertainty --digamma \
    --data_dir /path/to/CWRU --num_classes 10
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `mcswint` | 模型架构：`lenet`、`vit`、`mcswint` |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `16` | 批大小 |
| `--lr` | `1e-3` | 学习率 |
| `--num_classes` | `10` | 故障类别数 |
| `--uncertainty` | `False` | 启用证据深度学习 |
| `--digamma` | - | 使用 EDL Digamma 损失 |
| `--log` | - | 使用 EDL Log 损失 |
| `--mse` | - | 使用 EDL MSE 损失 |
| `--data_dir` | - | 数据集根目录路径 |
| `--normalize` | `0-1` | 归一化方式：`0-1`、`-1-1`、`mean-std` |

## 数据集

本代码默认使用 [CWRU 轴承数据集](https://engineering.case.edu/bearingdatacenter)。任何公开的轴承或齿轮振动信号数据集均可使用本框架，只需参照 `datasets/cwru.py` 的模式实现自定义数据加载器即可。

## 引用

如果本工作对您有帮助，请引用以下论文：

```bibtex
@article{luo2025eviformer,
  title     = {Eviformer: An uncertainty fault diagnosis framework guided by evidential deep learning},
  author    = {Luo, Jingjie and Li, Fucai and Xu, Xiaolei and Zhao, Wenqiang and Zhang, Dongqing},
  journal   = {Engineering Applications of Artificial Intelligence},
  volume    = {161},
  pages     = {112328},
  year      = {2025},
  doi       = {10.1016/j.engappai.2025.112328},
  publisher = {Elsevier}
}
```

## 联系方式

- luojingjie@hnu.edu.cn
- luojingjie@sjtu.edu.cn
