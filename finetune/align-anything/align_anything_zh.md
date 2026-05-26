# Align-Anything

:::{Note}
**支持版本：** MiniCPM-o 2.6
:::

## 环境准备

```bash
# 克隆仓库
git clone git@github.com:PKU-Alignment/align-anything.git
cd align-anything

# 创建虚拟环境
conda create -n align-anything python==3.11
conda activate align-anything
```

**Nvidia GPU 上**

- **`[可选]`** 推荐在 conda 环境中安装 [CUDA](https://anaconda.org/nvidia/cuda) 并设置环境变量。

```bash
# 我们在 H800 计算集群上做了测试，下面这个 CUDA 版本工作良好。
# 可以根据实际计算集群情况调整版本。

conda install nvidia/label/cuda-12.2.0::cuda
export CUDA_HOME=$CONDA_PREFIX
```

> 如果 CUDA 装在别的位置（比如 `/usr/local/cuda/bin/nvcc`），按下面方式设置环境变量：

```bash
export CUDA_HOME="/usr/local/cuda"
```

最后安装 `align-anything`：

```bash
pip3 install -e .

pip3 install vllm==0.7.2 # 在 vllm 引擎上跑 ppo 时需要
```

## 训练

`./scripts/minicpmo` 目录下提供了 SFT 和 DPO 的训练脚本。这些脚本会自动下载模型与数据集，并启动训练或评测。

例如 `scripts/minicpmo/minicpmo_dpo_vision.sh` 是 `Text + Image -> Text` 模态的训练脚本，运行：

```bash
cd scripts
bash minicpmo/minicpmo_dpo_vision.sh
```

**注意：** 脚本会自动从 huggingface 下载模型与数据集。如果国内访问受限，可以使用 `HF Mirror`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```
