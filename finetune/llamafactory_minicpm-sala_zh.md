# LLaMA-Factory（MiniCPM-SALA）

> [!NOTE]
> MiniCPM-SALA 微调使用 LLaMA-Factory，流程与 MiniCPM 4 基本一致，但需要 **支持 SALA 的 LLaMA-Factory 构建** 以正确识别稀疏 + 线性混合注意力层。与 [MiniCPM 4 指南](llamafactory.html)的主要差异在下文标出。

## 1. 安装 LLaMA-Factory 与 SALA 依赖

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed]"
```

同时安装 SALA 稀疏注意力 kernel（训练和推理都需要）：

```bash
git clone https://github.com/OpenBMB/infllmv2_cuda_impl.git
cd infllmv2_cuda_impl
pip install -e .
```

> [!IMPORTANT]
> SALA 建模代码通过 `trust_remote_code` 从 Hub 加载。如果当前 LLaMA-Factory 版本不识别 `MiniCPMSALAForCausalLM` 架构，可退回标准的 Transformers Trainer 配方（[SALA 技术报告 PDF](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf) 中有官方发布使用的训练超参）。

## 2. 准备数据

SALA 的核心优势是长上下文，所以数据应当真正用到长上下文。sharegpt 样例（`data/long_sft.json`）：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "阅读这篇 5 万令牌的文档并回答末尾的问题。\n\n<长文档>...\n\n问题：第三节的主要观点是什么？"},
      {"from": "gpt", "value": "第三节的主要观点是..."}
    ]
  }
]
```

在 `data/dataset_info.json` 中注册即可。

## 3. LoRA SFT 配置

`configs/lora_minicpm_sala.yaml`：

```yaml
### 模型
model_name_or_path: openbmb/MiniCPM-SALA
trust_remote_code: true

### 方法
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### 数据
dataset: long_sft
template: minicpm4              # SALA 继承自 MiniCPM 4 的 chat 模板
cutoff_len: 65536               # 调大以真正训练到长上下文层
overwrite_cache: true

### 输出
output_dir: saves/sala-lora
save_steps: 100

### 训练
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 1.0
bf16: true

### SALA 专用
flash_attn: fa2
use_unsloth: false
```

```bash
llamafactory-cli train configs/lora_minicpm_sala.yaml
```

## 4. 注意事项

- SALA 是**研究 checkpoint**，不同版本之间微调配方可能不能完全复现。需要稳定性请 pin 住模型与 kernel commit。
- 训练阶段也依赖 InfLLM-V2 稀疏注意力 kernel。缺失时训练会回落到很慢的参考实现，无法体现 SALA 的设计意图。
- 微调出的 adapter 推理时，可走 [SALA SGLang 指南](../deployment/sglang.html)（先合并 checkpoint）或直接通过 HF Transformers 加载。
