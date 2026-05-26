### 全参数微调

全参数微调会在整个训练过程中更新 LLM 的全部参数。请在 shell 脚本中正确设置 `MODEL`、`DATA` 与 `LLM_TYPE`。

训练脚本可在此查看：[finetune_ds.sh](./finetune_ds.sh)

```shell
MODEL="MiniCPM-o-2_6" # 或 "openbmb/MiniCPM-V-4"、"openbmb/MiniCPM-V-2_6"、"openbmb/MiniCPM-Llama3-V-2_5"、"openbmb/MiniCPM-V-2"
DATA="path/to/trainging_data" # json 文件
EVAL_DATA="path/to/test_data" # json 文件
LLM_TYPE="qwen" # 使用 openbmb/MiniCPM-V-2 时设为 minicpm；使用 openbmb/MiniCPM-Llama3-V-2_5 时设为 "llama3"；
# 使用 openbmb/MiniCPM-o-2_6 或 openbmb/MiniCPM-V-2_6 时设为 qwen；
# 使用 openbmb/MiniCPM-V-4 时设为 llama
```

启动训练：

```
sh finetune_ds.sh
```

训练完成后，可通过 adapter 路径加载模型。建议为预训练模型使用绝对路径，因为 LoRA 仅保存 adapter，adapter config json 中的绝对路径用于定位预训练模型。

```python
from peft import PeftModel
from transformers import AutoModel
model_type=  "openbmb/MiniCPM-o-2_6" # 或 "openbmb/MiniCPM-V-4"、"openbmb/MiniCPM-V-2_6"、"openbmb/MiniCPM-Llama3-V-2_5"、"openbmb/MiniCPM-V-2"
path_to_adapter="path_to_your_fine_tuned_checkpoint"

model =  AutoModel.from_pretrained(
        model_type,
        trust_remote_code=True
        )

lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()
```
