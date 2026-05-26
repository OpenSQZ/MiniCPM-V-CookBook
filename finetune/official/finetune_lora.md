### LoRA finetuning

The LoRA allows light-weight model tuning with only a small subset of parameters updated. We provide the LoRA implementation based on `peft`. To launch your training, run the following script:

You can find and review the training script here: [finetune_lora.sh](./finetune_lora.sh)

```shell
MODEL="MiniCPM-o-2_6" # or "openbmb/MiniCPM-V-4", "openbmb/MiniCPM-V-2_6", "openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-2"
DATA="path/to/trainging_data" # json file
EVAL_DATA="path/to/test_data" # json file
LLM_TYPE="qwen" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",
# if use openbmb/MiniCPM-o-2_6 or openbmb/MiniCPM-V-2_6, please set LLM_TYPE=qwen
# if use openbmb/MiniCPM-V-4, please set LLM_TYPE=llama
```

To launch your training, run the following script:
```
sh finetune_lora.sh
```

After training, you could load the model with the path to the adapter. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load.

```python
from peft import PeftModel
from transformers import AutoModel
model_type=  "openbmb/MiniCPM-o-2_6" # or "openbmb/MiniCPM-V-4", "openbmb/MiniCPM-V-2_6", "openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-2"
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