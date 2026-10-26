# MiniCPM-V 4.7 — llama.cpp

> [!NOTE]
> MiniCPM-V 4.7 已合入官方 `llama.cpp`（[PR #29416](https://github.com/ggml-org/llama.cpp/pull/29416)）。
> 请使用该 PR 之后（含）的 checkout 构建。
>
> **GGUF 格式变化：** 4.7 使用 canvas M-RoPE，新增了 GGUF 元数据键
> （`rope.section_order`）与 mmproj 张量（`v.tok_embd_sep`）。用**旧版** `convert_hf_to_gguf.py`
> 转出的 GGUF，在新构建里会加载成功但 **M-RoPE 计算错误**（或因缺 mmproj 张量直接失败）。
> 请始终使用[官方 4.7 GGUF](#2-gguf-文件)，或用当前脚本重新转换。

MiniCPM-V 4.7 提供两个尺寸，使用对应的一套 GGUF：

| 尺寸 | LLM 文件 | mmproj |
|---|---|---|
| 1B | `MiniCPM-V-4_7-1B-Q4_K_M.gguf` | `mmproj-model-f16.gguf` |
| 35B-A3B | `MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` | `mmproj-model-f16.gguf` |

## 1. 构建 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

用 CMake 构建（详见 [build 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)）：

**CPU / Metal：**

```bash
cmake -B build
cmake --build build --config Release
```

**CUDA：**

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

## 2. GGUF 文件

### 方式一：下载官方 GGUF

从以下地址下载语言模型 GGUF 与视觉投影（`mmproj`）：

- HuggingFace：
  - 1B — <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf>
  - 35B-A3B — <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf>
- ModelScope — `OpenBMB/MiniCPM-V-4.7-1B-gguf` / `OpenBMB/MiniCPM-V-4.7-35B-A3B-gguf`

每个仓库提供完整的 K-quant 档位（`F16`、`Q8_0`、`Q6_K`、`Q5_K_M`、`Q5_K_S`、`Q5_1`、`Q5_0`、`Q4_K_M`、`Q4_K_S`、`Q4_1`、`Q4_0`）以及两个 mmproj（`mmproj-model-f16.gguf`、`mmproj-model-f16-4x.gguf`）。35B-A3B 仓库额外提供 `mix_q3gud_im`（14.4 GiB 的 imatrix 混合量化，推荐用于显存受限设备）。

### 方式二：从 PyTorch 模型转换

下载 PyTorch 权重（如 `openbmb/MiniCPM-V-4.7-1B`），在**当前版本**的 llama.cpp 里运行标准脚本：

```bash
# 1) 转换语言模型 + 视觉 merger
python ./convert_hf_to_gguf.py /path/to/MiniCPM-V-4.7-1B \
    --outfile /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-F16.gguf \
    --outtype f16

# 2) 转换视觉投影（mmproj）
python ./convert_hf_to_gguf.py /path/to/MiniCPM-V-4.7-1B \
    --mmproj \
    --outfile /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf
```

导出 4× 视觉变体：复制模型目录，把其 `preprocessor_config.json` 里的 `"downsample_mode"` 改成 `"4x"`，再跑一次 `--mmproj`，`--outfile mmproj-model-f16-4x.gguf`。

## 3. 模型推理

> [!IMPORTANT]
> `llama.cpp` 会**无条件注入** `enable_thinking`（其自身默认值为 `true`），所以决定是否思考的是运行时，
> 而不是模板的默认值。仓库**根目录**的 GGUF 已在**内嵌模板里硬关闭思考**，因此**默认就是普通问答，
> 无需任何参数**，传 `--reasoning on` 也不会生效。若需要思考版本，请使用 `thinking/` 子目录，
> 见[思考模式](#思考模式)。

```bash
cd build/bin/

# F16 权重（1B）
./llama-mtmd-cli \
    -m  /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-F16.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 \
    --reasoning off \
    --image xx.jpg -p "What is in the image?"

# 量化 INT4 权重（35B-A3B）
./llama-mtmd-cli \
    -m  /path/to/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-35B-A3B/mmproj-model-f16.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 \
    --reasoning off \
    --image xx.jpg -p "What is in the image?"

# 交互模式
./llama-mtmd-cli \
    -m  /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 \
    --reasoning off \
    --image xx.jpg -i
```

### 思考模式

仓库**根目录**的 GGUF 无法思考：内嵌模板始终输出空的 think 块，无论 `--reasoning` 怎么设都直接作答。
要获得思考轨迹，有两种方式：

1. **使用 `thinking/` 版本**（推荐）。每个 GGUF 仓库在 `thinking/` 下提供了同样的量化档，文件名带
   `-thinking` 后缀；这些文件恢复了上游官方模板，除内嵌模板外与根目录文件完全一致：

   ```bash
   ./llama-cli -m thinking/MiniCPM-V-4_7-1B-Q4_K_M-thinking.gguf \
               --mmproj mmproj-model-f16.gguf --image xx.jpg -p "图里是什么？" -n 1024
   ```

   该版本默认开启思考；传 `--reasoning off` 可关闭。

2. **使用 Transformers 权重**并传 `enable_thinking=True`，或用未修改的模板自行转换 GGUF。

> 该模型并非专门针对推理训练——思考轨迹便于查看中间步骤，但会成倍增加生成的 token 数，
> 请相应调大 `-n`。

作为参考，`--jinja` 默认已开，对应参数为 `--reasoning on`：

```bash
./llama-mtmd-cli \
    -m  /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf \
    -c 8192 --image xx.jpg --jinja --reasoning on -p "What is in the image?"
```

**参数参考：**

| 参数 | 说明 |
| :--- | :--- |
| `-m, --model` | 语言模型 GGUF 路径 |
| `--mmproj` | 视觉投影 GGUF 路径 |
| `--image` | 输入图片路径 |
| `-p, --prompt` | 提示词 |
| `-c, --ctx-size` | 最大上下文长度 |
| `-rea, --reasoning [on\|off\|auto]` | 是否开启思考。默认 `auto`。**本仓库 GGUF 已硬关闭思考，该参数不生效** |
| `--jinja` | 使用模型 Jinja 对话模板（近期构建默认开启） |

## 4. 使用 DSpark 投机解码

MiniCPM-V 4.7 提供了 **DSpark** 草稿模型，可通过投机解码加速生成。请下载与目标模型规格匹配的 draft GGUF：

- HuggingFace：`openbmb/MiniCPM-V-4.7-1B-DSpark-gguf` / `openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf`
- ModelScope：`OpenBMB/MiniCPM-V-4.7-1B-DSpark-gguf` / `OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark-gguf`

两个仓库提供的量化档与目标模型一致（`F16`、`Q8_0`、`Q6_K`、`Q5_K_M`、`Q5_K_S`、`Q5_1`、`Q5_0`、`Q4_K_M`、`Q4_K_S`、`Q4_1`、`Q4_0`）。投机解码是无损的——每个候选 token 都由目标模型校验——因此 draft 量化只影响接受长度（速度），不影响输出；选显存放得下的最小档即可，`Q8_0` / `Q6_K` 是稳妥的默认值。

> [!IMPORTANT]
> `llama-mtmd-cli` **不支持** draft 模型。请使用 `llama-cli`（同时支持视觉投影与 draft）或 `llama-server`。

```bash
cd build/bin/

# 图像 + 投机解码
./llama-cli \
    -m  /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf \
    -md /path/to/MiniCPM-V-4.7-1B-DSpark-gguf/dspark-MiniCPM-V-4_7-1B-Q8_0.gguf \
    --spec-type draft-dspark \
    -c 8192 --reasoning off \
    --image xx.jpg -p "What is in the image?"

# 或使用服务模式
./llama-server \
    -m  /path/to/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-35B-A3B/mmproj-model-f16.gguf \
    -md /path/to/MiniCPM-V-4.7-35B-A3B-DSpark-gguf/dspark-MiniCPM-V-4_7-35B-A3B-Q8_0.gguf \
    --spec-type draft-dspark --spec-draft-n-max 15 \
    -c 8192 --jinja --port 8080
```

draft 与目标模型一一配对，务必使用与目标规格同名的 draft。`llama-server` 会在每次响应中报告接受长度（`draft_n` / `draft_n_accepted`）。

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
