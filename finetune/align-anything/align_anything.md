# Align-Anything

:::{Note}
**Support:** MiniCPM-o 2.6
:::

## Environment Setup

```bash
# clone the repository
git clone git@github.com:PKU-Alignment/align-anything.git
cd align-anything

# create virtual env
conda create -n align-anything python==3.11
conda activate align-anything
```

**On Nvidia GPU**

- **`[Optional]`** We recommend installing [CUDA](https://anaconda.org/nvidia/cuda) in the conda environment and set the environment variable.

```bash
# We tested on the H800 computing cluster, and this version of CUDA works well.
# You can adjust this version according to the actual situation of the computing cluster.

conda install nvidia/label/cuda-12.2.0::cuda
export CUDA_HOME=$CONDA_PREFIX
```

> If your CUDA installed in a different location, such as `/usr/local/cuda/bin/nvcc`, you can set the environment variables as follows:

```bash
export CUDA_HOME="/usr/local/cuda"
```

Finally, install `align-anything` by:

```bash
pip3 install -e .

pip3 install vllm==0.7.2 # to run ppo on vllm engine
```

## Training

You can find SFT & DPO training script in the `./scripts/minicpmo` directory. These scripts would automatically download the model and dataset, and run the training or evaluation.

For example, `scripts/minicpmo/minicpmo_dpo_vision.sh` is the script for `Text + Image -> Text` modality, you can run it by:

```bash
cd scripts
bash minicpmo/minicpmo_dpo_vision.sh
```

**Note:** The scripts will automatically download the model and dataset from huggingface. If you are prohibited from the internet, please try to use the `HF Mirror`:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

