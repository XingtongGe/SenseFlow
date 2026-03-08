# SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation

[![arXiv](https://img.shields.io/badge/Arxiv-2506.00523-b31b1b)](https://arxiv.org/abs/2506.00523)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/domiso/SenseFlow)

[Xingtong Ge](https://xingtongge.github.io/)<sup>1,2</sup>, Xin Zhang<sup>3</sup>, [Tongda Xu](https://tongdaxu.github.io/)<sup>4</sup>, [Yi Zhang](https://zhangyi-3.github.io/)<sup>3</sup>, [Xinjie Zhang](https://xinjie-q.github.io/)<sup>1</sup>, [Yan Wang](https://yanwang202199.github.io/)<sup>4</sup>, [Jun Zhang](https://eejzhang.people.ust.hk/)<sup>1</sup>\*

<sup>1</sup>The Hong Kong University of Science and Technology, <sup>2</sup>SenseTime Research, <sup>3</sup>Vivix AI, <sup>4</sup>Institute for AI Industry Research, Tsinghua University

The Fourteenth International Conference on Learning Representations (**ICLR**), 2026



## 📝 Abstract

The Distribution Matching Distillation (DMD) has been successfully applied to text-to-image diffusion models such as Stable Diffusion (SD) 1.5. However, vanilla DMD suffers from convergence difficulties on large-scale flow-based text-to-image models, such as SD 3.5 and FLUX. In this paper, we first analyze the issues when applying vanilla DMD on large-scale models. Then, to overcome the scalability challenge, we propose implicit distribution alignment (IDA) to regularize the distance between the generator and fake distribution. Furthermore, we propose intra-segment guidance (ISG) to relocate the timestep importance distribution from the teacher model. With IDA alone, DMD converges for SD 3.5; employing both IDA and ISG, DMD converges for SD 3.5 and FLUX.1 dev. Along with other improvements such as scaled up discriminator models, our final model, dubbed **SenseFlow**, achieves superior performance in distillation for both diffusion based text-to-image models such as SDXL, and flow-matching models such as SD 3.5 Large and FLUX. The source code and model weights are now available.

![1024 x 1024 examples of our 4-step generator distilled on FLUX.1 dev](assets/Fig1_final.jpg)

## ✅ TODO List

- [x] Single-node training scripts
- [ ] Multi-node training scripts
- [x] Inference scripts
- [ ] Open-source model weights

## 🤗 Model Weights

We have open-sourced the **SenseFlow-FLUX** model weights on Hugging Face! 🎉

### 📥 Download SenseFlow-FLUX

The SenseFlow-FLUX model (supports 4-8 step generation) is available at:
- **Hugging Face Model**: [domiso/SenseFlow](https://huggingface.co/domiso/SenseFlow)

The model includes:
- `xxx.safetensors`: the DiT checkpoint
- `config.json`: the config of DiT used in our model

### 🚀 Quick Start with SenseFlow-FLUX

1. Download the base FLUX.1-dev checkpoint to `Path/to/FLUX`
2. Download SenseFlow-FLUX from Hugging Face and replace the transformer folder:
   ```bash
   # Replace Path/to/FLUX/transformer with SenseFlow-FLUX folder
   ```
3. Use the model with diffusers (see [Hugging Face model card](https://huggingface.co/domiso/SenseFlow) for detailed usage examples)

## 💻 Installation

We provide two methods to set up the environment: using conda with `environment.yaml` or using pip with `requirements.txt`.

### Option 1: Using Conda (Recommended)

1. Create a new conda environment from the provided `environment.yaml`:
   ```bash
   conda env create -f environment.yaml
   ```

2. Activate the environment:
   ```bash
   conda activate senseflow
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

### Option 2: Using Pip

1. Create a new virtual environment (Python 3.10 is required):
   ```bash
   python3.10 -m venv senseflow_env
   source senseflow_env/bin/activate
   ```

2. Install PyTorch with CUDA support first (compatible with CUDA 12.4):
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   ```

3. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## ⚙️ Setup

Before training, you need to download the pretrained teacher models and configure the paths in the trainer files.

### SDXL

1. Download Stable Diffusion XL base model from HuggingFace:
   ```bash
   # Using huggingface-cli
   huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir /path/to/stable-diffusion-xl-base-1.0
   ```

2. Update the path in trainer files:
   - Open `senseflow/trainer/trainer_sdxl_senseflow.py` or `senseflow/trainer/trainer_sdxl_DMD2.py`
   - Replace `PLACEHOLDER_SDXL_PATH` with your local path to `stable-diffusion-xl-base-1.0`

### SD3.5 Medium

1. Download Stable Diffusion 3.5 Medium from HuggingFace:
   ```bash
   huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir /path/to/stable-diffusion-3.5-medium
   ```

2. Update the path in trainer file:
   - Open `senseflow/trainer/trainer_sd35_senseflow.py`
   - Replace `PLACEHOLDER_SD35_MEDIUM_PATH` with your local path to `stable-diffusion-3.5-medium`

### SD3.5 Large

1. Download Stable Diffusion 3.5 Large from HuggingFace:
   ```bash
   huggingface-cli download stabilityai/stable-diffusion-3.5-large --local-dir /path/to/stable-diffusion-3.5-large
   ```

2. Update the path in trainer file:
   - Open `senseflow/trainer/trainer_sd35_large_senseflow.py`
   - Replace `PLACEHOLDER_SD35_LARGE_PATH` with your local path to `stable-diffusion-3.5-large`

### FLUX

1. Download FLUX.1-dev from HuggingFace:
   ```bash
   huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir /path/to/FLUX.1-dev
   ```

2. Create a directory for FLUX without guidance embedding:
   ```bash
   mkdir -p exp_flux/flux-wo-guidance-embed/transformer
   ```

3. Create symlinks for transformer files (all files except config.json, which uses a modified version):
   ```bash
   # Navigate to your SenseFlowCode directory
   cd /path/to/SenseFlowCode/exp_flux/flux-wo-guidance-embed/transformer
   
   # Create symlinks for all files from FLUX.1-dev/transformer except config.json
   for file in /path/to/FLUX.1-dev/transformer/*; do
       filename=$(basename "$file")
       if [ "$filename" != "config.json" ]; then
           ln -s "$file" "$filename"
       fi
   done
   ```

4. The `config.json` with `guidance_embeds: false` is already provided in `exp_flux/flux-wo-guidance-embed/transformer/config.json`. This modified config file disables guidance embeddings for training.

5. Update the paths in trainer file:
   - Open `senseflow/trainer/trainer_flux_senseflow.py`
   - Replace `PLACEHOLDER_FLUX_PATH` with your local path to `FLUX.1-dev`
   - Replace `PLACEHOLDER_FLUX_WO_GUIDANCE_EMBED_PATH` with the absolute path to `exp_flux/flux-wo-guidance-embed`

## 📊 Dataset Preparation

### SDXL Training (DMD2 and SenseFlow)

For SDXL training, we use LMDB datasets from DMD2:

1. Download the LMDB dataset from DMD2 HuggingFace repository:
   ```bash
   # Navigate to: https://huggingface.co/tianweiy/DMD2/tree/main/data/laion_vae_latents
   # Download the LMDB dataset files
   ```

2. Update the dataset path in trainer files:
   - Open `senseflow/trainer/trainer_sdxl_senseflow.py` or `senseflow/trainer/trainer_sdxl_DMD2.py`
   - Replace `PLACEHOLDER_LMDB_DATASET_PATH` with your local path to the LMDB dataset directory

### SD3.5 Medium/Large and FLUX Training

For SD3.5 and FLUX training, we use text-image datasets with a JSON file format.

1. Prepare your dataset JSON file with the following structure:
   ```json
   {
       "keys": ["00000000", "00000001", "00000002", ...],
       "image_paths": [
           "/path/to/images/00000000.png",
           "/path/to/images/00000001.png",
           "/path/to/images/00000002.png",
           ...
       ],
       "prompts": [
           "A beautiful sunset over the ocean",
           "A cat sitting on a windowsill",
           "A modern city skyline at night",
           ...
       ]
   }
   ```
   
   **Important**: The three lists (`keys`, `image_paths`, `prompts`) must have the same length, and each index corresponds to one sample.

2. Update the dataset path in trainer files:
   - For SD3.5 Medium: Open `senseflow/trainer/trainer_sd35_senseflow.py`
   - For SD3.5 Large: Open `senseflow/trainer/trainer_sd35_large_senseflow.py`
   - For FLUX: Open `senseflow/trainer/trainer_flux_senseflow.py`
   - Replace `PLACEHOLDER_JSON_DATASET_PATH` with your local path to the JSON file

3. Ensure image paths in the JSON file are absolute paths or paths relative to where you run the training script.

## 🏋️ Training

We provide training scripts in the `exp_*` directories. Each script takes 4 arguments: number of nodes, number of GPUs per node, config file path, and save directory path.

### FLUX SenseFlow

```bash
sh exp_flux/train_flux_senseflow.sh \
    1 8 \
    configs/FLUX/flux_senseflow.yaml \
    /path/to/save/directory
```

### SDXL SenseFlow

```bash
sh exp_sdxl/train_sdxl_senseflow.sh \
    1 8 \
    configs/sdxl/sdxl_senseflow.yaml \
    /path/to/save/directory
```

### SDXL DMD2

```bash
sh exp_sdxl/train_sdxl_dmd2.sh \
    1 8 \
    configs/sdxl/sdxl_dmd2.yaml \
    /path/to/save/directory
```

### SD3.5 Medium SenseFlow

```bash
sh exp_sd35/train_SD35_senseflow.sh \
    1 8 \
    configs/SD35/sd35_senseflow.yaml \
    /path/to/save/directory
```

### SD3.5 Large SenseFlow

```bash
sh exp_sd35/train_SD35_large_senseflow.sh \
    1 8 \
    configs/SD35/sd35_senseflow.yaml \
    /path/to/save/directory
```

**Training Arguments:**
- First argument: Number of nodes
- Second argument: Number of GPUs per node
- Third argument: Path to config file
- Fourth argument: Path to save directory

## 🎨 Inference

We provide inference scripts for different models:

### FLUX SenseFlow

```bash
python scripts_flux/test_flux_senseflow.py \
    --flux_ckpt /path/to/FLUX.1-dev \
    --checkpoint /path/to/senseflow_checkpoint.pth \
    --output_dir ./outputs
```

### SDXL SenseFlow

```bash
python scripts_sdxl/test_sdxl_senseflow.py \
    --sdxl_ckpt /path/to/stable-diffusion-xl-base-1.0 \
    --checkpoint /path/to/senseflow_checkpoint.pth \
    --output_dir ./outputs
```

### SDXL DMD2

```bash
python scripts_sdxl/test_sdxl_dmd2.py \
    --sdxl_ckpt /path/to/stable-diffusion-xl-base-1.0 \
    --checkpoint /path/to/dmd2_checkpoint.pth \
    --output_dir ./outputs
```

### SD3.5 Medium SenseFlow

```bash
python scripts_sd35/test_senseflow_sd35.py \
    --sd35_ckpt /path/to/stable-diffusion-3.5-medium \
    --checkpoint /path/to/senseflow_checkpoint.pth \
    --output_dir ./outputs
```

### SD3.5 Large SenseFlow

```bash
python scripts_sd35/test_senseflow_sd35_large.py \
    --sd35_ckpt /path/to/stable-diffusion-3.5-large \
    --checkpoint /path/to/senseflow_checkpoint.pth \
    --output_dir ./outputs
```

### Inference Arguments

All inference scripts support the following optional arguments:

- `--prompts_file`: Path to prompts text file (default: `senseflow_test_prompts.txt`)
- `--start_idx`: Starting index in prompts file (default: 0)
- `--num_prompts`: Number of prompts to process (default: 23)
- `--batch_size`: Batch size for inference (default: 1)
- `--output_dir`: Output directory for generated images (default: `./outputs`)

For FLUX:
- `--dit_config`: Path to DIT transformer config file (default: `exp_flux/flux-wo-guidance-embed/transformer/config.json`)

For SDXL:
- `--unet_config`: Path to UNet config file (default: `<sdxl_ckpt>/unet/config.json`)

For SD35:
- `--transformer_config`: Path to transformer config file (default: `<sd35_ckpt>/transformer/config.json`)

## 📈 Results

### Table 1: Quantitative Results on COCO-5K Dataset

**Bold** = best, <ins>Underline</ins> = second best. All results on 4-step generation.

#### Stable Diffusion XL Comparison

| Method           | NFE | FID-T | Patch FID-T | CLIP | HPSv2 | Pick | ImageReward |
|------------------|--------|----------|----------------|--------|---------|---------|----------------|
| SDXL             | 80     | --       | --             | 0.3293 | 0.2930  | 22.67   | 0.8719         |
| LCM-SDXL         | 4      | 18.47    | 30.63          | 0.3230 | 0.2824  | 22.22   | 0.5693         |
| PCM-SDXL         | 4      | 14.38    | 17.77          | 0.3242 | 0.2920  | 22.54   | 0.6926         |
| Flash-SDXL       | 4      | 17.97    | 23.24          | 0.3216 | 0.2830  | 22.17   | 0.4295         |
| SDXL-Lightning   | 4      | **13.67**| **16.57**      | 0.3214 | 0.2931  | 22.80   | 0.7799         |
| Hyper-SDXL       | 4      | <ins>13.71</ins>  | <ins>17.49</ins>        | 0.3254 | <ins>0.3000</ins> | <ins>22.98</ins> | <ins>0.9777</ins> |
| DMD2-SDXL        | 4      | 15.04    | 18.72          | **0.3277** | 0.2963 | <ins>22.98</ins> | 0.9324         |
| Ours-SDXL        | 4      | 17.76    | 21.01          | 0.3248 | **0.3010** | **23.17** | **0.9951** |

#### Stable Diffusion 3.5 Comparison

| Method               | NFE | FID-T | Patch FID-T | CLIP | HPSv2 | Pick | ImageReward |
|----------------------|--------|----------|----------------|--------|---------|---------|----------------|
| SD 3.5 Large         | 100    | --       | --             | 0.3310 | 0.2993  | 22.98   | 1.1629         |
| SD 3.5 Large Turbo   | 4      | <ins>13.58</ins>  | 22.88          | 0.3262 | 0.2909  | 22.89   | 1.0116         |
| Ours-SD 3.5          | 4      | **13.38**| **17.48**      | <ins>0.3286</ins> | **0.3016** | **23.01** | <ins>1.1713</ins> |
| Ours-SD 3.5 (Euler)  | 4      | 15.24    | <ins>20.26</ins>        | **0.3287** | <ins>0.3008</ins> | <ins>22.90</ins> | **1.2062** |

#### FLUX Comparison

| Method            | NFE | FID-T | Patch FID-T | CLIP | HPSv2 | Pick | ImageReward |
|-------------------|--------|----------|----------------|--------|---------|---------|----------------|
| FLUX.1 dev        | 50     | --       | --             | 0.3202 | 0.3000  | 23.18   | 1.1170         |
| FLUX.1 dev        | 25     | --       | --             | 0.3207 | 0.2986  | 23.14   | 1.1063         |
| FLUX.1-schnell    | 4      | --       | --             | **0.3264** | 0.2962 | 22.77   | 1.0755         |
| Hyper-FLUX        | 4      | <ins>11.24</ins>  | 23.47          | <ins>0.3238</ins> | 0.2963  | 23.09   | <ins>1.0983</ins> |
| FLUX-Turbo-Alpha  | 4      | **11.22**| 24.52          | 0.3218 | 0.2907  | 22.89   | 1.0106         |
| Ours-FLUX         | 4      | 15.64    | **19.60**      | 0.3167 | <ins>0.2997</ins> | <ins>23.13</ins> | 1.0921         |
| Ours-FLUX (Euler) | 4      | 16.50    | <ins>20.29</ins>        | 0.3171 | **0.3008** | **23.26** | **1.1424**     |


### 1024 x 1024 examples of our 4-step generator distilled on SD 3.5 Large
<img src="assets/fig_supp_sd35.jpg" style="zoom:50%;" />

### 1024 x 1024 examples of our 4-step generator distilled on SDXL
<img src="assets/fig_supp_sdxl.jpg" style="zoom:50%;" />

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{ge2025senseflow,
  title={SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation},
  author={Ge, Xingtong and Zhang, Xin and Xu, Tongda and Zhang, Yi and Zhang, Xinjie and Wang, Yan and Zhang, Jun},
  journal={arXiv preprint arXiv:2506.00523},
  year={2025}
}
```

## ⚖️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

**Note**: This codebase is based on several open-source models including:
- [Stable Diffusion XL](https://github.com/Stability-AI/generative-models) (CreativeML Open RAIL-M License)
- [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) (CreativeML Open RAIL-M License)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (CreativeML Open RAIL-M License)

Please ensure compliance with their respective licenses when using the teacher models.
