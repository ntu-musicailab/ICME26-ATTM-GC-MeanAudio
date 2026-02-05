# ICME26 ATTM-GC Baseline Model

This repository provides comprehensive instructions for training a FluxAudio model using the Jamendo dataset, serving as a baseline and starting point for participants of the ICME 2026 Academic Text-to-Music Grand Challenge (ATTM-GC). The goal is to help teams get started quickly, while exploration of alternative architectures is also encouraged.

## Environmental Setup

**1. Create a new conda environment:**

```bash
conda create -n meanaudio python=3.11 -y
conda activate meanaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
<!-- ```
conda install -c conda-forge 'ffmpeg<7
```
(Optional, if you use miniforge and don't already have the appropriate ffmpeg) -->

**2. Install with pip:**

```bash
git clone https://github.com/ntu-musicailab/ICME26-ATTM-GC-MeanAudio.git

cd ICME26-ATTM-GC-MeanAudio
pip install -e .
```

<!-- (If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip) -->

## Training
Before training, make sure that all files from [here](https://huggingface.co/AndreasXi/MeanAudio) are placed in the `ICME26-ATTM-GC-MeanAudio/weights/` directory. 

### 1. Jamendo Dataset Preparation:

To train the FluxAudio model with the provided Jamendo dataset, you need to complete the following steps:

#### Prerequisites:
- Download the `raw_30s` subset from [the MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset)
- Preprocess the Jamendo dataset using the [ICME26-ATTM-GC-Preprocessing](https://github.com/ntu-musicailab/ICME26-ATTM-GC-Preprocessing) official preprocessing pipeline
  - The preprocessed audio should be organized as: `{audio_root}/{sub_folder}/{sample_id}_instrumental.mp3`
  - Example: `mtg_jamendo_separated/00/1085700_instrumental.mp3`

#### Prepare MeanAudio-Ready Dataset:

Run the preparation script to organize the dataset into train/val/test splits and desired structure:

```bash
python training/prepare_jamendo_for_meanaudio.py \
    --audio_root /path/to/mtg_jamendo_separated \
    --val_samples 100 \
    --test_samples 100 \
```

**Arguments:**
- `--audio_root`: Path to preprocessed Jamendo audio files (with instrumental stems)
- `--val_samples`: Number of samples for validation (default: 100, adjust as desired)
- `--test_samples`: Number of samples for testing (default: 100, adjust as desired)

The script will create train/val/test/all splits in `./data/jamendo_meanaudio_ready/`, with audio symlinks and TSV files containing sample IDs and captions for each split.

### 2. Latent & Text Feature Extraction: 
Extract VAE latents and text encoder embeddings to enable efficient training. The extraction pipeline consists of two steps: (a) partitioning audio files into 10-second clips, and (b) extracting latents and embeddings into NPZ files.

Before running, edit `scripts/extract_audio_latents.sh` to configure the paths for each split (train/val/test).

Then run the extraction script for each split:
```bash
bash scripts/extract_audio_latents.sh
```

**Note:** This extraction process must be completed separately for train, validation, and test splits by updating the paths in the script accordingly.

### 3. Install Validation Packages: 
Install [av-benchmark](https://github.com/hkchengrex/av-benchmark) for validation and evaluation metrics. After installation, create a symlink to the `av_bench` module in this repository's root directory:

```bash
ln -s /path/to/av-benchmark/av_bench ./av_bench
```

Verify the symlink is created correctly: `ICME26-ATTM-GC-MeanAudio/av_bench` should point to `av-benchmark/av_bench`.

### 4. Train FluxAudio: 
Train a Flux-style transformer model using the conditional flow matching objective. Choose between two model variants based on your computational resources and performance requirements:

#### Model Variants

| Model Name  | Size | Training Script                             |
| ----------- | ---- | ------------------------------------------- |
| FluxAudio-S | 120M | `scripts/flowmatching/train_fluxaudio_s.sh` |
| FluxAudio-L | 480M | `scripts/flowmatching/train_fluxaudio_l.sh` |

**Train FluxAudio-S (smaller, faster):**
```bash 
bash scripts/flowmatching/train_fluxaudio_s.sh
```

**Train FluxAudio-L (larger, higher quality):**
```bash
bash scripts/flowmatching/train_fluxaudio_l.sh
```

## Citation

```bibtex
@article{li2025meanaudio,
  title={MeanAudio: Fast and Faithful Text-to-Audio Generation with Mean Flows},
  author={Li, Xiquan and Liu, Junxi and Liang, Yuzhe and Niu, Zhikang and Chen, Wenxi and Chen, Xie},
  journal={arXiv preprint arXiv:2508.06098},
  year={2025}
}
```



## Acknowledgement

This repository is built upon [xiquan-li/MeanAudio](https://github.com/xiquan-li/MeanAudio.git).

Many thanks to:
- [MMAudio](https://github.com/hkchengrex/MMAudio) for the MMDiT code and training & inference structure
- [MeanFlow-pytorch](https://github.com/haidog-yaqub/MeanFlow) and [MeanFlow-official](https://github.com/Gsunshine/meanflow) for the mean flow implementation
- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) BigVGAN Vocoder and the VAE
- [av-benchmark](https://github.com/hkchengrex/av-benchmark) for benchmarking results
