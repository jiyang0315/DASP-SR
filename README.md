# DASP-SR

Official repository for the paper:

**Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution**

## Overview

DASP-SR is a diffusion-based framework for real-world image super-resolution.  
It improves restoration quality with degradation-aware conditioning and structure-preserving noise design.

This repository contains the training and inference code used for DASP-SR. The implementation uses Diffusers-style components and adds degradation-aware and structure-preserving restoration modules.

## Environment

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The code was developed with PyTorch 2.0.1, Diffusers 0.21.0, Accelerate, Transformers, and xFormers.

## Model Preparation

Place the required pretrained models under `preset/models/`:

```text
preset/models/
  stable-diffusion-2-base/
  DAPE.pth
  ram_swin_large_14m.pth
  daspsr/
    unet/
    controlnet/
```

Large pretrained weights are not tracked by Git. Please download or prepare them separately before running inference or training.

## Inference

Example inference command:

```bash
bash infer.sh
```

The script calls `test_daspsr.py` with the expected model, input, and output paths. Update `--image_path`, `--output_dir`, and model paths in `infer.sh` for your local setup.

## Training

Example training command:

```bash
bash train.sh
```

Training data should be organized under `preset/datasets/`. The helper scripts `get_train_data.sh` and `get_train_caption.sh` provide examples for data preparation and caption/tag generation.

## Repository Layout

- `train_daspsr.py`: training entry point.
- `test_daspsr.py`: inference/testing entry point.
- `models/`: ControlNet/UNet model definitions.
- `pipelines/`: diffusion pipeline implementation.
- `dataloaders/`: paired dataset loaders.
- `utils/`: image, noise, and restoration utilities.
- `utils_data/`: data preparation scripts.
- `ram/`: RAM/DAPE tagging model components.
- `figs/`: figures and visual examples.

## License

This project is released under the Apache-2.0 license. See `LICENSE` for details.

## Acknowledgements

This codebase builds on open-source real-world image super-resolution and diffusion restoration components. Attribution notices in adapted source files are preserved where applicable. See `NOTICE` for details.
