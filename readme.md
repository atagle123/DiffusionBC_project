# DiffusionBC Project

Requires installation of Mujoco 2.1 first

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate diffusion_bc
conda develop .
```

## Run a experiment

```bash
python scripts/train.py datasets={dataset_name} seed={your_seed} wandb.log=true/false
```