# Code Supplement: A Reassessment of Global Weight-Space Sampling

This repository contains minimal exemplary code for the paper *"A Reassessment of Global Weight-Space Sampling via Deterministic Flat Minima"*.

## Repository Structure

* `sam.py`: Implementation of the Sharpness-Aware Minimization (SAM) and Adaptive SAM (ASAM) optimizers.
* `train.py`: Fine-tuning script to create the deterministic flat-minima basin.
* `eval_deterministic.py`: Evaluation script (with no overhead, $M=0$) to measure predictive accuracy and implicit entropy regularization.
* `eval_stochastic_surrogate.py`: Implementation of Algorithm 1, used to inject geometrically bounded Wasserstein noise for the Dimensionality Dilution / Functional Disruption ablations.

## Data Preparation

The scripts expect datasets formatted in the standard PyTorch ImageFolder structure. By default, the code points to ./data/imagenet. For out-of-distribution and anomaly evaluations, point the --data_dir argument to your local ImageNet-C or ImageNet-O directories.


```
data/
└── imagenet/
    ├── train/
    │   ├── class_0/
    │   └── class_1/
    └── val/
        ├── class_0/
        └── class_1/
```

## Usage Instructions

### 1. Training

Fine-tune a base model using SAM (Euclidean space) or ASAM (Mahalanobis space). The --rho parameter dictates the Wasserstein perturbation radius.

```
# Standard SAM (e.g., for ResNet-50)
python train.py --model resnet50 --rho 0.5 --epochs 10

# Adaptive SAM (ASAM) (e.g., for ViT-B/16)
python train.py --model vit_b_16 --rho 0.2 --adaptive --epochs 10
```

### 2. Deterministic Inference

Evaluate the training checkpoint without requiring stochastic passes or batch normalization recalibration.

```
python eval_deterministic.py --model resnet50 --weights deterministic_resnet50_rho0.5.pth --data_dir ./data/imagenet/val
```

### 3. Inference of the Stochastic Surrogate

Reproduce the "Scale Sweep" experiments (Section 4.2.3 and Appendix A.5).
This script injects isotropic Gaussian noise bounded by $\rho$ and scales it via the --multiplier argument ($k$).

```
python eval_stochastic_surrogate.py \
    --model resnet50 \
    --weights deterministic_resnet50_rho0.5.pth \
    --data_dir ./data/imagenet-c/pixelate/5 \
    --rho 0.5 \
    --multiplier 10.0 \
    --samples 10
```






