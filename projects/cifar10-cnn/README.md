# CIFAR-10 CNN learning exercise

This PyTorch exercise demonstrates a small image classifier, dataset loading,
a reproducible training/validation split, and a validation helper. It is a
learning project with an unfinished training workflow.

## Status and provenance

**Work in progress.** A model, loss, optimiser and validation helper exist.
There is no training loop, checkpoint workflow or trained-model evaluation in
these files. See [PROVENANCE.md](PROVENANCE.md) for the existing source and
Codex-assisted preparation. Earlier authorship and tutorial sources need review.

## Method

- Dataset: CIFAR-10, 10 classes, RGB images of size 32 x 32.
- Split: 45,000 training / 5,000 validation images from the training set;
  10,000 separate test images. The split generator uses seed 42.
- Transform: tensor conversion and per-channel normalisation with mean 0.5
  and standard deviation 0.5. No augmentation is implemented.
- Model: 3-to-32 convolution, ReLU, 2x2 max pooling; 32-to-32 convolution,
  ReLU, 2x2 max pooling; flatten 2,048 features; linear layer with 10 logits.
  Both convolutions use a 3x3 kernel and padding 1.
- Configured objective: cross-entropy. Configured optimiser: SGD with learning
  rate 0.1, momentum 0.9 and weight decay 0.0005. No optimiser steps are run.
- `val_loader(...)` returns mean batch loss and classification accuracy in
  percent. Mean batch loss weights batches equally, including a shorter last
  batch; it is not an exact per-example mean in that case.

## Reproduce

Use Python 3.10 or newer with compatible PyTorch and torchvision installations.
For a fresh CPU environment, from this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe cifar10_cnn.py
```

For an existing compatible environment, run `python cifar10_cnn.py`. It selects
CUDA when available, otherwise CPU. CUDA requires a compatible PyTorch build.
The dependency list is minimal, not a locked environment. Preparation checks
used Python 3.12, torch 2.14.0+cu126 and torchvision 0.29.0+cu126.

The script downloads CIFAR-10 into `./data` if necessary, then runs a single
forward-pass check. An initial download needs network access. Expected shapes
for the default batch are images `(64, 3, 32, 32)`, labels `(64,)`, and logits
`(64, 10)`; dataset sizes should print as `45000 5000 10000`.
It does not train, save weights, write graphs or report trained accuracy.
Dataset information: https://www.cs.toronto.edu/~kriz/cifar.html

A model-only check without downloading data:

```powershell
python -B -c "import torch; from cifar10_cnn import model, device; x = torch.zeros(2, 3, 32, 32, device=device); model.eval(); y = model(x); assert y.shape == (2, 10); assert torch.isfinite(y).all(); print('Model shape check passed:', tuple(y.shape))"
```

See [VALIDATION.md](VALIDATION.md) for the checks actually run during preparation.

## Results

**Not yet evaluated.** No training history, loss curves or trained validation
or test accuracy has been verified. A synthetic-input shape check establishes
execution and output shape only.

## Limitations and next steps

The split is seeded, but model initialisation and training-loader order are
not fixed. Training and final evaluation remain to be implemented or recovered
from an existing version. Confirm sources and authorship, then add a documented
training workflow, run it, and record its actual metrics and conditions.

No licence has been selected; the containing portfolio currently has no licence.
