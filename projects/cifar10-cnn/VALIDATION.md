# Preparation checks — 2 November 2026

Environment: Python 3.12; torch 2.14.0+cu126; torchvision 0.29.0+cu126;
NVIDIA GeForce RTX 4080 SUPER. These are observed installed versions, not an
independently verified package-installation recipe.

| Check | Observed result |
|---|---|
| Parse both Python files | Passed |
| Compare original/prepared model parameters with the same initialisation seed | Exact match |
| Compare original/prepared forward outputs on identical synthetic inputs | Exact match on CPU and CUDA |
| Synthetic output shape and finite-value checks | Passed: `(2, 10)` |
| Validation helper on synthetic inputs labelled by its predictions | Passed on CPU and CUDA; checks helper mechanics only |
| Compare original/prepared data-loader source | Byte-for-byte match |
| Load cached CIFAR-10 through `get_data_loaders` | Passed: 45,000 / 5,000 / 10,000 examples |
| Forward one actual training batch on CUDA | Passed: images `(64, 3, 32, 32)`, labels `(64,)`, finite logits `(64, 10)` |

The actual-data check called the existing loader with the cached dataset root,
took one batch, then ran `model(images.to(device))` under `torch.no_grad()`.
The public entry point uses the same operations with its default `./data` root.
Its network download path and a fresh environment installation were not tested.

The original smoke check failed on CUDA because images remained on CPU; the
prepared copy fixes that device mismatch. Data-folder access initially failed
inside the restricted environment; the later authorised check could read it.

No optimiser steps were executed. No trained accuracy, loss curves or benchmark
results are available. Passing these checks does not establish learning quality.
