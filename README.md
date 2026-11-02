# Mamadou Foune Barry — programming portfolio

A collection of learning exercises in Python, data analysis and scientific
computing. Project status and measured evidence are stated individually.

## PyTorch practice

### [CIFAR-10 CNN](projects/cifar10-cnn/README.md)

An image classifier with two convolutional layers and a complete training,
validation and held-out test workflow. The training loop is adapted from the
existing MLP exercise. Seeded data loading, checkpoint selection, measured
metrics and learning curves are included.

- [Reproduction and limitations](projects/cifar10-cnn/README.md)
- [Checks actually run](projects/cifar10-cnn/VALIDATION.md)
- [Existing source and AI-assisted preparation](projects/cifar10-cnn/PROVENANCE.md)

### [KMNIST MLP](projects/kmnist-mlp/README.md)

The existing 784-512-512-10 multilayer perceptron, with its training loop adapted
into the shared [classifier trainer](projects/train_classifier.py). The cleanup
adds a validation split and selects checkpoints before final test evaluation.
These are learning baselines, not tuned research benchmarks.

| Exercise | Training budget | Selected epoch | Held-out test accuracy |
|---|---:|---:|---:|
| CIFAR-10 CNN | 20 epochs, seed 42 | 11 | 50.43% |
| KMNIST MLP | 20 epochs, seed 42 | 20 | 65.34% |

Selection uses validation loss; each test set contains 10,000 examples. The
datasets differ, so the percentages are not an architecture comparison. Full
metrics, limitations and learning curves are linked from each project README.

Both projects use [shared dependencies](projects/requirements.txt) and
[dataset-free training tests](tests/test_training.py). Each project documents
the changes from its original implementation and the actual run conditions.

## Earlier notebooks

These existing exercises remain available. They have not been rerun or fully
reviewed during the portfolio preparation; filenames are not validation of
scientific results.

- [Exoplanet data-analysis notebook](.Week1_Exoplanets_data_analysis.ipynb)
- [Monte Carlo particle-collider notebook](MonteCarloParticleColliderSimulator.ipynb)
- [Pandas practice](Week1%28Pandas%29.ipynb)
- [Exploratory data-analysis practice](Week1_EDA.ipynb)

## Scope and provenance

The PyTorch exercises are supporting programming evidence, with earlier
authorship and sources still to be confirmed. Codex prepared this landing page,
adapted the supplied training loop and added the documented workflow support.
Existing notebooks were left unchanged.
Research projects and manuscripts are not included in this update.

No licence has been selected for this portfolio.
