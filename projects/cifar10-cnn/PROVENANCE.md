# Provenance

The two Python files were copied from Mamadou's existing PyTorch learning
folder on 2 November 2026. Their earlier authorship, tutorial sources and
independent implementation history have not been verified.

Codex prepared this copy and authored the README, dependency list, ignore rules
and this provenance note. In `cifar10_cnn.py`, Codex added a status docstring,
removed an unused import, adjusted spacing and redundant parentheses, and moved
the smoke-check input to the model's selected device. The device fix addresses
a runtime mismatch reproduced on CUDA in the original file.

Following Mamadou's explicit request to reuse the MLP training loop, Codex
adapted `train_loop` from his existing `kminst_mlp.py` into the shared
`../train_classifier.py`. Codex added example-weighted metrics, deterministic
seeding, validation-based checkpoint selection, final test evaluation, plots,
CLI options and tests. Gradient clearing is performed before each forward pass;
the original MLP clears after each optimiser step. Both implement the same
gradient-reset cycle when starting with no accumulated gradients.

The CIFAR loader now accepts a shuffle seed and download flag; the 45,000/5,000
split and split seed remain unchanged. The model architecture, preprocessing,
SGD learning rate, momentum and weight decay are unchanged. A five-epoch
check was selected to verify the workflow, followed by the MLP script's original
20-epoch schedule. The five-epoch record is retained; neither run is an
optimisation claim. Measured results are recorded separately after execution. Original source
files outside this repository were left unchanged.

Before publication, Mamadou should confirm authorship and any external sources,
choose an appropriate licence, and review the exact repository and files.
