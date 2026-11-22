# Modular PyTorch source package (`src/`)

This package contains reusable model definitions and experiment helpers extracted/refactored from the seven final notebooks in `notebooks/`. The notebooks remain the authoritative record of the completed Colab runs and preserve their exact training histories, checkpoints, metrics, and figure-generation procedures.

## Contents

- `src/data/fashion_mnist.py`: pooled Fashion-MNIST loading, the shared seed-42 stratified 80/10/10 split, SHA-256 checks, and DataLoaders.
- `src/models/`: the six generative architectures plus the Model 07 MLP classifier and frozen latent decoder.
- `src/training/`: reusable loss and one-batch optimizer-step helpers matching each model family.
- `src/evaluation/`: reconstruction metrics, FID from extracted features, and synchronized batch-latency measurement.
- `src/utils/`: seeding, atomic checkpoint I/O, and config loading.
- `tests/test_smoke.py`: CPU shape and parameter-count smoke checks; this does not train the models.

## Important experiment conventions

1. Dataset order is official Fashion-MNIST `train=True` first (60,000), then `train=False` (10,000).
2. The pooled 70,000 images are split with `np.random.default_rng(42)`, by class, into 5,600/700/700 indices per class. The final arrays are shuffled in train, validation, test order. This produces 56,000/7,000/7,000 samples.
3. Canonical image tensors are float32 `[B,1,28,28]` in `[0,1]`. GAN/DCGAN and DDPM convert to `[-1,1]` at the model boundary.
4. VAE reconstruction objective is per-image summed SSE plus beta times summed KL, averaged over the batch. Sparse AE objective is mean MSE plus `lambda_l1 * mean(abs(z))`.
5. Vanilla GAN uses sigmoid discriminator probabilities with `BCELoss`; DCGAN uses raw logits with `BCEWithLogitsLoss`.
6. DDPM predicts epsilon and its reported inference latency is the complete reverse process, not one U-Net call.
7. Model 07 classification accuracy and frozen-decoder reconstruction metrics are separate quantities.

## Run smoke checks

From the repository root:

```bash
python -m tests.test_smoke
```

The test intentionally uses a reduced DDPM U-Net width for a quick CPU forward-pass check. The full experiment DDPM configuration remains the one in the Model 06 notebook (`base_channels=32`, `channel_mult=(1,2,4)`, `time_emb_dim=128`, `T=1000`).

## Using a model

```python
from src.models import VanillaAutoencoder, VAE, DCGANGenerator, MLPClassifier

model = VanillaAutoencoder()
```

## Scope note

The original notebooks are self-contained execution artifacts. This package provides a modular, reusable source layer for model/data/objective/evaluation code. It does not replace the recorded notebook runs or silently change their outputs. Before claiming that the notebooks execute through `src/`, the notebook imports and execution paths would need to be explicitly refactored and tested; this package alone does not rewrite the already-completed experiments.
