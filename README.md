# Comparative Study of Generative Models on Fashion-MNIST

An empirical comparison of six generative architectures and one MLP classifier experiment using Fashion-MNIST. The project evaluates reconstruction fidelity, sample quality, latent-space behavior, training dynamics, inference cost, and the limitations of discriminative classifier representations for image generation.

**Author:** Matthew Sutiono  
**Repository:** [champsand/generative-model-comparison-dl](https://github.com/champsand/generative-model-comparison-dl)  
**Execution environment:** Google Colab, NVIDIA Tesla T4 GPU

---

## Project scope

The project contains seven experiments:

| ID | Model | Main purpose | Main evaluation |
|---|---|---|---|
| 01 | Vanilla Autoencoder | Deterministic image reconstruction | MSE, PSNR, SSIM; latent interpolation |
| 02 | Sparse Autoencoder | Reconstruction with an L1 latent-activation penalty | MSE, PSNR, SSIM; latent sparsity; interpolation |
| 03 | Vanilla GAN | Fully connected unconditional generation | Inception FID; sample diversity diagnostics; approximate latent inversion |
| 04 | Variational Autoencoder (VAE) | Probabilistic latent representation and generation | MSE, PSNR, SSIM, Inception FID; prior sampling; interpolation |
| 05 | DCGAN | Convolutional adversarial generation | Inception FID; sample diversity diagnostics; approximate latent inversion |
| 06 | DDPM | Iterative diffusion-based generation | Inception FID; epsilon-prediction loss; reverse denoising trajectory |
| 07 | MLP Classifier | Test the limits of classifier weights and latent features for image generation | Classification accuracy; transpose projections; frozen-decoder reconstruction metrics; Gaussian latent sampling |

The six generative models are independent, self-contained Colab notebooks. Model 07 is a separate classifier-limitation experiment and is not included as a seventh generative model in the generative FID comparison.

## Headline results

The following values are read from the final experiment `metrics.json` files committed under `results/metrics/`. MSE, PSNR, and SSIM are reported only for models that reconstruct a corresponding input. FID is reported for the four unconditional/probabilistic generative models for which it was computed.

| Model | Test MSE | Test PSNR (dB) | Test SSIM | Test Inception FID |
|---|---:|---:|---:|---:|
| Vanilla Autoencoder | 0.008073 | 20.930 | 0.7452 | N/A |
| Sparse Autoencoder | 0.008382 | 20.767 | 0.7405 | N/A |
| Vanilla GAN | N/A | N/A | N/A | 149.307 |
| VAE | 0.017124 | 17.664 | 0.5825 | 89.461 |
| DCGAN | N/A | N/A | N/A | 22.957 |
| DDPM | N/A | N/A | N/A | **20.748** |

**Separate Model 07 result:** the MLP classifier achieved **89.86% test accuracy**, exceeding the experiment's 88% target. Its independently trained frozen latent decoder achieved test MSE **0.018038**, PSNR **17.438 dB**, and SSIM **0.5962**. These decoder metrics must not be confused with classifier accuracy.

These results are experiment-specific and should be interpreted with the protocol notes below. The complete-precision values and per-run metadata are preserved in the corresponding JSON files.

## Dataset and split protocol

The experiments use torchvision Fashion-MNIST. The official 60,000-image training partition (`train=True`) and 10,000-image test partition (`train=False`) were pooled, in that order, into a 70,000-image dataset. A deterministic class-stratified 80/10/10 split was then generated with seed 42:

| Partition | Total images | Images per class |
|---|---:|---:|
| Train | 56,000 | 5,600 |
| Validation | 7,000 | 700 |
| Test | 7,000 | 700 |

All seven experiments use the same ordered split, verified using SHA-256 hashes of the split-index arrays.

**Important split limitation:** because the two official torchvision partitions were pooled before splitting, the original official 10,000-image test partition is not retained as an independent test set. The project's test set is the 7,000-image stratified partition drawn from the pooled 70,000 images. Results should therefore be described as results on the project's held-out stratified test split, not as evaluation on the untouched official Fashion-MNIST test partition.

Canonical image representation is grayscale `[1, 28, 28]` with values in `[0, 1]`. Vanilla GAN and DCGAN convert their real inputs to `[-1, 1]` for training. DDPM also uses `[-1, 1]` image inputs while predicting unconstrained Gaussian noise. Generated and reconstructed images are converted to `[0, 1]` for common evaluation and visualization.

## Model configurations

The resolved experiment configurations are stored in `configs/` and are also preserved alongside each model's metrics in `results/metrics/`.

| Model | Architecture summary | Key training configuration |
|---|---|---|
| Vanilla AE | 784 → 256 → 64 → 256 → 784 | MSE, Adam, learning rate 0.001, 30 epochs |
| Sparse AE | Same 64-dimensional AE backbone | MSE + `0.001 × mean(abs(latent))`, Adam, 0.001, 30 epochs |
| Vanilla GAN | G: 100 → 256 → 512 → 784; D: 784 → 512 → 256 → 1 | BCE adversarial objective, Adam, learning rate 0.0002, betas (0.5, 0.999), 30 epochs |
| VAE | 784 → 256 → (`mu`, `logvar`), latent 64; decoder 64 → 256 → 784 | Reconstruction SSE + `1 × KL`, no KL warm-up, Adam, 0.001, 30 epochs |
| DCGAN | Convolutional generator 7×7 → 14×14 → 28×28; mirrored discriminator | BCE-with-logits adversarial objective, Adam, learning rate 0.0002, betas (0.5, 0.999), 30 epochs |
| DDPM | Small time-conditioned U-Net noise predictor | Linear beta schedule, 1,000 diffusion timesteps, epsilon-prediction MSE, Adam, learning rate 0.0002, 30 epochs |
| MLP Classifier | 784 → 256 → 64 → 10; separate frozen-latent decoder 64 → 256 → 784 | Cross-entropy classifier, Adam, learning rate 0.001; 40 classifier epochs and 30 decoder epochs |

## Evaluation protocol and interpretation

### Reconstruction metrics

- **MSE:** global pixel-wise mean squared error over the held-out test images.
- **PSNR:** computed from aggregate test MSE with a data range of 1.0.
- **SSIM:** arithmetic mean of per-image `skimage.metrics.structural_similarity` scores using `data_range=1.0`, Gaussian weights, `sigma=1.5`, `win_size=11`, and `use_sample_covariance=False`.
- Reconstruction metrics are not applicable to unconditional Vanilla GAN, DCGAN, or DDPM because they do not provide a paired input-reconstruction path.

### Inception FID

The project uses `pytorch-fid==0.3.0`, FID InceptionV3 pool3 features (2,048 dimensions), 7,000 held-out real test images, and 7,000 generated images. Grayscale images are repeated to three channels and resized to 299×299 as required by the feature extractor.

**FID limitation:** InceptionV3 is ImageNet-pretrained on natural RGB images, whereas Fashion-MNIST contains 28×28 grayscale clothing images. The resulting FID is used as a relative comparison under the shared project protocol, not as an absolute perceptual-quality score or a direct comparison with unrelated datasets and papers.

### Latency

Latency measurements were performed on a Tesla T4 GPU with batch size 64. The measured operation differs by model family:

- AE / Sparse AE: encoder plus decoder forward pass.
- VAE: prior latent sampling plus decoder forward pass.
- Vanilla GAN / DCGAN: generator forward pass.
- DDPM: the complete 1,000-step reverse denoising process from noise to an image.

DDPM latency is not equivalent to a single generator forward pass. It used one warm-up and three timed runs; the other model-family benchmarks generally used 20 warm-ups and 100 timed runs. These protocol differences are recorded in the metrics and should be disclosed when discussing latency.

### MLP classifier limitations

The classifier is trained to estimate class labels from images, not to model the image distribution or preserve all information needed for reconstruction. Model 07 includes five experiments: classifier training and feature extraction; one-hot class-vector transpose projection; real latent-feature transpose projection; training a separate decoder while the classifier is frozen; and Gaussian sampling from empirical test-latent statistics. The transpose projections are diagnostics, not true mathematical inverses. Gaussian sampling from empirical classifier features is not a learned generative prior.

## Repository structure

```text
generative-model-comparison-dl/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── AI_ASSISTANCE.md
├── configs/                 # Resolved JSON configuration for Models 01–07
├── notebooks/               # Seven self-contained Google Colab notebooks
├── results/
│   ├── metrics/              # JSON metrics, split reports, and CSV histories
│   └── figures/
│       ├── per_model/        # Original figures generated by each experiment
│       └── comparative/      # Final cross-model figures (add after figure export)
├── src/                      # Reusable modules extracted/refactored from notebooks (to be added)
├── docs/                     # Methodology and reproducibility documentation (to be added)
└── reports/                  # Final scientific report (to be added)
```

The current experiment implementation is contained in the seven notebooks. The `src/`, `docs/`, and `reports/` directories are intended follow-up repository components; they should only be described as complete once the corresponding tested files have been added. The model checkpoints and downloaded dataset remain in Google Drive and are intentionally excluded from GitHub.

## How to use the notebooks

1. Open the repository on GitHub and select the required notebook under `notebooks/`.
2. Open the notebook in Google Colab (GitHub tab) or upload the notebook to Colab.
3. Select a GPU runtime where available and confirm the installed PyTorch/torchvision versions before running.
4. Mount Google Drive when prompted. The notebooks write persistent datasets, checkpoints, logs, and experiment artifacts under `MyDrive/generative-model-comparison/`.
5. Run cells in order. Full training is compute-intensive; the completed results in this repository are provided for inspection and reporting, so do not rerun a full experiment unless intentionally reproducing it.

The notebooks are self-contained and may contain model/data/training/evaluation code directly. The JSON files under `configs/` document the corresponding resolved experiment setup. This repository currently does not expose a command-line training interface.

## Reproducibility and artifact policy

- Global and split seed: 42; each notebook records additional sampling/evaluation seeds in its resolved config and metrics.
- Split counts and split hashes are preserved for every model.
- Per-epoch CSV histories and JSON metrics/configurations are committed for traceability.
- Original experiment figures are stored under `results/figures/per_model/`.
- Large checkpoints (`.pth`, `.pt`, `.ckpt`), raw datasets, cached tensors, and large intermediate feature arrays are kept in Google Drive and must not be committed to GitHub.
- Selected small figures, JSON/CSV metrics, notebooks, configs, and documentation may be committed.

## References

- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- PyTorch: https://pytorch.org/
- Torchvision datasets: https://pytorch.org/vision/stable/datasets.html
- `pytorch-fid`: https://github.com/mseitzer/pytorch-fid
- scikit-image SSIM: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

## License

This project is distributed under the license in [`LICENSE`](LICENSE).
