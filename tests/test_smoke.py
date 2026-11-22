"""CPU smoke checks for source modules. These do not train any model."""
import torch

from src.data.fashion_mnist import make_stratified_split, verify_split
from src.training.reconstruction import autoencoder_loss, sparse_autoencoder_loss, vae_loss
from src.models import (
    VanillaAutoencoder, SparseAutoencoder, VAE,
    VanillaGANGenerator, VanillaGANDiscriminator,
    DCGANGenerator, DCGANDiscriminator, UNet,
    MLPClassifier, FrozenLatentDecoder,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 28, 28)

    ae = VanillaAutoencoder().eval()
    sparse = SparseAutoencoder().eval()
    vae = VAE().eval()
    with torch.no_grad():
        y_ae, z_ae = ae(x)
        y_sparse, z_sparse = sparse(x)
        y_vae, mu, logvar, z_vae = vae(x)
    assert y_ae.shape == y_sparse.shape == y_vae.shape == x.shape
    assert z_ae.shape == z_sparse.shape == z_vae.shape == (2, 64)
    assert count_parameters(ae) == count_parameters(sparse) == 435_536
    assert count_parameters(vae) == 451_984

    gan_g, gan_d = VanillaGANGenerator(), VanillaGANDiscriminator()
    with torch.no_grad():
        fake = gan_g(torch.randn(2, 100))
        prob = gan_d(fake)
    assert fake.shape == (2, 1, 28, 28) and prob.shape == (2, 1)
    assert count_parameters(gan_g) == 559_632
    assert count_parameters(gan_d) == 533_505

    dc_g, dc_d = DCGANGenerator(), DCGANDiscriminator()
    with torch.no_grad():
        fake_dc = dc_g(torch.randn(2, 100, 1, 1))
        logits = dc_d(fake_dc)
    assert fake_dc.shape == (2, 1, 28, 28) and logits.shape == (2,)
    assert count_parameters(dc_g) == 759_680
    assert count_parameters(dc_d) == 138_625

    # Check the full Model 06 parameter count, then a reduced-width fast CPU forward pass.
    full_ddpm = UNet(in_channels=1, base_channels=32, channel_mult=(1, 2, 4),
                     num_res_blocks=2, time_emb_dim=128, groupnorm_groups=8, dropout=0.1)
    assert count_parameters(full_ddpm) == 2_863_393
    del full_ddpm
    ddpm = UNet(base_channels=8, channel_mult=(1, 2, 4), num_res_blocks=2,
                time_emb_dim=32, groupnorm_groups=8, dropout=0.1).eval()
    with torch.no_grad():
        eps = ddpm(torch.randn(2, 1, 28, 28), torch.tensor([0, 999]))
    assert eps.shape == (2, 1, 28, 28) and torch.isfinite(eps).all()

    # Synthetic balanced labels exercise split sizing/disjointness without downloading data.
    labels = torch.arange(10).repeat_interleave(7000).numpy()
    split_a = make_stratified_split(labels, seed=42)
    split_b = make_stratified_split(labels, seed=42)
    assert all((split_a[k] == split_b[k]).all() for k in split_a)
    split_report = verify_split(split_a, labels)
    assert split_report["all_passed"]

    # Sanity-check project objective definitions.
    recon = torch.zeros_like(x)
    ae_loss = autoencoder_loss(recon, x)
    sparse_total, sparse_mse, sparse_l1 = sparse_autoencoder_loss(recon, x, torch.ones(2, 64))
    assert torch.isfinite(ae_loss) and torch.isfinite(sparse_total + sparse_mse + sparse_l1)
    dummy_mu = torch.zeros(2, 64)
    dummy_logvar = torch.zeros(2, 64)
    vae_total, vae_recon, vae_kl, _, _ = vae_loss(recon, x, dummy_mu, dummy_logvar, beta=1.0)
    assert torch.isfinite(vae_total + vae_recon + vae_kl)

    mlp, decoder = MLPClassifier(), FrozenLatentDecoder()
    with torch.no_grad():
        logits, z = mlp(x)
        recon = decoder(z)
    assert logits.shape == (2, 10) and z.shape == (2, 64)
    assert recon.shape == x.shape
    assert count_parameters(mlp) == 218_058
    assert count_parameters(decoder) == 218_128

    print("PASS: model shapes, finite outputs, parameter counts, split logic, and objectives.")
    print("NOTE: no training was performed; this is an architecture/forward-pass smoke test only.")


if __name__ == "__main__":
    main()
