"""Training-step helpers for Model 07 classifier and frozen-latent decoder."""
import torch.nn.functional as F


def classifier_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def train_classifier_batch(model, images, labels, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, latent = model(images)
    loss = classifier_loss(logits, labels)
    loss.backward()
    optimizer.step()
    accuracy = (logits.detach().argmax(dim=1) == labels).float().mean()
    return {"cross_entropy": float(loss.detach()), "accuracy": float(accuracy),
            "latent": latent.detach()}


def train_frozen_decoder_batch(decoder, latent_features, target_images_flat, optimizer):
    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    reconstruction = decoder(latent_features).view(-1, target_images_flat.shape[-1])
    loss = F.mse_loss(reconstruction, target_images_flat, reduction="mean")
    loss.backward()
    optimizer.step()
    return {"mse": float(loss.detach()), "reconstruction": reconstruction.detach()}
