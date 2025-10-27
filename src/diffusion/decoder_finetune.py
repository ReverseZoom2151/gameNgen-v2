"""
VAE Decoder Fine-tuning for GameNGen
Improves visual quality, especially for HUD and small details (Section 3.2.2)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.diffusion.dataset import create_dataloader
from src.diffusion.model import ActionConditionedDiffusionModel


def finetune_decoder(config: dict):
    """
    Fine-tune VAE decoder on game frames

    Paper: Section 3.2.2 - "we train just the decoder of the latent
    auto-encoder using an MSE loss computed against the target frame pixels"
    """

    device = config.get("device", "cuda")

    print("=" * 60)
    print("VAE Decoder Fine-tuning for GameNGen")
    print("=" * 60)

    # Load model
    model = ActionConditionedDiffusionModel(
        pretrained_model_name=config["diffusion"]["pretrained_model"],
        num_actions=config["environment"].get("num_actions", 3),
        action_embedding_dim=config["diffusion"]["action_embedding_dim"],
        context_length=config["diffusion"]["context_length"],
        device=device,
        dtype=torch.float32,
    )

    # Unfreeze decoder only
    for param in model.vae.parameters():
        param.requires_grad = False

    for param in model.vae.decoder.parameters():
        param.requires_grad = True

    print(
        f"Trainable params: {sum(p.numel() for p in model.vae.decoder.parameters()):,}"
    )

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=config["data_dir"],
        batch_size=config["decoder"]["batch_size"],
        context_length=config["diffusion"]["context_length"],
        resolution=(
            config["environment"]["resolution"]["height"],
            config["environment"]["resolution"]["width"],
        ),
        num_workers=config.get("num_workers", 4),
        shuffle=True,
    )

    # Optimizer
    optimizer = optim.Adam(
        model.vae.decoder.parameters(), lr=config["decoder"]["learning_rate"]
    )

    # Training loop
    num_steps = config["decoder"]["num_steps"]
    global_step = 0

    pbar = tqdm(total=num_steps, desc="Fine-tuning decoder")

    dataloader_iter = iter(dataloader)

    while global_step < num_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Get target frames
        target_frames = batch["target_frame"].to(device)

        # Normalize to [-1, 1]
        target_frames = target_frames / 127.5 - 1.0

        # Encode to latents (frozen)
        with torch.no_grad():
            latents = model.vae.encode(target_frames).latent_dist.sample()
            latents = latents * model.vae.config.scaling_factor

        # Decode (trainable)
        reconstructed = model.vae.decode(
            latents / model.vae.config.scaling_factor
        ).sample

        # MSE loss
        loss = nn.functional.mse_loss(reconstructed, target_frames)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save checkpoint
        if global_step % 500 == 0:
            checkpoint_path = (
                Path(config["checkpoint_dir"]) / f"decoder_step_{global_step}.pt"
            )
            torch.save(
                {
                    "decoder": model.vae.decoder.state_dict(),
                    "step": global_step,
                },
                checkpoint_path,
            )
            print(f"\nSaved decoder checkpoint: {checkpoint_path}\n")

    pbar.close()

    # Save final decoder
    final_path = Path(config["checkpoint_dir"]) / "decoder_finetuned.pt"
    torch.save(
        {
            "decoder": model.vae.decoder.state_dict(),
            "step": global_step,
        },
        final_path,
    )

    print("\n" + "=" * 60)
    print(f"Decoder fine-tuning complete!")
    print(f"Saved to: {final_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VAE decoder")
    parser.add_argument("--config", type=str, default="configs/tier1_chrome_dino.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    finetune_decoder(config)


if __name__ == "__main__":
    main()
