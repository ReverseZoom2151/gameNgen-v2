"""
Compare Different Model Checkpoints
Evaluate quality across different training steps or configurations
"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.diffusion.dataset import create_dataloader
from src.diffusion.model import ActionConditionedDiffusionModel
from src.utils.evaluation import GameNGenEvaluator


def load_model(checkpoint_path: str, config: dict, device: str = "cuda"):
    """Load model from checkpoint"""

    model = ActionConditionedDiffusionModel(
        pretrained_model_name=config["diffusion"]["pretrained_model"],
        num_actions=config["environment"].get("num_actions", 3),
        action_embedding_dim=config["diffusion"]["action_embedding_dim"],
        context_length=config["diffusion"]["context_length"],
        device=device,
        dtype=torch.float32,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.unet.load_state_dict(checkpoint["unet"])
    model.action_embedding.load_state_dict(checkpoint["action_embedding"])
    model.noise_aug_embedding.load_state_dict(checkpoint["noise_aug_embedding"])
    model.action_proj.load_state_dict(checkpoint["action_proj"])

    model.eval()

    return model


def evaluate_checkpoint(
    model, dataloader, num_samples: int = 100, device: str = "cuda"
) -> dict:
    """Evaluate a single checkpoint"""

    evaluator = GameNGenEvaluator(device=device)

    all_metrics = []

    for i, batch in enumerate(tqdm(dataloader, total=num_samples, desc="Evaluating")):
        if i >= num_samples:
            break

        context_frames = batch["context_frames"].to(device)
        context_actions = batch["context_actions"].to(device)
        target_frame = batch["target_frame"].to(device)

        # Generate
        with torch.no_grad():
            generated = model.generate(
                context_frames,
                context_actions,
                num_inference_steps=4,
                guidance_scale=1.5,
            )

        # Compute metrics
        metrics = evaluator.compute_all_metrics(generated, target_frame)
        all_metrics.append(metrics)

    # Aggregate
    avg_metrics = {
        "psnr": np.mean([m["psnr"] for m in all_metrics]),
        "lpips": np.mean([m["lpips"] for m in all_metrics]),
        "mse": np.mean([m["mse"] for m in all_metrics]),
    }

    return avg_metrics


def compare_checkpoints(
    checkpoint_paths: List[str],
    config_path: str,
    data_dir: str,
    num_samples: int = 100,
    output_file: str = "model_comparison.csv",
):
    """
    Compare multiple checkpoints

    Args:
        checkpoint_paths: List of checkpoint paths
        config_path: Config file path
        data_dir: Data directory for evaluation
        num_samples: Number of samples to evaluate
        output_file: Output CSV file
    """
    print("=" * 60)
    print("Model Checkpoint Comparison")
    print("=" * 60)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=1,
        context_length=config["diffusion"]["context_length"],
        resolution=(
            config["environment"]["resolution"]["height"],
            config["environment"]["resolution"]["width"],
        ),
        num_workers=0,
        shuffle=True,
        max_trajectories=num_samples,
    )

    # Evaluate each checkpoint
    results = []

    for ckpt_path in checkpoint_paths:
        print(f"\nEvaluating: {ckpt_path}")

        try:
            model = load_model(ckpt_path, config, device)

            # Load step number from checkpoint
            ckpt = torch.load(ckpt_path, map_location="cpu")
            step = ckpt.get("step", 0)

            # Evaluate
            metrics = evaluate_checkpoint(model, dataloader, num_samples, device)

            result = {"checkpoint": Path(ckpt_path).name, "step": step, **metrics}

            results.append(result)

            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  LPIPS: {metrics['lpips']:.4f}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Create comparison table
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("step")

        print("\n" + "=" * 60)
        print("Comparison Results:")
        print("=" * 60)
        print(df.to_string(index=False))

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved comparison to {output_file}")

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # PSNR over steps
        axes[0].plot(df["step"], df["psnr"], marker="o")
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].set_title("PSNR vs Training Steps")
        axes[0].grid(True, alpha=0.3)

        # LPIPS over steps
        axes[1].plot(df["step"], df["lpips"], marker="o", color="orange")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("LPIPS")
        axes[1].set_title("LPIPS vs Training Steps")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=150)
        print(f"✓ Saved comparison plot to model_comparison.png")

    else:
        print("\n✗ No results to compare")


def main():
    parser = argparse.ArgumentParser(description="Compare model checkpoints")
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="Checkpoint files to compare"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tier1_chrome_dino.yaml",
        help="Config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/recordings",
        help="Data directory for evaluation",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output", type=str, default="model_comparison.csv", help="Output CSV file"
    )

    args = parser.parse_args()

    compare_checkpoints(
        args.checkpoints, args.config, args.data_dir, args.num_samples, args.output
    )


if __name__ == "__main__":
    main()
