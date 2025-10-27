"""
Resume Training Helper Script
Makes it easy to resume interrupted training from checkpoint
"""

import argparse
from pathlib import Path
import sys
import yaml
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> Path:
    """Find the most recent checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = list(checkpoint_dir.glob(pattern))

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    return latest


def resume_diffusion_training(
    config_path: str,
    checkpoint_dir: str = "checkpoints",
    additional_steps: Optional[int] = None
):
    """Resume diffusion model training from latest checkpoint"""

    print("="*60)
    print("Resuming Diffusion Training")
    print("="*60)

    # Find latest checkpoint
    try:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        print(f"\nFound checkpoint: {latest_checkpoint}")

        # Load checkpoint to get info
        import torch
        ckpt = torch.load(latest_checkpoint, map_location='cpu')

        print(f"Checkpoint step: {ckpt.get('step', 'unknown')}")

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        current_step = ckpt.get('step', 0)
        total_steps = config['diffusion']['num_train_steps']

        if additional_steps:
            total_steps = current_step + additional_steps

        remaining_steps = total_steps - current_step

        print(f"\nTraining progress:")
        print(f"  Current step: {current_step:,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Remaining: {remaining_steps:,}")
        print(f"  Progress: {current_step/total_steps*100:.1f}%")

        if remaining_steps <= 0:
            print("\n✓ Training already complete!")
            return

        print(f"\nResuming training for {remaining_steps:,} more steps...")
        print(f"\nCommand to run:")
        print(f"  python src/diffusion/train.py --config {config_path} --steps {total_steps}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNo checkpoint found. Start fresh training with:")
        print(f"  python src/diffusion/train.py --config {config_path}")


def resume_rl_training(checkpoint_dir: str = "checkpoints", env_type: str = "dqn"):
    """Resume RL agent training"""

    print("="*60)
    print("Resuming RL Agent Training")
    print("="*60)

    try:
        if env_type == "dqn":
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir, "dqn_*.pt")
            script = "src/agent/train_dqn.py"
        else:  # ppo
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir, "ppo_*.zip")
            script = "src/agent/train_ppo_doom.py"

        print(f"\nFound checkpoint: {latest_checkpoint}")
        print(f"\nTo resume, modify {script} to load from this checkpoint")
        print("Or continue training - it will create new episodes")

    except Exception as e:
        print(f"\n✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument(
        "type",
        choices=["diffusion", "rl-dqn", "rl-ppo"],
        help="Type of training to resume"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tier1_chrome_dino.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Additional steps to train"
    )

    args = parser.parse_args()

    if args.type == "diffusion":
        resume_diffusion_training(
            args.config,
            args.checkpoint_dir,
            args.steps
        )
    elif args.type == "rl-dqn":
        resume_rl_training(args.checkpoint_dir, "dqn")
    elif args.type == "rl-ppo":
        resume_rl_training(args.checkpoint_dir, "ppo")


if __name__ == "__main__":
    main()
