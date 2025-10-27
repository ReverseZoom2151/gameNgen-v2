"""
Visualize Recorded Gameplay Data
Helps understand what data was collected during RL training
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


def visualize_episode(episode: dict, output_path: Optional[str] = None):
    """
    Visualize a single episode

    Args:
        episode: Episode dictionary with frames, actions, rewards
        output_path: Optional path to save visualization
    """
    frames = episode["frames"]
    actions = episode["actions"]
    rewards = episode.get("rewards", [])

    num_frames = len(frames)

    print(f"Episode {episode.get('episode_id', 'unknown')}")
    print(f"  Frames: {num_frames}")
    print(f"  Total reward: {sum(rewards) if rewards else 'N/A'}")
    print(f"  Actions: {len(actions)}")

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Plot 1: Sample frames
    ax = axes[0]
    ax.set_title("Sample Frames from Episode")
    ax.axis("off")

    # Show 8 evenly spaced frames
    sample_indices = np.linspace(0, num_frames - 1, min(8, num_frames), dtype=int)
    sample_frames = [frames[i] for i in sample_indices]

    # Create horizontal montage
    if sample_frames:
        montage = np.concatenate(sample_frames, axis=1)  # Concatenate horizontally
        ax.imshow(montage)

    # Plot 2: Actions and rewards over time
    ax = axes[1]
    ax.set_title("Actions and Rewards Over Time")

    # Plot actions
    ax.plot(actions, label="Actions", alpha=0.7)

    if rewards:
        ax2 = ax.twinx()
        ax2.plot(rewards, "r-", label="Rewards", alpha=0.7)
        ax2.set_ylabel("Reward", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Action")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_dataset(data_dir: str, num_episodes_to_show: int = 5):
    """
    Analyze recorded dataset

    Args:
        data_dir: Directory with recorded episodes
        num_episodes_to_show: Number of episodes to visualize
    """
    data_dir = Path(data_dir)

    print("=" * 60)
    print("Dataset Analysis")
    print("=" * 60)
    print(f"Data directory: {data_dir}\n")

    # Find all batch files
    batch_files = sorted(data_dir.glob("batch_*.pkl"))

    if not batch_files:
        print(f"No data found in {data_dir}")
        return

    print(f"Found {len(batch_files)} batch files\n")

    # Load and analyze
    total_episodes = 0
    total_frames = 0
    episode_lengths = []
    total_rewards = []

    print("Loading data...")

    for batch_file in batch_files[
        : min(10, len(batch_files))
    ]:  # Sample first 10 batches
        with open(batch_file, "rb") as f:
            batch = pickle.load(f)

        for episode in batch:
            total_episodes += 1
            num_frames = len(episode["frames"])
            total_frames += num_frames
            episode_lengths.append(num_frames)

            if "rewards" in episode:
                total_rewards.append(sum(episode["rewards"]))

    print(f"\nDataset Statistics:")
    print(f"  Total episodes (sampled): {total_episodes}")
    print(f"  Total frames (sampled): {total_frames:,}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f} frames")
    print(f"  Std episode length: {np.std(episode_lengths):.1f} frames")

    if total_rewards:
        print(f"  Average reward: {np.mean(total_rewards):.2f}")
        print(f"  Std reward: {np.std(total_rewards):.2f}")

    # Visualize distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(episode_lengths, bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Episode Length (frames)")
    plt.ylabel("Count")
    plt.title("Episode Length Distribution")
    plt.grid(True, alpha=0.3)

    if total_rewards:
        plt.subplot(1, 2, 2)
        plt.hist(total_rewards, bins=30, alpha=0.7, edgecolor="black", color="green")
        plt.xlabel("Total Reward")
        plt.ylabel("Count")
        plt.title("Episode Reward Distribution")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=150)
    print(f"\n✓ Saved dataset analysis to dataset_analysis.png")
    plt.close()

    # Visualize sample episodes
    print(f"\nVisualizing {num_episodes_to_show} sample episodes...")

    episode_idx = 0
    for batch_file in batch_files[:3]:  # First 3 batches
        with open(batch_file, "rb") as f:
            batch = pickle.load(f)

        for episode in batch[:num_episodes_to_show]:
            visualize_episode(episode, f"episode_{episode_idx}.png")
            episode_idx += 1

            if episode_idx >= num_episodes_to_show:
                break

        if episode_idx >= num_episodes_to_show:
            break

    print(f"\n✓ Created {episode_idx} episode visualizations")


def main():
    parser = argparse.ArgumentParser(description="Visualize recorded gameplay data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/recordings",
        help="Directory with recorded episodes",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=5, help="Number of episodes to visualize"
    )

    args = parser.parse_args()

    analyze_dataset(args.data_dir, args.num_episodes)


if __name__ == "__main__":
    main()
