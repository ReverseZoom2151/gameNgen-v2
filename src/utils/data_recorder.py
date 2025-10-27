"""
Data Recording Utilities for GameNGen
Records episodes with frames and actions for training diffusion model
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
from tqdm import tqdm
import pickle


class EpisodeRecorder:
    """Records episodes with frames and actions"""

    def __init__(
        self,
        output_dir: str,
        compress: bool = True,
        save_frequency: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compress = compress
        self.save_frequency = save_frequency

        # Current episode buffer
        self.current_episode = {
            "frames": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        # Statistics
        self.episode_count = 0
        self.total_frames = 0

        # Batch buffer for efficient saving
        self.batch_buffer = []

    def add_step(
        self,
        frame: np.ndarray,
        action: int,
        reward: float,
        done: bool
    ):
        """Add single step to current episode"""
        self.current_episode["frames"].append(frame)
        self.current_episode["actions"].append(action)
        self.current_episode["rewards"].append(reward)
        self.current_episode["dones"].append(done)

        if done:
            self.finish_episode()

    def finish_episode(self):
        """Finish and save current episode"""
        if len(self.current_episode["frames"]) == 0:
            return

        # Add to batch buffer
        episode_data = {
            "frames": np.array(self.current_episode["frames"], dtype=np.uint8),
            "actions": np.array(self.current_episode["actions"], dtype=np.int32),
            "rewards": np.array(self.current_episode["rewards"], dtype=np.float32),
            "episode_id": self.episode_count,
            "length": len(self.current_episode["frames"]),
        }

        self.batch_buffer.append(episode_data)
        self.episode_count += 1
        self.total_frames += len(self.current_episode["frames"])

        # Save batch if frequency reached
        if len(self.batch_buffer) >= self.save_frequency:
            self._save_batch()

        # Reset current episode
        self.current_episode = {
            "frames": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

    def _save_batch(self):
        """Save batch of episodes to disk"""
        if len(self.batch_buffer) == 0:
            return

        batch_id = self.episode_count // self.save_frequency
        filename = self.output_dir / f"batch_{batch_id:06d}.pkl"

        # Save as pickle
        with open(filename, "wb") as f:
            pickle.dump(self.batch_buffer, f)

        print(f"Saved batch {batch_id} with {len(self.batch_buffer)} episodes to {filename}")

        # Clear buffer
        self.batch_buffer = []

    def finalize(self):
        """Save remaining data and create metadata"""
        # Save remaining episodes
        if len(self.batch_buffer) > 0:
            self._save_batch()

        # Create metadata
        metadata = {
            "total_episodes": self.episode_count,
            "total_frames": self.total_frames,
            "save_frequency": self.save_frequency,
            "compressed": self.compress,
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nRecording complete!")
        print(f"Total episodes: {self.episode_count}")
        print(f"Total frames: {self.total_frames}")
        print(f"Metadata saved to: {metadata_path}")


class DatasetLoader:
    """Load recorded episodes for training"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("batch_*.pkl"))
        print(f"Found {len(self.batch_files)} batch files")

    def load_batch(self, batch_idx: int) -> List[Dict[str, Any]]:
        """Load specific batch"""
        if batch_idx >= len(self.batch_files):
            raise IndexError(f"Batch {batch_idx} not found")

        with open(self.batch_files[batch_idx], "rb") as f:
            batch = pickle.load(f)

        return batch

    def iter_episodes(self, shuffle: bool = False):
        """Iterate over all episodes"""
        batch_indices = list(range(len(self.batch_files)))

        if shuffle:
            np.random.shuffle(batch_indices)

        for batch_idx in batch_indices:
            batch = self.load_batch(batch_idx)

            if shuffle:
                np.random.shuffle(batch)

            for episode in batch:
                yield episode

    def create_trajectory_dataset(
        self,
        context_length: int = 32,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create dataset of trajectories for diffusion model training

        Each trajectory contains:
        - context_frames: (context_length, H, W, C)
        - context_actions: (context_length,)
        - target_frame: (H, W, C)
        - target_action: int
        """
        trajectories = []

        print(f"Creating trajectory dataset with context length {context_length}...")

        for episode in tqdm(self.iter_episodes()):
            frames = episode["frames"]
            actions = episode["actions"]

            # Create trajectories from this episode
            for i in range(context_length, len(frames)):
                trajectory = {
                    "context_frames": frames[i - context_length : i],
                    "context_actions": actions[i - context_length : i],
                    "target_frame": frames[i],
                    "target_action": actions[i],
                    "episode_id": episode["episode_id"],
                }

                trajectories.append(trajectory)

        print(f"Created {len(trajectories)} trajectories")

        # Save if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                pickle.dump(trajectories, f)

            print(f"Saved trajectories to {output_path}")

        return trajectories


def visualize_episode(episode: Dict[str, Any], output_path: Optional[str] = None):
    """Visualize an episode as video"""
    import imageio

    frames = episode["frames"]
    actions = episode["actions"]

    # Add action labels to frames
    labeled_frames = []
    action_names = ["No Action", "Jump", "Duck"]

    for frame, action in zip(frames, actions):
        # Copy frame
        frame_labeled = frame.copy()

        # Add action text
        cv2.putText(
            frame_labeled,
            f"Action: {action_names[action]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        labeled_frames.append(frame_labeled)

    # Save as video if path provided
    if output_path:
        imageio.mimsave(output_path, labeled_frames, fps=20)
        print(f"Saved video to {output_path}")

    return labeled_frames


if __name__ == "__main__":
    # Test recorder
    print("Testing DataRecorder...")

    output_dir = "data/test_recordings"
    recorder = EpisodeRecorder(output_dir, save_frequency=2)

    # Simulate recording 5 episodes
    for ep in range(5):
        print(f"\nEpisode {ep + 1}")
        for step in range(50):
            # Dummy data
            frame = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            done = step == 49

            recorder.add_step(frame, action, reward, done)

    recorder.finalize()

    # Test loader
    print("\n" + "="*50)
    print("Testing DataLoader...")

    loader = DatasetLoader(output_dir)
    print(f"Metadata: {loader.metadata}")

    # Load first episode
    batch = loader.load_batch(0)
    print(f"First batch has {len(batch)} episodes")
    print(f"First episode shape: {batch[0]['frames'].shape}")

    print("\nTest complete!")
