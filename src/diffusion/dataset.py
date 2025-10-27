"""
PyTorch Dataset for GameNGen
Loads recorded gameplay episodes and creates training samples
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class GameplayDataset(Dataset):
    """
    Dataset for training diffusion model on gameplay recordings

    Each sample contains:
    - context_frames: (context_length, 3, H, W) - past frames
    - context_actions: (context_length,) - past actions
    - target_frame: (3, H, W) - next frame to predict
    - target_action: (,) - action taken at target frame
    """

    def __init__(
        self,
        data_dir: str,
        context_length: int = 32,
        resolution: Tuple[int, int] = (256, 512),  # (H, W)
        max_trajectories: Optional[int] = None,
        cache_in_memory: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing recorded episodes
            context_length: Number of past frames to use as context
            resolution: Target resolution (H, W)
            max_trajectories: Maximum number of trajectories to load
            cache_in_memory: Whether to cache all data in RAM
        """
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.resolution = resolution
        self.max_trajectories = max_trajectories
        self.cache_in_memory = cache_in_memory

        # Load metadata if available
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            import json

            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata: {self.metadata}")
        else:
            self.metadata = {}

        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("batch_*.pkl"))

        if len(self.batch_files) == 0:
            raise ValueError(f"No batch files found in {self.data_dir}")

        print(f"Found {len(self.batch_files)} batch files")

        # Create trajectory index
        print("Creating trajectory index...")
        self.trajectories = self._create_trajectory_index()

        print(f"Dataset created with {len(self.trajectories)} trajectories")

        # Cache data if requested
        self.cache = {}
        if cache_in_memory:
            print("Caching data in memory...")
            self._cache_all_data()

    def _create_trajectory_index(self) -> List[Dict]:
        """
        Create index of all valid trajectories across all batches

        Returns:
            List of trajectory descriptors
        """
        trajectories = []

        for batch_idx, batch_file in enumerate(tqdm(self.batch_files, desc="Indexing")):
            # Load batch
            with open(batch_file, "rb") as f:
                batch = pickle.load(f)

            # Process each episode in batch
            for episode_idx, episode in enumerate(batch):
                frames = episode["frames"]
                actions = episode["actions"]

                # Create trajectories from this episode
                # Need at least context_length + 1 frames
                if len(frames) < self.context_length + 1:
                    continue

                # Create one trajectory for each valid position
                for start_idx in range(len(frames) - self.context_length):
                    trajectory = {
                        "batch_idx": batch_idx,
                        "batch_file": str(batch_file),
                        "episode_idx": episode_idx,
                        "start_idx": start_idx,
                        "episode_id": episode.get("episode_id", -1),
                    }
                    trajectories.append(trajectory)

                    # Stop if we've reached max trajectories
                    if (
                        self.max_trajectories
                        and len(trajectories) >= self.max_trajectories
                    ):
                        return trajectories

        return trajectories

    def _load_episode(self, batch_idx: int, episode_idx: int) -> Dict:
        """Load specific episode from disk"""
        # Check cache first
        cache_key = (batch_idx, episode_idx)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Load from disk
        batch_file = self.batch_files[batch_idx]
        with open(batch_file, "rb") as f:
            batch = pickle.load(f)

        episode = batch[episode_idx]

        # Cache if enabled
        if self.cache_in_memory:
            self.cache[cache_key] = episode

        return episode

    def _cache_all_data(self):
        """Cache all episodes in memory"""
        for traj in tqdm(self.trajectories, desc="Caching"):
            batch_idx = traj["batch_idx"]
            episode_idx = traj["episode_idx"]
            cache_key = (batch_idx, episode_idx)

            if cache_key not in self.cache:
                episode = self._load_episode(batch_idx, episode_idx)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame

        Args:
            frame: (H, W, 3) numpy array, values in [0, 255]
        Returns:
            (3, H, W) torch tensor, values in [0, 255] float32
        """
        # Convert to tensor
        frame = torch.from_numpy(frame).float()

        # Permute to (C, H, W)
        frame = frame.permute(2, 0, 1)

        # Resize if needed
        if frame.shape[1:] != self.resolution:
            import torchvision.transforms.functional as TF

            frame = TF.resize(
                frame.unsqueeze(0), self.resolution, antialias=True
            ).squeeze(0)

        return frame

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single training sample

        Returns:
            dict with:
            - context_frames: (context_length, 3, H, W)
            - context_actions: (context_length,)
            - target_frame: (3, H, W)
            - target_action: int
        """
        # Get trajectory info
        traj = self.trajectories[idx]

        # Load episode
        episode = self._load_episode(traj["batch_idx"], traj["episode_idx"])

        frames = episode["frames"]
        actions = episode["actions"]

        start_idx = traj["start_idx"]
        end_idx = start_idx + self.context_length

        # Get context frames and actions
        context_frames = frames[start_idx:end_idx]
        context_actions = actions[start_idx:end_idx]

        # Get target frame and action
        target_frame = frames[end_idx]
        target_action = actions[end_idx]

        # Preprocess frames
        context_frames = torch.stack(
            [self._preprocess_frame(frame) for frame in context_frames]
        )  # (T, 3, H, W)

        target_frame = self._preprocess_frame(target_frame)  # (3, H, W)

        # Convert actions to tensor
        context_actions = torch.from_numpy(np.array(context_actions, dtype=np.int64))

        target_action = torch.tensor(target_action, dtype=torch.long)

        return {
            "context_frames": context_frames,
            "context_actions": context_actions,
            "target_frame": target_frame,
            "target_action": target_action,
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    context_length: int = 32,
    resolution: Tuple[int, int] = (256, 512),
    num_workers: int = 4,
    shuffle: bool = True,
    max_trajectories: Optional[int] = None,
    cache_in_memory: bool = False,
) -> DataLoader:
    """
    Create dataloader for training

    Args:
        data_dir: Directory with recorded episodes
        batch_size: Batch size
        context_length: Number of past frames
        resolution: Target resolution (H, W)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        max_trajectories: Maximum number of trajectories
        cache_in_memory: Cache all data in RAM

    Returns:
        DataLoader
    """
    dataset = GameplayDataset(
        data_dir=data_dir,
        context_length=context_length,
        resolution=resolution,
        max_trajectories=max_trajectories,
        cache_in_memory=cache_in_memory,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return dataloader


def test_dataset(data_dir: str):
    """Test dataset loading"""
    print(f"Testing dataset with data from {data_dir}")

    # Create dataset
    dataset = GameplayDataset(
        data_dir=data_dir,
        context_length=32,
        resolution=(256, 512),
        max_trajectories=100,  # Limit for testing
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print("\nSample shapes:")
    print(f"  context_frames: {sample['context_frames'].shape}")
    print(f"  context_actions: {sample['context_actions'].shape}")
    print(f"  target_frame: {sample['target_frame'].shape}")
    print(f"  target_action: {sample['target_action']}")

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        max_trajectories=100,
    )

    print(f"\nDataloader created with {len(dataloader)} batches")

    # Get a batch
    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print(f"  context_frames: {batch['context_frames'].shape}")
    print(f"  context_actions: {batch['context_actions'].shape}")
    print(f"  target_frame: {batch['target_frame'].shape}")
    print(f"  target_action: {batch['target_action'].shape}")

    print("\nDataset test complete!")


if __name__ == "__main__":
    # Test with sample data
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/recordings"

    test_dataset(data_dir)
