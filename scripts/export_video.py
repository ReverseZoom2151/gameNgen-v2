"""
Batch Video Export Tool
Generate and export multiple gameplay videos for demonstration
"""

import argparse
from pathlib import Path
import sys
import torch
import yaml
import imageio
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.diffusion.model import ActionConditionedDiffusionModel
from src.diffusion.dataset import create_dataloader


def export_single_trajectory(
    model,
    initial_context_frames,
    initial_context_actions,
    actions_sequence,
    output_path: str,
    fps: int = 20,
):
    """
    Export single gameplay trajectory as video

    Args:
        model: Trained diffusion model
        initial_context_frames: Initial frames to start from
        initial_context_actions: Initial actions
        actions_sequence: Sequence of actions to execute
        output_path: Output video path
        fps: Frames per second
    """
    generated_frames = []

    # Initialize context
    context_frames = initial_context_frames.clone()
    context_actions = initial_context_actions.clone()

    print(f"Generating {len(actions_sequence)} frames...")

    for action_idx, action in enumerate(tqdm(actions_sequence)):
        # Generate next frame
        with torch.no_grad():
            generated_frame = model.generate(
                context_frames,
                context_actions,
                num_inference_steps=4,
                guidance_scale=1.5
            )

        # Convert to numpy (H, W, C)
        frame_np = generated_frame[0].permute(1, 2, 0).cpu().numpy()
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        generated_frames.append(frame_np)

        # Update context
        context_frames = torch.cat([
            context_frames[:, 1:],  # Remove oldest
            generated_frame.unsqueeze(1)  # Add newest
        ], dim=1)

        action_tensor = torch.tensor([[action]], device=model.device)
        context_actions = torch.cat([
            context_actions[:, 1:],
            action_tensor
        ], dim=1)

    # Save as video
    imageio.mimsave(output_path, generated_frames, fps=fps)
    print(f"\n✓ Saved video to {output_path} ({len(generated_frames)} frames @ {fps} FPS)")


def batch_export_videos(
    checkpoint_path: str,
    config_path: str,
    data_dir: str,
    num_videos: int = 10,
    video_length_frames: int = 300,  # 15 seconds @ 20 FPS
    output_dir: str = "exported_videos",
    fps: int = 20,
):
    """
    Batch export multiple videos

    Args:
        checkpoint_path: Path to trained model
        config_path: Config file
        data_dir: Data directory with trajectories
        num_videos: Number of videos to export
        video_length_frames: Length of each video
        output_dir: Output directory
        fps: Frames per second
    """
    print("="*60)
    print("Batch Video Export")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = ActionConditionedDiffusionModel(
        pretrained_model_name=config['diffusion']['pretrained_model'],
        num_actions=config['environment'].get('num_actions', 3),
        action_embedding_dim=config['diffusion']['action_embedding_dim'],
        context_length=config['diffusion']['context_length'],
        device=device,
        dtype=torch.float32,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.unet.load_state_dict(checkpoint['unet'])
    model.action_embedding.load_state_dict(checkpoint['action_embedding'])
    model.noise_aug_embedding.load_state_dict(checkpoint['noise_aug_embedding'])
    model.action_proj.load_state_dict(checkpoint['action_proj'])

    model.eval()
    print("✓ Model loaded")

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=1,
        context_length=config['diffusion']['context_length'],
        resolution=(
            config['environment']['resolution']['height'],
            config['environment']['resolution']['width']
        ),
        num_workers=0,
        shuffle=True,
        max_trajectories=num_videos,
    )

    print(f"\nExporting {num_videos} videos...")

    # Export videos
    for video_idx, batch in enumerate(dataloader):
        if video_idx >= num_videos:
            break

        print(f"\nVideo {video_idx + 1}/{num_videos}")

        context_frames = batch['context_frames'].to(device)
        context_actions = batch['context_actions'].to(device)

        # Generate random or sampled actions for the rest of the video
        # For demo, we'll sample random actions
        num_actions = config['environment'].get('num_actions', 3)
        actions_sequence = np.random.randint(0, num_actions, video_length_frames)

        # Export
        output_path = output_dir / f"gameplay_{video_idx:03d}.mp4"

        export_single_trajectory(
            model,
            context_frames,
            context_actions,
            actions_sequence,
            str(output_path),
            fps=fps
        )

    print("\n" + "="*60)
    print(f"✓ Exported {num_videos} videos to {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Batch export gameplay videos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint to use"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tier1_chrome_dino.yaml",
        help="Config file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/recordings",
        help="Data directory"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=10,
        help="Number of videos to export"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=300,
        help="Length of each video (frames)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_videos",
        help="Output directory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second"
    )

    args = parser.parse_args()

    batch_export_videos(
        args.checkpoint,
        args.config,
        args.data_dir,
        args.num_videos,
        args.length,
        args.output_dir,
        args.fps
    )


if __name__ == "__main__":
    main()
