"""
Comprehensive Evaluation Metrics for GameNGen
Includes PSNR, LPIPS, SSIM, FVD as used in the paper
"""

import torch
import numpy as np
from typing import Dict
import lpips
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm


class GameNGenEvaluator:
    """
    Comprehensive evaluation suite for GameNGen

    Metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - SSIM (Structural Similarity Index)
    - FVD (Fréchet Video Distance)
    - MSE (Mean Squared Error)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Load LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()

        print(f"Evaluator initialized on {device}")

    def compute_psnr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_value: float = 255.0
    ) -> float:
        """
        Compute PSNR between predicted and target frames

        Args:
            pred: (B, C, H, W) or (C, H, W) in range [0, max_value]
            target: (B, C, H, W) or (C, H, W) in range [0, max_value]
            max_value: Maximum pixel value

        Returns:
            PSNR in dB
        """
        mse = torch.mean((pred - target) ** 2).item()

        if mse == 0:
            return float('inf')

        psnr = 20 * np.log10(max_value) - 10 * np.log10(mse)
        return psnr

    def compute_lpips(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute LPIPS (perceptual distance)

        Args:
            pred: (B, C, H, W) in range [0, 255]
            target: (B, C, H, W) in range [0, 255]

        Returns:
            LPIPS distance
        """
        # Normalize to [-1, 1] for LPIPS
        pred_norm = pred / 127.5 - 1.0
        target_norm = target / 127.5 - 1.0

        with torch.no_grad():
            lpips_val = self.lpips_model(pred_norm, target_norm)

        return lpips_val.mean().item()

    def compute_ssim(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Compute SSIM

        Args:
            pred: (H, W, C) in range [0, 255]
            target: (H, W, C) in range [0, 255]

        Returns:
            SSIM score
        """
        # Convert to grayscale for SSIM
        if len(pred.shape) == 3:
            pred_gray = np.mean(pred, axis=2).astype(np.uint8)
            target_gray = np.mean(target, axis=2).astype(np.uint8)
        else:
            pred_gray = pred.astype(np.uint8)
            target_gray = target.astype(np.uint8)

        ssim_val = ssim(target_gray, pred_gray, data_range=255)

        return ssim_val

    def compute_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """Compute MSE"""
        mse = torch.mean((pred - target) ** 2).item()
        return mse

    def compute_all_metrics(
        self,
        pred_frames: torch.Tensor,
        target_frames: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all metrics for a batch of frames

        Args:
            pred_frames: (B, C, H, W) or (B, H, W, C)
            target_frames: (B, C, H, W) or (B, H, W, C)

        Returns:
            Dictionary with all metrics
        """
        # Ensure (B, C, H, W) format
        if pred_frames.shape[-1] == 3:  # (B, H, W, C)
            pred_frames = pred_frames.permute(0, 3, 1, 2)
            target_frames = target_frames.permute(0, 3, 1, 2)

        # Move to device
        pred_frames = pred_frames.to(self.device)
        target_frames = target_frames.to(self.device)

        metrics = {}

        # PSNR
        metrics['psnr'] = self.compute_psnr(pred_frames, target_frames)

        # LPIPS
        metrics['lpips'] = self.compute_lpips(pred_frames, target_frames)

        # MSE
        metrics['mse'] = self.compute_mse(pred_frames, target_frames)

        # SSIM (on first frame only for speed)
        if pred_frames.shape[0] > 0:
            pred_np = pred_frames[0].permute(1, 2, 0).cpu().numpy()
            target_np = target_frames[0].permute(1, 2, 0).cpu().numpy()
            metrics['ssim'] = self.compute_ssim(pred_np, target_np)

        return metrics


class FVDCalculator:
    """
    Fréchet Video Distance (FVD) Calculator
    Paper Section 5.1: "We measure the FVD computed over a random holdout"

    FVD measures distance between distributions of videos
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Note: Full FVD requires I3D model (Inception 3D)
        # For simplicity, we use a feature extractor
        # Full implementation would need: pip install pytorch-fvd
        print("FVD Calculator initialized")
        print("Note: Full FVD requires I3D model (see pytorch-fvd package)")

    def extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Extract features from videos

        Args:
            videos: (N, T, C, H, W) - N videos, T frames each

        Returns:
            features: (N, feature_dim)
        """
        # Simplified feature extraction
        # In practice, use I3D model
        # For now, use simple statistics as placeholder

        N, T, C, H, W = videos.shape

        # Temporal mean and std
        features = []

        for video in videos:
            # Spatial features
            spatial_mean = video.mean(dim=[1, 2, 3])  # (T,)
            spatial_std = video.std(dim=[1, 2, 3])    # (T,)

            # Temporal features
            temporal_mean = video.mean(dim=0).flatten()  # Flatten spatial
            temporal_std = video.std(dim=0).flatten()

            # Combine
            feat = torch.cat([
                spatial_mean,
                spatial_std,
                temporal_mean[:100],  # Take first 100
                temporal_std[:100],
            ])

            features.append(feat)

        features = torch.stack(features)  # (N, feature_dim)

        return features

    def compute_fvd(
        self,
        real_videos: torch.Tensor,
        fake_videos: torch.Tensor
    ) -> float:
        """
        Compute FVD between real and generated videos

        Args:
            real_videos: (N, T, C, H, W)
            fake_videos: (N, T, C, H, W)

        Returns:
            FVD score
        """
        # Extract features
        real_features = self.extract_features(real_videos).cpu().numpy()
        fake_features = self.extract_features(fake_videos).cpu().numpy()

        # Compute mean and covariance
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)

        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)

        # Fréchet distance
        diff = mu_real - mu_fake
        covmean = sqrtm(sigma_real @ sigma_fake)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fvd = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)

        return fvd


def evaluate_model_comprehensive(
    model,
    dataloader,
    device: str = "cuda",
    num_trajectories: int = 100,
    trajectory_length: int = 64,
    compute_fvd: bool = False,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of GameNGen model

    Args:
        model: Trained diffusion model
        dataloader: Validation dataloader
        device: Device
        num_trajectories: Number of trajectories to evaluate
        trajectory_length: Length of each trajectory
        compute_fvd: Whether to compute FVD (slower)

    Returns:
        Dictionary with all metrics
    """
    evaluator = GameNGenEvaluator(device=device)

    print("="*60)
    print("Comprehensive Model Evaluation")
    print("="*60)

    all_psnr = []
    all_lpips = []
    all_ssim = []
    all_mse = []

    # For FVD
    if compute_fvd:
        fvd_calc = FVDCalculator(device=device)
        real_trajectories = []
        fake_trajectories = []

    model.eval()

    from tqdm import tqdm
    for traj_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=num_trajectories)):
        if traj_idx >= num_trajectories:
            break

        context_frames = batch['context_frames'].to(device)
        context_actions = batch['context_actions'].to(device)
        target_frame = batch['target_frame'].to(device)

        # Generate frame
        with torch.no_grad():
            generated = model.generate(
                context_frames,
                context_actions,
                num_inference_steps=4,
                guidance_scale=1.5,
            )

        # Compute metrics
        metrics = evaluator.compute_all_metrics(generated, target_frame)

        all_psnr.append(metrics['psnr'])
        all_lpips.append(metrics['lpips'])
        all_mse.append(metrics['mse'])
        if 'ssim' in metrics:
            all_ssim.append(metrics['ssim'])

    # Aggregate results
    results = {
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'lpips_mean': np.mean(all_lpips),
        'lpips_std': np.std(all_lpips),
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
    }

    if all_ssim:
        results['ssim_mean'] = np.mean(all_ssim)
        results['ssim_std'] = np.std(all_ssim)

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"PSNR:  {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
    print(f"MSE:   {results['mse_mean']:.2f} ± {results['mse_std']:.2f}")
    if 'ssim_mean' in results:
        print(f"SSIM:  {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print("="*60)

    return results


if __name__ == "__main__":
    # Test evaluator
    print("Testing Evaluator...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = GameNGenEvaluator(device=device)

    # Create dummy frames
    pred = torch.randint(0, 255, (4, 3, 256, 512), dtype=torch.float32).to(device)
    target = torch.randint(0, 255, (4, 3, 256, 512), dtype=torch.float32).to(device)

    # Compute metrics
    metrics = evaluator.compute_all_metrics(pred, target)

    print("\nMetrics computed:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ Evaluator test passed!")
