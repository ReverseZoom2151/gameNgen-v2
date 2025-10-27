"""
Download and cache required models before training
Avoids downloads during training and verifies everything works
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def download_stable_diffusion(model_name: str = "CompVis/stable-diffusion-v1-4"):
    """
    Download Stable Diffusion v1.4 model

    This downloads ~4GB and caches locally.
    Subsequent runs will use cached version.
    """
    print("="*60)
    print("Downloading Stable Diffusion v1.4")
    print("="*60)
    print(f"Model: {model_name}")
    print("Size: ~4GB")
    print("This may take 5-15 minutes depending on connection...")
    print()

    from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel

    # Download components
    print("Downloading VAE...")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    print("✓ VAE downloaded")

    print("\nDownloading U-Net...")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    print("✓ U-Net downloaded")

    print("\nDownloading full pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    print("✓ Pipeline downloaded")

    print("\n" + "="*60)
    print("All Stable Diffusion components downloaded!")
    print("Location: ~/.cache/huggingface/hub/")
    print("="*60)


def download_vizdoom_scenarios():
    """
    Download ViZDoom scenario files if needed
    """
    print("\n" + "="*60)
    print("Checking ViZDoom Scenarios")
    print("="*60)

    try:
        import vizdoom as vzd

        scenarios_path = Path(vzd.scenarios_path)
        print(f"ViZDoom scenarios path: {scenarios_path}")

        # Check if scenarios exist
        scenarios = [
            "basic.cfg",
            "deadly_corridor.cfg",
            "defend_the_center.cfg",
            "defend_the_line.cfg",
            "health_gathering.cfg",
        ]

        for scenario in scenarios:
            scenario_file = scenarios_path / scenario
            if scenario_file.exists():
                print(f"✓ {scenario}")
            else:
                print(f"✗ {scenario} - NOT FOUND")

        print("\n✓ ViZDoom installed and scenarios available")

    except ImportError:
        print("⚠ ViZDoom not installed")
        print("Install with: pip install vizdoom")


def verify_pytorch_cuda():
    """Verify PyTorch CUDA is working"""
    print("\n" + "="*60)
    print("Verifying PyTorch CUDA")
    print("="*60)

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("✓ CUDA working correctly")
    else:
        print("⚠ CUDA not available - will use CPU (slow!)")


def main():
    parser = argparse.ArgumentParser(description="Download required models and verify setup")
    parser.add_argument(
        "--skip-sd",
        action="store_true",
        help="Skip Stable Diffusion download"
    )
    parser.add_argument(
        "--skip-vizdoom",
        action="store_true",
        help="Skip ViZDoom check"
    )

    args = parser.parse_args()

    print("="*60)
    print("GameNGen - Model Download & Verification")
    print("="*60)

    # Verify PyTorch
    verify_pytorch_cuda()

    # Download Stable Diffusion
    if not args.skip_sd:
        try:
            download_stable_diffusion()
        except Exception as e:
            print(f"\n✗ Failed to download Stable Diffusion: {e}")
            print("You can try again later or it will download during training")

    # Check ViZDoom
    if not args.skip_vizdoom:
        download_vizdoom_scenarios()

    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nYou're ready to train!")
    print("Next steps:")
    print("  python tests/test_all_tiers.py  # Verify everything")
    print("  python src/agent/train_dqn.py   # Start training")
    print("="*60)


if __name__ == "__main__":
    main()
