"""
Training Monitoring Dashboard
Real-time monitoring of training progress
"""

import argparse
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))


def monitor_training(
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    refresh_interval: int = 30
):
    """
    Monitor training progress in real-time

    Args:
        log_dir: TensorBoard log directory
        checkpoint_dir: Checkpoint directory
        refresh_interval: Seconds between updates
    """
    print("="*70)
    print("GameNGen Training Monitor")
    print("="*70)
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Refresh interval: {refresh_interval}s")
    print("\nPress Ctrl+C to stop monitoring")
    print("="*70 + "\n")

    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)

    try:
        while True:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status Update:")
            print("-" * 70)

            # Check latest checkpoint
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pt"))
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

                    # Load checkpoint info
                    import torch
                    try:
                        ckpt = torch.load(latest, map_location='cpu')
                        step = ckpt.get('step', 'unknown')

                        print(f"Latest checkpoint: {latest.name}")
                        print(f"  Step: {step:,}" if isinstance(step, int) else f"  Step: {step}")
                        print(f"  Last modified: {time.strftime('%H:%M:%S', time.localtime(latest.stat().st_mtime))}")

                    except Exception as e:
                        print(f"Latest checkpoint: {latest.name} (couldn't load details)")
                else:
                    print("No checkpoints found yet")
            else:
                print("Checkpoint directory not found")

            # Check log files
            if log_dir.exists():
                log_files = list(log_dir.rglob("events.out.tfevents.*"))
                if log_files:
                    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                    print(f"\nTensorBoard log: {latest_log.parent.name}")
                    print(f"  Last update: {time.strftime('%H:%M:%S', time.localtime(latest_log.stat().st_mtime))}")
                else:
                    print("\nNo TensorBoard logs found yet")

            # GPU status (if available)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
                    print(f"\nGPU Status:")
                    print(f"  Utilization: {gpu_util}%")
                    print(f"  Memory: {mem_used}MB / {mem_total}MB")
            except:
                pass  # nvidia-smi not available

            print("-" * 70)
            print(f"Next update in {refresh_interval}s...")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval in seconds"
    )

    args = parser.parse_args()

    monitor_training(args.log_dir, args.checkpoint_dir, args.interval)


if __name__ == "__main__":
    main()
