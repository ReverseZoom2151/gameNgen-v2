"""
Human Evaluation Framework for GameNGen
Based on paper Section 5.1: Human Evaluation

"We provided 10 human raters with 130 random short clips (of lengths 1.6 seconds
and 3.2 seconds) of our simulation side by side with the real game."
"""

import random
from pathlib import Path
from typing import List, Optional
import json
import time
from dataclasses import dataclass, asdict


@dataclass
class EvaluationClip:
    """Single evaluation clip"""
    clip_id: int
    duration_seconds: float
    real_video_path: str
    fake_video_path: str
    real_is_on_left: bool  # Randomize which side is real


@dataclass
class EvaluationResult:
    """Result from single evaluation"""
    clip_id: int
    duration_seconds: float
    user_choice: str  # "left" or "right"
    correct: bool
    confidence: int  # 1-5 scale
    time_taken_seconds: float


class HumanEvaluationFramework:
    """
    Framework for conducting human evaluation studies

    Paper methodology:
    - 10 human raters
    - 130 clips (1.6s and 3.2s lengths)
    - Side-by-side comparison
    - Task: Identify which is the real game
    """

    def __init__(
        self,
        output_dir: str = "human_eval_results",
        clip_lengths: List[float] = [1.6, 3.2]
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clip_lengths = clip_lengths
        self.evaluation_clips: List[EvaluationClip] = []

    def create_evaluation_clips(
        self,
        real_videos: List[str],
        fake_videos: List[str],
        num_clips_per_length: int = 65  # 130 total / 2 lengths
    ) -> List[EvaluationClip]:
        """
        Create evaluation clip pairs

        Args:
            real_videos: Paths to real gameplay videos
            fake_videos: Paths to generated gameplay videos
            num_clips_per_length: Number of clips per duration

        Returns:
            List of evaluation clips
        """
        clips = []
        clip_id = 0

        for duration in self.clip_lengths:
            for _ in range(num_clips_per_length):
                # Randomly select videos
                real_video = random.choice(real_videos)
                fake_video = random.choice(fake_videos)

                # Randomize which side is real
                real_on_left = random.choice([True, False])

                clip = EvaluationClip(
                    clip_id=clip_id,
                    duration_seconds=duration,
                    real_video_path=real_video,
                    fake_video_path=fake_video,
                    real_is_on_left=real_on_left
                )

                clips.append(clip)
                clip_id += 1

        self.evaluation_clips = clips
        return clips

    def save_evaluation_protocol(self, filename: str = "evaluation_protocol.json"):
        """Save evaluation protocol for reproducibility"""
        protocol = {
            'num_clips': len(self.evaluation_clips),
            'clip_lengths': self.clip_lengths,
            'clips': [asdict(clip) for clip in self.evaluation_clips]
        }

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(protocol, f, indent=2)

        print(f"Saved evaluation protocol to {output_path}")

    def run_evaluation_session(
        self,
        evaluator_id: str,
        start_clip_idx: int = 0
    ) -> List[EvaluationResult]:
        """
        Run evaluation session with human rater

        Args:
            evaluator_id: Unique identifier for this evaluator
            start_clip_idx: Clip to start from (for resuming)

        Returns:
            List of evaluation results
        """
        print("="*70)
        print("GameNGen Human Evaluation Session")
        print("="*70)
        print(f"Evaluator ID: {evaluator_id}")
        print(f"Total clips: {len(self.evaluation_clips)}")
        print(f"Starting from clip: {start_clip_idx}")
        print("\nInstructions:")
        print("- You will see two videos side by side")
        print("- One is the real game, one is neural simulation")
        print("- Press 'L' if you think LEFT is real")
        print("- Press 'R' if you think RIGHT is real")
        print("- Press 'Q' to quit and save progress")
        print("="*70 + "\n")

        results = []

        for i in range(start_clip_idx, len(self.evaluation_clips)):
            clip = self.evaluation_clips[i]

            print(f"\nClip {i+1}/{len(self.evaluation_clips)} "
                  f"(Duration: {clip.duration_seconds}s)")

            start_time = time.time()

            # In actual implementation, would display videos side by side
            # For now, just simulate the interface
            print(f"  [Simulated] Showing clip {clip.clip_id}")
            print(f"  Left: {'REAL' if clip.real_is_on_left else 'FAKE'} (hidden from user)")
            print(f"  Right: {'FAKE' if clip.real_is_on_left else 'REAL'} (hidden from user)")

            # Simulate user input (in real implementation, would wait for keypress)
            print("\n  Press 'L' for left, 'R' for right, 'Q' to quit:")

            # For demo purposes, we'll skip actual input
            # In production, use: choice = input().lower()
            choice = "demo_mode"

            if choice == 'q':
                print("\nQuitting and saving progress...")
                break

            elapsed = time.time() - start_time

            # Record result
            if choice in ['l', 'r']:
                correct = (
                    (choice == 'l' and clip.real_is_on_left) or
                    (choice == 'r' and not clip.real_is_on_left)
                )

                # Ask for confidence (1-5)
                print("  Confidence (1=guessing, 5=certain):")
                confidence = 3  # Demo mode

                result = EvaluationResult(
                    clip_id=clip.clip_id,
                    duration_seconds=clip.duration_seconds,
                    user_choice=choice,
                    correct=correct,
                    confidence=confidence,
                    time_taken_seconds=elapsed
                )

                results.append(result)

        # Save results
        self.save_results(evaluator_id, results)

        # Print summary
        self.print_summary(results)

        return results

    def save_results(self, evaluator_id: str, results: List[EvaluationResult]):
        """Save evaluation results"""
        results_data = {
            'evaluator_id': evaluator_id,
            'timestamp': time.time(),
            'num_clips_evaluated': len(results),
            'results': [asdict(r) for r in results]
        }

        filename = self.output_dir / f"results_{evaluator_id}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to {filename}")

    def print_summary(self, results: List[EvaluationResult]):
        """Print evaluation summary"""
        if not results:
            return

        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / len(results) * 100

        # By duration
        durations = set(r.duration_seconds for r in results)
        duration_stats = {}

        for dur in durations:
            dur_results = [r for r in results if r.duration_seconds == dur]
            dur_correct = sum(1 for r in dur_results if r.correct)
            dur_accuracy = dur_correct / len(dur_results) * 100 if dur_results else 0
            duration_stats[dur] = {
                'count': len(dur_results),
                'correct': dur_correct,
                'accuracy': dur_accuracy
            }

        print("\n" + "="*70)
        print("Evaluation Summary:")
        print("="*70)
        print(f"Total clips evaluated: {len(results)}")
        print(f"Correct identifications: {correct_count}/{len(results)}")
        print(f"Overall accuracy: {accuracy:.1f}%")
        print(f"Paper reference: 58% (1.6s), 60% (3.2s)")
        print()

        for dur, stats in sorted(duration_stats.items()):
            print(f"{dur}s clips:")
            print(f"  Accuracy: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['count']})")

        print("="*70)

    def analyze_results(self, results_dir: Optional[str] = None) -> dict:
        """
        Analyze all evaluation results

        Args:
            results_dir: Directory with result JSON files

        Returns:
            Aggregated statistics
        """
        if results_dir is None:
            results_dir = self.output_dir

        results_dir = Path(results_dir)

        # Load all result files
        all_results = []
        evaluator_count = 0

        for result_file in results_dir.glob("results_*.json"):
            with open(result_file, 'r') as f:
                data = json.load(f)
                all_results.extend(data['results'])
                evaluator_count += 1

        if not all_results:
            print("No results found")
            return {}

        # Convert to EvaluationResult objects
        results = [
            EvaluationResult(**r) for r in all_results
        ]

        # Compute statistics
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        accuracy = correct / total * 100 if total > 0 else 0

        # By duration
        duration_stats = {}
        for dur in self.clip_lengths:
            dur_results = [r for r in results if abs(r.duration_seconds - dur) < 0.1]
            if dur_results:
                dur_correct = sum(1 for r in dur_results if r.correct)
                duration_stats[dur] = {
                    'total': len(dur_results),
                    'correct': dur_correct,
                    'accuracy': dur_correct / len(dur_results) * 100
                }

        stats = {
            'evaluators': evaluator_count,
            'total_clips': total,
            'correct': correct,
            'accuracy': accuracy,
            'by_duration': duration_stats,
        }

        # Print report
        print("\n" + "="*70)
        print("Human Evaluation Analysis")
        print("="*70)
        print(f"Evaluators: {evaluator_count}")
        print(f"Total evaluations: {total}")
        print(f"Overall accuracy: {accuracy:.1f}%")
        print(f"\nPaper results: 58% (1.6s), 60% (3.2s)")
        print()

        for dur, data in duration_stats.items():
            print(f"{dur}s clips: {data['accuracy']:.1f}% ({data['correct']}/{data['total']})")

        print("="*70)

        return stats


if __name__ == "__main__":
    # Demo usage
    print("Human Evaluation Framework Demo")

    framework = HumanEvaluationFramework()

    # Create dummy clips
    print("\nCreating evaluation protocol...")
    clips = framework.create_evaluation_clips(
        real_videos=["real1.mp4", "real2.mp4"],
        fake_videos=["fake1.mp4", "fake2.mp4"],
        num_clips_per_length=5
    )

    print(f"Created {len(clips)} evaluation clips")

    # Save protocol
    framework.save_evaluation_protocol()

    print("\nFramework ready!")
    print("To run actual evaluation, use:")
    print("  framework.run_evaluation_session('evaluator_001')")
