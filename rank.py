"""
rank.py — SINGLE ENTRY POINT for the ranking pipeline.
Runs within the 5-minute time constraint. No API calls.

Usage:
    python rank.py --candidates ./data/candidates.jsonl \
                   --jd ./data/job_description.docx \
                   --out ./outputs/submission.csv
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline, validate_submission


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranking Pipeline — Ranking Step"
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--jd", required=True,
        help="Path to job_description.docx"
    )
    parser.add_argument(
        "--out", default="./outputs/submission.csv",
        help="Output path for submission.csv"
    )
    parser.add_argument(
        "--artifacts", default="./artifacts",
        help="Directory for precomputed artifacts"
    )
    parser.add_argument(
        "--no-precomputed", action="store_true",
        help="Don't use precomputed features (extract fresh)"
    )
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Validate output after generation"
    )
    args = parser.parse_args()

    t_start = time.time()

    # Run the pipeline
    output_path = run_pipeline(
        candidates_path=args.candidates,
        jd_path=args.jd,
        output_path=args.out,
        artifacts_dir=args.artifacts,
        top_k=100,
        use_precomputed=not args.no_precomputed,
    )

    # Validate
    if args.validate:
        print("\n[rank.py] Validating submission...")
        errors = validate_submission(output_path, args.candidates)
        if errors:
            print("[rank.py] VALIDATION ERRORS:")
            for e in errors:
                print(f"  [FAIL] {e}")
            sys.exit(1)
        else:
            print("[rank.py] [OK] Submission is valid!")

    t_end = time.time()
    elapsed = t_end - t_start
    print(f"\n[rank.py] Total elapsed: {elapsed:.2f}s")
    if elapsed > 300:
        print("[rank.py] WARNING: Exceeded 5-minute limit!")
    else:
        print(f"[rank.py] [OK] Within 5-minute limit ({elapsed:.1f}s / 300s)")


if __name__ == "__main__":
    main()
