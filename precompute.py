"""
precompute.py — Offline pre-computation script (no time limit).
Extracts features for all candidates and saves to artifacts/.

Usage:
    python precompute.py --candidates ./data/candidates.jsonl \
                         --jd ./data/job_description.docx
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from src.jd_parser import parse_jd
from src.feature_extractor import extract_features
from src.scorer import compute_composite_score
from src.honeypot_detector import honeypot_multiplier
from src.reasoning_generator import generate_reasoning


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Pre-computation — Extract features offline"
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--jd", required=True, help="Path to job_description.docx")
    parser.add_argument("--artifacts", default="./artifacts", help="Output dir")
    args = parser.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)
    t0 = time.time()

    # ── Step 1: Parse JD ────────────────────────────────────────────
    print("[precompute] Step 1: Parsing JD...")
    jd_out = os.path.join(args.artifacts, "jd_structured.json")
    parse_jd(args.jd, jd_out)

    # ── Step 2: Load all candidates ─────────────────────────────────
    print("[precompute] Step 2: Loading candidates...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"[precompute] Loaded {len(candidates)} candidates")

    # ── Step 3: Extract features ────────────────────────────────────
    print("[precompute] Step 3: Extracting features...")
    all_features = []
    for c in tqdm(candidates, desc="Feature extraction"):
        feat = extract_features(c)
        all_features.append(feat)

    features_path = os.path.join(args.artifacts, "candidate_features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(all_features, f, ensure_ascii=False)
    print(f"[precompute] Features saved to {features_path}")

    # ── Step 4: Score and rank to find top candidates ───────────────
    print("[precompute] Step 4: Scoring all candidates...")
    raw_map = {c["candidate_id"]: c for c in candidates}
    scored = []
    for feat in tqdm(all_features, desc="Scoring"):
        cid = feat["candidate_id"]
        raw = raw_map.get(cid)
        score = compute_composite_score(feat, raw)
        scored.append((cid, score, feat))

    scored.sort(key=lambda x: (-x[1], x[0]))

    # ── Step 5: Generate reasoning for top 200 ──────────────────────
    print("[precompute] Step 5: Generating reasoning for top 200...")
    reasonings = {}
    for rank_idx, (cid, score, feat) in enumerate(scored[:200]):
        rank = rank_idx + 1
        reasonings[cid] = generate_reasoning(rank, feat)

    reasonings_path = os.path.join(args.artifacts, "reasonings.json")
    with open(reasonings_path, "w", encoding="utf-8") as f:
        json.dump(reasonings, f, ensure_ascii=False, indent=2)
    print(f"[precompute] Reasonings saved to {reasonings_path}")

    # ── Summary ─────────────────────────────────────────────────────
    t_end = time.time()
    print(f"\n[precompute] Done in {t_end-t0:.1f}s")
    print(f"[precompute] Total candidates: {len(candidates)}")
    print(f"[precompute] Disqualified: {sum(1 for f in all_features if f.get('disqualified'))}")
    print(f"[precompute] Top 5:")
    for i, (cid, score, feat) in enumerate(scored[:5]):
        print(f"  {i+1}. {cid} score={score:.4f} "
              f"title={feat.get('current_title','')} "
              f"yoe={feat.get('yoe',0):.1f}")


if __name__ == "__main__":
    main()
