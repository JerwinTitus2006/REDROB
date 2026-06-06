"""
pipeline.py — Orchestrates the full ranking pipeline end-to-end.
"""

import json
import os
import csv
import time
from pathlib import Path
from tqdm import tqdm

from src.feature_extractor import extract_features
from src.scorer import compute_composite_score
from src.honeypot_detector import honeypot_multiplier
from src.reasoning_generator import generate_reasoning


def load_candidates_jsonl(path: str) -> list:
    """Load candidates from a JSONL file."""
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def run_pipeline(
    candidates_path: str,
    jd_path: str,
    output_path: str,
    artifacts_dir: str = "artifacts",
    top_k: int = 100,
    use_precomputed: bool = True,
) -> str:
    """
    Run the full ranking pipeline.
    
    Args:
        candidates_path: path to candidates.jsonl
        jd_path: path to job_description.docx
        output_path: path for submission.csv
        artifacts_dir: directory for precomputed artifacts
        top_k: number of candidates to output (default 100)
        use_precomputed: whether to use precomputed features if available
    
    Returns:
        path to the output CSV file
    """
    t0 = time.time()
    print(f"[pipeline] Starting ranking pipeline...")
    print(f"[pipeline] Candidates: {candidates_path}")
    print(f"[pipeline] JD: {jd_path}")

    # ── Step 1: Load precomputed features or extract fresh ──────────
    features_path = os.path.join(artifacts_dir, "candidate_features.json")
    
    if use_precomputed and os.path.exists(features_path):
        print(f"[pipeline] Loading precomputed features from {features_path}...")
        with open(features_path, "r", encoding="utf-8") as f:
            all_features = json.load(f)
        print(f"[pipeline] Loaded {len(all_features)} candidate features")
        
        # Load raw candidates for honeypot detection
        print(f"[pipeline] Loading raw candidates for honeypot detection...")
        candidates = load_candidates_jsonl(candidates_path)
        raw_map = {c["candidate_id"]: c for c in candidates}
    else:
        print(f"[pipeline] No precomputed features found. Extracting fresh...")
        candidates = load_candidates_jsonl(candidates_path)
        print(f"[pipeline] Loaded {len(candidates)} candidates")
        
        raw_map = {c["candidate_id"]: c for c in candidates}
        all_features = []
        for c in tqdm(candidates, desc="Extracting features"):
            all_features.append(extract_features(c))

    t1 = time.time()
    print(f"[pipeline] Feature loading/extraction: {t1-t0:.2f}s")

    # ── Step 2: Score all candidates ────────────────────────────────
    print(f"[pipeline] Scoring {len(all_features)} candidates...")
    scored = []
    for feat in all_features:
        cid = feat["candidate_id"]
        raw = raw_map.get(cid)
        score = compute_composite_score(feat, raw)
        scored.append((cid, score, feat))

    t2 = time.time()
    print(f"[pipeline] Scoring: {t2-t1:.2f}s")

    # ── Step 3: Sort and select top-K ───────────────────────────────
    # Sort by score descending, then by candidate_id ascending for ties
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Select top-K
    top_k_results = scored[:top_k]

    # Ensure we have exactly top_k
    if len(top_k_results) < top_k:
        print(f"[pipeline] WARNING: Only {len(top_k_results)} candidates available, "
              f"need {top_k}. Padding with remaining candidates.")
        selected_ids = {r[0] for r in top_k_results}
        remaining = [(cid, sc, ft) for cid, sc, ft in scored
                      if cid not in selected_ids]
        remaining.sort(key=lambda x: (-x[1], x[0]))
        for r in remaining:
            if len(top_k_results) >= top_k:
                break
            top_k_results.append(r)

    t3 = time.time()
    print(f"[pipeline] Sorting: {t3-t2:.2f}s")

    # ── Step 4: Generate reasoning ──────────────────────────────────
    print(f"[pipeline] Generating reasoning for top {top_k}...")
    reasonings_path = os.path.join(artifacts_dir, "reasonings.json")
    precomputed_reasonings = {}
    if use_precomputed and os.path.exists(reasonings_path):
        with open(reasonings_path, "r", encoding="utf-8") as f:
            precomputed_reasonings = json.load(f)

    ranked_output = []
    for rank_idx, (cid, score, feat) in enumerate(top_k_results):
        rank = rank_idx + 1  # 1-indexed
        if cid in precomputed_reasonings:
            reasoning = precomputed_reasonings[cid]
        else:
            reasoning = generate_reasoning(rank, feat)
        ranked_output.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning,
        })

    t4 = time.time()
    print(f"[pipeline] Reasoning: {t4-t3:.2f}s")

    # ── Step 5: Enforce monotonically non-increasing scores ─────────
    for i in range(1, len(ranked_output)):
        if ranked_output[i]["score"] > ranked_output[i-1]["score"]:
            ranked_output[i]["score"] = ranked_output[i-1]["score"]

    # ── Step 6: Write CSV ───────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in ranked_output:
            writer.writerow(row)

    t5 = time.time()
    print(f"[pipeline] CSV written to {output_path}")
    print(f"[pipeline] Total time: {t5-t0:.2f}s")
    print(f"[pipeline] Top 5 candidates:")
    for row in ranked_output[:5]:
        print(f"  Rank {row['rank']}: {row['candidate_id']} "
              f"(score={row['score']:.4f}) — {row['reasoning'][:80]}")

    return output_path


def validate_submission(csv_path: str, jsonl_path: str) -> list:
    """Validate submission CSV format. Returns list of error strings."""
    errors = []
    
    # Load valid candidate IDs
    valid_ids = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                valid_ids.add(c["candidate_id"])

    # Read CSV
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Check row count
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    # Check ranks
    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks must be 1-100, each used exactly once")

    # Check unique candidate IDs
    cids = [r["candidate_id"] for r in rows]
    if len(set(cids)) != len(cids):
        errors.append("Duplicate candidate_id found")

    # Check all IDs exist
    for cid in cids:
        if cid not in valid_ids:
            errors.append(f"candidate_id {cid} not in candidates.jsonl")

    # Check scores monotonically non-increasing
    scores = [float(r["score"]) for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1]:
            errors.append(f"Scores not monotonically non-increasing at rank {i+1}")
            break

    # Check score range
    for i, s in enumerate(scores):
        if s < 0 or s > 1:
            errors.append(f"Score {s} at rank {i+1} outside [0, 1]")

    return errors
