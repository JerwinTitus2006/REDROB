"""
scorer.py — 5-pillar deterministic scoring engine.
Computes composite_score from pre-extracted features.
"""

from src.honeypot_detector import honeypot_multiplier


# Pillar weights (must sum to 1.0)
WEIGHTS = {
    "skill_fit": 0.30,
    "experience": 0.25,
    "redrob_signal": 0.25,
    "education": 0.10,
    "logistics": 0.10,
}


def compute_composite_score(features: dict, candidate_raw: dict = None) -> float:
    """
    Compute the final composite score from extracted features.
    
    Args:
        features: dict from feature_extractor.extract_features()
        candidate_raw: original candidate JSON (for honeypot detection)
    
    Returns:
        composite_score in [0.0, 1.0]
    """
    # Disqualifier gate
    if features.get("disqualified", False):
        return 0.0

    # 5-pillar weighted score
    composite = (
        features.get("skill_fit_score", 0) * WEIGHTS["skill_fit"] +
        features.get("experience_score", 0) * WEIGHTS["experience"] +
        features.get("redrob_signal_score", 0) * WEIGHTS["redrob_signal"] +
        features.get("education_score", 0) * WEIGHTS["education"] +
        features.get("logistics_score", 0) * WEIGHTS["logistics"]
    )

    # Honeypot penalty
    if candidate_raw is not None:
        hp_mult = honeypot_multiplier(candidate_raw)
        composite *= hp_mult

    return round(min(1.0, max(0.0, composite)), 4)


def score_all(features_list: list, candidates_raw: list = None) -> list:
    """
    Score all candidates. Returns list of (candidate_id, score, features).
    """
    raw_map = {}
    if candidates_raw:
        raw_map = {c["candidate_id"]: c for c in candidates_raw}

    results = []
    for feat in features_list:
        cid = feat["candidate_id"]
        raw = raw_map.get(cid)
        score = compute_composite_score(feat, raw)
        results.append((cid, score, feat))

    return results
