"""
tests/test_scorer.py — Unit tests for scoring logic.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_extractor import extract_features, count_must_have_clusters
from src.scorer import compute_composite_score, WEIGHTS
from src.honeypot_detector import honeypot_multiplier


def _make_candidate(**overrides):
    """Create a minimal valid candidate dict with overrides."""
    base = {
        "candidate_id": "CAND_9999999",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer",
            "summary": "ML engineer with Python, FAISS, embeddings experience.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 7,
            "current_title": "ML Engineer",
            "current_company": "TestCo",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [{
            "company": "TestCo",
            "title": "ML Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 78,
            "is_current": True,
            "industry": "Software",
            "company_size": "51-200",
            "description": "Built recommendation system using embeddings and FAISS. Deployed to production serving 1M users. Evaluated with NDCG and A/B tests.",
        }],
        "education": [{
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2013,
            "end_year": 2017,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        }],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 84},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 15, "duration_months": 30},
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 25, "duration_months": 48},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 85,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 3,
            "skill_assessment_scores": {"Python": 85, "FAISS": 70},
            "connection_count": 300,
            "endorsements_received": 50,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 65,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 10,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    # Apply overrides
    for key, val in overrides.items():
        if "." in key:
            parts = key.split(".")
            d = base
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = val
        else:
            base[key] = val
    return base


def test_pillar_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_disqualified_all_consulting_gets_zero():
    c = _make_candidate()
    c["career_history"] = [
        {"company": "TCS", "title": "Developer", "start_date": "2020-01-01",
         "end_date": None, "duration_months": 48, "is_current": True,
         "industry": "IT Services", "company_size": "10001+",
         "description": "Software development"},
        {"company": "Infosys", "title": "Developer", "start_date": "2018-01-01",
         "end_date": "2019-12-31", "duration_months": 24, "is_current": False,
         "industry": "IT Services", "company_size": "10001+",
         "description": "Software development"},
    ]
    feat = extract_features(c)
    assert feat["disqualified"] is True
    score = compute_composite_score(feat, c)
    assert score == 0.0


def test_inactive_180_days_has_low_redrob_score():
    c = _make_candidate()
    c["redrob_signals"]["last_active_date"] = "2025-10-01"
    feat = extract_features(c)
    assert feat["recency_score"] <= 0.30


def test_sentinel_github_minus_one_is_neutral():
    c = _make_candidate()
    c["redrob_signals"]["github_activity_score"] = -1
    feat = extract_features(c)
    assert feat["github_score"] == 0.45


def test_skill_expert_scores_higher_than_beginner():
    c_expert = _make_candidate()
    c_expert["skills"][0]["proficiency"] = "expert"
    c_beginner = _make_candidate()
    c_beginner["skills"][0]["proficiency"] = "beginner"
    f_expert = extract_features(c_expert)
    f_beginner = extract_features(c_beginner)
    assert f_expert["skill_fit_score"] > f_beginner["skill_fit_score"]


def test_notice_30_days_scores_higher_than_notice_120():
    c30 = _make_candidate()
    c30["redrob_signals"]["notice_period_days"] = 30
    c120 = _make_candidate()
    c120["redrob_signals"]["notice_period_days"] = 120
    f30 = extract_features(c30)
    f120 = extract_features(c120)
    assert f30["notice_score"] > f120["notice_score"]


def test_insufficient_experience_disqualified():
    c = _make_candidate()
    c["profile"]["years_of_experience"] = 2.0
    feat = extract_features(c)
    assert feat["disqualified"] is True
    assert feat["disqualify_reason"] == "insufficient_experience"


def test_ideal_yoe_gets_max_score():
    c = _make_candidate()
    c["profile"]["years_of_experience"] = 7.0
    feat = extract_features(c)
    assert feat["yoe_score"] == 1.0


def test_honeypot_yoe_gap():
    c = _make_candidate()
    c["profile"]["years_of_experience"] = 15
    c["career_history"] = [{
        "company": "TestCo", "title": "Dev", "start_date": "2023-01-01",
        "end_date": None, "duration_months": 12, "is_current": True,
        "industry": "Software", "company_size": "51-200",
        "description": "work"
    }]
    c["redrob_signals"]["profile_completeness_score"] = 20
    c["skills"] = [{"name": f"Skill{i}", "proficiency": "beginner",
                     "endorsements": 0, "duration_months": 1} for i in range(20)]
    mult = honeypot_multiplier(c)
    assert mult == 0.50
