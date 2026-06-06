"""
feature_extractor.py — Extract scoring features from raw candidate JSON.
"""
import json
import os
from datetime import date
from rapidfuzz import fuzz


# ── Skill relevance scores ──────────────────────────────────────────────
SKILL_SCORES = {
    "sentence-transformers": 0.20, "faiss": 0.20, "elasticsearch": 0.18,
    "opensearch": 0.18, "pinecone": 0.18, "weaviate": 0.18, "qdrant": 0.18,
    "milvus": 0.17, "vector database": 0.19, "embeddings": 0.19,
    "dense retrieval": 0.20, "semantic search": 0.18, "rag": 0.17,
    "hybrid search": 0.18, "bm25": 0.15, "ndcg": 0.18,
    "information retrieval": 0.18, "ranking": 0.16,
    "recommendation systems": 0.16, "python": 0.15,
    "lora": 0.12, "qlora": 0.12, "peft": 0.12, "fine-tuning llms": 0.12,
    "llm": 0.10, "langchain": 0.06, "xgboost": 0.10,
    "learning to rank": 0.13, "pytorch": 0.10, "transformer": 0.10,
    "nlp": 0.12, "mlflow": 0.08, "distributed systems": 0.09,
    "tensorflow": 0.09, "keras": 0.08, "scikit-learn": 0.07,
    "hugging face": 0.12, "huggingface": 0.12, "bert": 0.12,
    "gpt": 0.08, "llamaindex": 0.06, "llama-index": 0.06,
    "react": 0.01, "figma": 0.0, "six sigma": 0.0, "sap": 0.0,
    "accounting": 0.0, "content writing": 0.0, "sales": 0.0,
    "marketing": 0.0, "photoshop": 0.0, "powerpoint": 0.0,
    "excel": 0.0, "word": 0.0,
}

PROFICIENCY_MULT = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.65, "beginner": 0.40}

# ── Must-have clusters ──────────────────────────────────────────────────
MUST_HAVE_CLUSTERS = {
    "embeddings_retrieval": [
        "sentence-transformers", "embeddings", "vector similarity",
        "dense retrieval", "semantic search", "rag", "bge", "e5",
        "cosine similarity", "ann", "approximate nearest neighbor",
        "embedding", "two-tower", "dense passage retrieval",
    ],
    "vector_db_hybrid_search": [
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "vector database", "hybrid search",
        "bm25", "vector index", "vector store",
    ],
    "python_production": ["python"],
    "ranking_evaluation": [
        "ndcg", "mrr", "map", "a/b test", "evaluation framework",
        "ranking metrics", "recall@", "precision@", "offline evaluation",
    ],
}

CAREER_SIGNALS = {
    "recommendation system": 0.18, "search system": 0.17,
    "ranking model": 0.18, "retrieval system": 0.18,
    "vector index": 0.16, "embedding model": 0.17,
    "deployed to production": 0.05, "billion scale": 0.08,
    "million users": 0.06, "low latency": 0.04, "a/b test": 0.06,
    "recall@": 0.10, "ndcg": 0.12, "mrr": 0.10,
    "offline evaluation": 0.08, "feature store": 0.05,
    "two-tower model": 0.15, "dense passage retrieval": 0.15,
    "approximate nearest neighbor": 0.15, "ann": 0.10,
    "langchain tutorial": -0.05, "chatgpt wrapper": -0.05,
    "hackathon project": -0.03,
}

INDUSTRY_RELEVANCE = {
    "ai/ml": 1.0, "artificial intelligence": 1.0, "machine learning": 1.0,
    "software": 0.90, "technology": 0.90, "fintech": 0.85,
    "financial services": 0.80, "edtech": 0.80, "education technology": 0.80,
    "e-commerce": 0.80, "ecommerce": 0.80, "food delivery": 0.75,
    "food technology": 0.75, "healthcare": 0.70, "conglomerate": 0.50,
    "it services": 0.35, "consulting": 0.25, "transportation": 0.60,
    "logistics": 0.60, "manufacturing": 0.20, "paper products": 0.10,
    "retail": 0.55, "media": 0.50, "entertainment": 0.50,
    "telecommunications": 0.55, "telecom": 0.55, "banking": 0.65,
    "insurance": 0.45, "real estate": 0.25, "construction": 0.15,
    "agriculture": 0.15, "automotive": 0.40, "aerospace": 0.50,
    "defense": 0.45, "energy": 0.30, "pharmaceuticals": 0.35,
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "l&t infotech",
}

LOCATION_SCORES = {
    "pune": 1.0, "noida": 1.0, "hyderabad": 0.90, "mumbai": 0.85,
    "delhi": 0.85, "new delhi": 0.85, "gurgaon": 0.85, "gurugram": 0.85,
    "bangalore": 0.85, "bengaluru": 0.85, "chennai": 0.75,
    "chandigarh": 0.75, "ahmedabad": 0.75, "jaipur": 0.75,
    "indore": 0.70, "kochi": 0.65, "trivandrum": 0.60,
    "thiruvananthapuram": 0.60, "bhubaneswar": 0.60, "kolkata": 0.65,
    "lucknow": 0.60, "nagpur": 0.60, "coimbatore": 0.55,
}

WORK_MODE_COMPAT = {"hybrid": 1.0, "flexible": 0.95, "onsite": 0.80, "remote": 0.50}

TIER_SCORES = {"tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.60, "tier_4": 0.40, "unknown": 0.50}

FIELD_RELEVANCE = {
    "computer science": 1.0, "computer engineering": 1.0,
    "information technology": 0.95, "data science": 1.0,
    "artificial intelligence": 1.0, "ai": 1.0, "machine learning": 1.0,
    "electrical engineering": 0.80, "electronics": 0.75,
    "electronics and communication": 0.75,
    "mathematics": 0.85, "statistics": 0.85, "applied mathematics": 0.85,
    "physics": 0.60, "mechanical engineering": 0.20,
    "civil engineering": 0.10, "chemical engineering": 0.10,
    "accounting": 0.05, "marketing": 0.05, "commerce": 0.05,
    "business administration": 0.10, "mba": 0.15,
}

CERT_BONUSES = {
    "aws certified machine learning": 0.08,
    "google cloud professional ml engineer": 0.08,
    "deep learning specialization": 0.07,
    "nlp specialization": 0.07,
    "tensorflow developer certificate": 0.06,
    "aws certified cloud practitioner": 0.03,
    "scrum master certified": 0.01,
    "six sigma": 0.0,
}

NON_TECH_TITLES = {
    "accountant", "hr manager", "civil engineer", "mechanical engineer",
    "graphic designer", "content writer", "marketing manager",
    "customer support", "operations manager", "sales executive",
    "sales manager", "sales",
}

TECH_KEYWORDS_IN_TITLE = {
    "engineer", "developer", "scientist", "analyst", "ml", "ai",
    "data", "software", "devops", "sre", "platform", "backend",
    "frontend", "fullstack", "machine learning", "deep learning",
    "research", "architect",
}

PRODUCTION_SIGNALS = [
    "deployed to production", "served", "requests per",
    "reduced latency", "improved recall", "a/b test", "shipped",
    "at scale", "production system", "million users", "production ml",
    "real-time", "low latency", "end-to-end",
]

RESEARCH_ONLY_SIGNALS = [
    "research paper", "academic", "proof of concept",
    "prototype only", "thesis",
]


def _fuzzy_match_skill(skill_name: str, reference: str) -> bool:
    """Check if a skill name fuzzy-matches a reference term."""
    sn = skill_name.lower().strip()
    ref = reference.lower().strip()
    if ref in sn or sn in ref:
        return True
    return fuzz.ratio(sn, ref) >= 82


def _get_skill_base_score(skill_name: str) -> float:
    """Get the base relevance score for a skill using fuzzy matching."""
    sn = skill_name.lower().strip()
    if sn in SKILL_SCORES:
        return SKILL_SCORES[sn]
    for ref_skill, score in SKILL_SCORES.items():
        if _fuzzy_match_skill(sn, ref_skill):
            return score
    return 0.0


def _get_assessment_multiplier(skill_name: str, assessments: dict) -> float:
    """Get assessment score multiplier for a skill."""
    if not assessments:
        return 1.0
    sn = skill_name.lower().strip()
    for aname, ascore in assessments.items():
        if _fuzzy_match_skill(sn, aname.lower()):
            if ascore >= 80:
                return 1.25
            elif ascore >= 60:
                return 1.10
            else:
                return 0.90
    return 1.0


def count_must_have_clusters(candidate: dict) -> int:
    """Count how many of the 4 must-have skill clusters the candidate has."""
    all_text = _build_candidate_text(candidate).lower()
    count = 0
    for cluster_name, keywords in MUST_HAVE_CLUSTERS.items():
        for kw in keywords:
            if kw in all_text:
                count += 1
                break
    return count


def _build_candidate_text(candidate: dict) -> str:
    """Build a combined text block from all candidate fields."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(parts)


def _scan_career_descriptions(career_history: list) -> float:
    """Scan career descriptions for skill signals not in skills[]."""
    combined = " ".join(
        c.get("description", "").lower() for c in career_history
    )
    bonus = 0.0
    for signal, value in CAREER_SIGNALS.items():
        if signal in combined:
            bonus += value
    return min(0.25, max(-0.10, bonus))


def _is_langchain_only(candidate: dict) -> bool:
    """Check if candidate's only AI skills are LangChain/LlamaIndex wrappers."""
    skill_names = [s["name"].lower().strip() for s in candidate.get("skills", [])]
    wrapper_skills = {"langchain", "llamaindex", "llama-index", "llama index"}
    real_ai_skills = {
        "pytorch", "tensorflow", "keras", "scikit-learn", "faiss",
        "sentence-transformers", "embeddings", "ndcg", "xgboost",
        "recommendation systems", "ranking", "information retrieval",
        "nlp", "transformer", "bert", "dense retrieval", "semantic search",
        "learning to rank", "peft", "lora", "qlora", "fine-tuning llms",
        "milvus", "pinecone", "weaviate", "qdrant", "elasticsearch",
        "opensearch", "vector database", "hybrid search",
    }
    has_wrapper = any(sn in wrapper_skills for sn in skill_names)
    has_real = any(
        any(_fuzzy_match_skill(sn, rai) for rai in real_ai_skills)
        for sn in skill_names
    )
    # Also check career descriptions
    combined_desc = " ".join(
        c.get("description", "").lower()
        for c in candidate.get("career_history", [])
    )
    has_real_career = any(
        sig in combined_desc
        for sig in ["pytorch", "custom retrieval", "evaluation", "ndcg",
                     "ranking model", "retrieval system", "deployed"]
    )
    return has_wrapper and not has_real and not has_real_career


def _get_location_score(candidate: dict) -> float:
    """Compute location fitness score."""
    profile = candidate.get("profile", {})
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    best = 0.30  # default for unknown locations
    for city, score in LOCATION_SCORES.items():
        if city in loc:
            best = max(best, score)

    if "india" not in country and best < 0.50:
        best = 0.50  # outside India case-by-case

    if willing and best < 0.85:
        best = max(best, 0.85 * 0.7 + best * 0.3)  # boost toward 0.85

    return min(1.0, best)


def _get_field_relevance(field: str) -> float:
    """Get field of study relevance score using fuzzy matching."""
    fl = field.lower().strip()
    if fl in FIELD_RELEVANCE:
        return FIELD_RELEVANCE[fl]
    best = 0.15  # default for unmatched fields
    for ref, score in FIELD_RELEVANCE.items():
        if ref in fl or fl in ref:
            best = max(best, score)
        elif fuzz.ratio(fl, ref) >= 80:
            best = max(best, score)
    return best


def extract_features(candidate: dict) -> dict:
    """
    Extract all scoring features for a single candidate.
    Returns a flat dict of features used by the scorer.
    """
    cid = candidate["candidate_id"]
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})

    features = {"candidate_id": cid}

    # ── Disqualifier checks ─────────────────────────────────────────
    companies = {c.get("company", "").lower().strip() for c in career}
    is_all_consulting = (
        len(companies) > 0 and
        companies.issubset(CONSULTING_FIRMS)
    )
    must_have_count = count_must_have_clusters(candidate)
    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "").lower().strip()

    is_non_tech_title = current_title in NON_TECH_TITLES
    has_tech_history = any(
        any(t in c.get("title", "").lower() for t in TECH_KEYWORDS_IN_TITLE)
        for c in career
    )

    disqualified = False
    disqualify_reason = ""
    if is_all_consulting:
        disqualified = True
        disqualify_reason = "entire_career_at_consulting_firms"
    elif must_have_count == 0:
        disqualified = True
        disqualify_reason = "zero_must_have_skill_clusters"
    elif yoe < 3:
        disqualified = True
        disqualify_reason = "insufficient_experience"
    elif is_non_tech_title and not has_tech_history:
        disqualified = True
        disqualify_reason = "non_technical_with_no_tech_history"

    features["disqualified"] = disqualified
    features["disqualify_reason"] = disqualify_reason
    features["must_have_clusters"] = must_have_count

    # ── PILLAR 1: Skill Fit ─────────────────────────────────────────
    raw_skill_score = 0.0
    for skill in skills:
        sname = skill.get("name", "")
        prof = skill.get("proficiency", "beginner")
        base = _get_skill_base_score(sname)
        prof_mult = PROFICIENCY_MULT.get(prof, 0.40)
        assess_mult = _get_assessment_multiplier(sname, assessments)
        raw_skill_score += base * prof_mult * assess_mult

    career_bonus = _scan_career_descriptions(career)
    skill_fit_score = min(1.0, (raw_skill_score + career_bonus) / 1.5)

    if _is_langchain_only(candidate):
        skill_fit_score *= 0.70

    features["raw_skill_score"] = round(raw_skill_score, 4)
    features["career_bonus"] = round(career_bonus, 4)
    features["skill_fit_score"] = round(skill_fit_score, 4)
    features["is_langchain_only"] = _is_langchain_only(candidate)

    # ── PILLAR 2: Experience ────────────────────────────────────────
    # Sub 1: YoE fit
    if 6 <= yoe <= 8:
        yoe_score = 1.0
    elif 5 <= yoe <= 9:
        yoe_score = 0.85
    elif 4 <= yoe < 5:
        yoe_score = 0.70
    elif 9 < yoe <= 12:
        yoe_score = 0.75
    elif yoe > 12:
        yoe_score = 0.60
    else:
        yoe_score = 0.30

    # Sub 2: Domain/industry relevance
    total_months = 0
    weighted_industry = 0.0
    product_months = 0
    for job in career:
        dur = job.get("duration_months", 0)
        ind = job.get("industry", "").lower().strip()
        comp = job.get("company", "").lower().strip()
        ind_score = INDUSTRY_RELEVANCE.get(ind, 0.40)
        weighted_industry += ind_score * dur
        total_months += dur
        if comp not in CONSULTING_FIRMS:
            product_months += dur

    domain_score = weighted_industry / max(total_months, 1)
    if product_months > 0 and total_months > 0:
        product_ratio = product_months / total_months
        if product_ratio > 0.5:
            domain_score = min(1.0, domain_score + 0.10)

    # Sub 3: Title trajectory
    relevant_titles = {
        "ml engineer", "machine learning engineer", "ai engineer",
        "data scientist", "senior data scientist", "nlp engineer",
        "search engineer", "recommendation engineer", "research engineer",
        "software engineer", "senior software engineer", "backend engineer",
        "senior ml engineer", "senior ai engineer", "staff engineer",
        "principal engineer", "lead engineer", "tech lead",
        "senior machine learning engineer", "junior ml engineer",
    }
    title_score = 0.40  # default
    if current_title in relevant_titles:
        title_score = 0.95
    elif any(t in current_title for t in ["engineer", "developer", "scientist"]):
        title_score = 0.75
    elif any(t in current_title for t in ["analyst", "architect", "researcher"]):
        title_score = 0.65

    # Narrative modifier
    combined_desc = " ".join(c.get("description", "").lower() for c in career)
    narrative_mod = 1.0
    prod_count = sum(1 for sig in PRODUCTION_SIGNALS if sig in combined_desc)
    research_count = sum(1 for sig in RESEARCH_ONLY_SIGNALS if sig in combined_desc)
    if prod_count >= 2:
        narrative_mod = min(1.10, 1.0 + prod_count * 0.02)
    if research_count >= 2 and prod_count == 0:
        narrative_mod = max(0.90, narrative_mod - 0.05)

    experience_score = (
        yoe_score * 0.35 +
        domain_score * 0.40 +
        title_score * 0.25
    ) * narrative_mod
    experience_score = min(1.0, experience_score)

    features["yoe"] = yoe
    features["yoe_score"] = round(yoe_score, 4)
    features["domain_score"] = round(domain_score, 4)
    features["title_score"] = round(title_score, 4)
    features["narrative_modifier"] = round(narrative_mod, 4)
    features["experience_score"] = round(experience_score, 4)
    features["current_title"] = profile.get("current_title", "")
    features["current_company"] = profile.get("current_company", "")
    features["current_industry"] = profile.get("current_industry", "")

    # ── PILLAR 3: Redrob Signals ────────────────────────────────────
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.40
    last_active = signals.get("last_active_date", "2025-01-01")
    try:
        last_date = date.fromisoformat(last_active)
        today = date(2026, 6, 6)
        days_inactive = (today - last_date).days
    except (ValueError, TypeError):
        days_inactive = 365

    if days_inactive <= 7:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.90
    elif days_inactive <= 60:
        recency = 0.75
    elif days_inactive <= 90:
        recency = 0.55
    elif days_inactive <= 180:
        recency = 0.30
    else:
        recency = 0.10

    openness_recency = open_to_work * 0.50 + recency * 0.50

    resp_rate = signals.get("recruiter_response_rate", 0.0)
    resp_time = signals.get("avg_response_time_hours", 200)
    if resp_time <= 4:
        resp_time_score = 1.0
    elif resp_time <= 24:
        resp_time_score = 0.85
    elif resp_time <= 72:
        resp_time_score = 0.65
    else:
        resp_time_score = 0.40
    responsiveness = resp_rate * 0.65 + resp_time_score * 0.35

    interview_rate = signals.get("interview_completion_rate", 0.5)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate == -1:
        offer_rate = 0.55
    reliability = interview_rate * 0.60 + offer_rate * 0.40

    github = signals.get("github_activity_score", -1)
    if github == -1:
        github_score = 0.45
    else:
        github_score = github / 100.0

    profile_comp = signals.get("profile_completeness_score", 50) / 100.0
    verified_email = 1.0 if signals.get("verified_email", False) else 0.0
    verified_phone = 1.0 if signals.get("verified_phone", False) else 0.0
    linkedin = 1.0 if signals.get("linkedin_connected", False) else 0.0
    verified_bonus = verified_email * 0.4 + verified_phone * 0.3 + linkedin * 0.3
    credibility = profile_comp * 0.60 + verified_bonus * 0.40

    redrob_score = (
        openness_recency * 0.35 +
        responsiveness * 0.25 +
        reliability * 0.20 +
        github_score * 0.10 +
        credibility * 0.10
    )

    features["days_inactive"] = days_inactive
    features["open_to_work"] = signals.get("open_to_work_flag", False)
    features["recency_score"] = round(recency, 4)
    features["responsiveness"] = round(responsiveness, 4)
    features["reliability"] = round(reliability, 4)
    features["github_score"] = round(github_score, 4)
    features["credibility"] = round(credibility, 4)
    features["redrob_signal_score"] = round(redrob_score, 4)

    # ── PILLAR 4: Education ─────────────────────────────────────────
    if education:
        tier_score = max(
            TIER_SCORES.get(e.get("tier", "unknown"), 0.50)
            for e in education
        )
        field_rel = max(
            _get_field_relevance(e.get("field_of_study", ""))
            for e in education
        )
    else:
        tier_score = 0.50
        field_rel = 0.15

    cert_bonus = 0.0
    for cert in certs:
        cn = cert.get("name", "").lower()
        matched = False
        for ref_cert, bonus in CERT_BONUSES.items():
            if ref_cert in cn or fuzz.ratio(cn, ref_cert) >= 78:
                cert_bonus += bonus
                matched = True
                break
        if not matched:
            if any(kw in cn for kw in ["machine learning", "deep learning", "nlp", "ai"]):
                cert_bonus += 0.04
    cert_bonus = min(0.15, cert_bonus)

    education_score = min(1.0, tier_score * 0.60 + field_rel * 0.40 + cert_bonus)

    features["tier_score"] = round(tier_score, 4)
    features["field_relevance"] = round(field_rel, 4)
    features["cert_bonus"] = round(cert_bonus, 4)
    features["education_score"] = round(education_score, 4)

    # ── PILLAR 5: Logistics ─────────────────────────────────────────
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.80
    elif notice <= 90:
        notice_score = 0.60
    elif notice <= 120:
        notice_score = 0.40
    else:
        notice_score = 0.20

    location_score = _get_location_score(candidate)

    work_mode = signals.get("preferred_work_mode", "remote")
    work_mode_score = WORK_MODE_COMPAT.get(work_mode, 0.50)

    logistics_score = (
        notice_score * 0.40 +
        location_score * 0.35 +
        work_mode_score * 0.25
    )

    features["notice_period_days"] = notice
    features["notice_score"] = round(notice_score, 4)
    features["location_score"] = round(location_score, 4)
    features["work_mode_score"] = round(work_mode_score, 4)
    features["logistics_score"] = round(logistics_score, 4)
    features["location"] = profile.get("location", "")
    features["country"] = profile.get("country", "")
    features["willing_to_relocate"] = signals.get("willing_to_relocate", False)

    # ── Store raw data for reasoning ────────────────────────────────
    features["headline"] = profile.get("headline", "")
    features["summary_snippet"] = profile.get("summary", "")[:200]
    features["num_skills"] = len(skills)
    features["career_companies"] = [c.get("company", "") for c in career]
    features["career_titles"] = [c.get("title", "") for c in career]

    return features


def extract_all_features(candidates: list) -> list:
    """Extract features for all candidates. Returns list of feature dicts."""
    return [extract_features(c) for c in candidates]
