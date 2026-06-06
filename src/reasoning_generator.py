"""
reasoning_generator.py — Generate reasoning strings for ranked candidates.
Template-based, no API calls. References actual candidate data.
"""


def generate_reasoning(rank: int, features: dict) -> str:
    """
    Generate a 1-2 sentence reasoning string for a candidate at a given rank.
    All claims are derived from the features dict (validated data).
    """
    title = features.get("current_title", "Professional")
    yoe = features.get("yoe", 0)
    company = features.get("current_company", "")
    industry = features.get("current_industry", "")
    skill_score = features.get("skill_fit_score", 0)
    exp_score = features.get("experience_score", 0)
    redrob_score = features.get("redrob_signal_score", 0)
    notice = features.get("notice_period_days", 90)
    location = features.get("location", "")
    country = features.get("country", "")
    days_inactive = features.get("days_inactive", 999)
    must_have = features.get("must_have_clusters", 0)
    disqualified = features.get("disqualified", False)
    dq_reason = features.get("disqualify_reason", "")
    github = features.get("github_score", 0)
    domain = features.get("domain_score", 0)
    num_skills = features.get("num_skills", 0)
    career_bonus = features.get("career_bonus", 0)
    is_langchain = features.get("is_langchain_only", False)
    responsiveness = features.get("responsiveness", 0)
    loc_str = f"{location}, {country}".strip(", ")

    # Build strength phrases
    strengths = []
    concerns = []

    if skill_score >= 0.6:
        strengths.append(f"strong skill fit ({must_have}/4 must-have clusters)")
    elif skill_score >= 0.3:
        strengths.append(f"moderate skill alignment ({must_have}/4 must-have clusters)")

    if 6 <= yoe <= 8:
        strengths.append(f"ideal {yoe:.1f} YoE")
    elif 5 <= yoe <= 9:
        strengths.append(f"{yoe:.1f} YoE in target range")
    elif yoe > 0:
        strengths.append(f"{yoe:.1f} YoE")

    if domain >= 0.7:
        strengths.append("product-company background")
    if career_bonus >= 0.10:
        strengths.append("career history shows production ML work")
    if github >= 0.5:
        strengths.append("active GitHub profile")
    if responsiveness >= 0.6:
        strengths.append("high recruiter responsiveness")

    # Build concern phrases
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if days_inactive > 90:
        concerns.append(f"inactive for {days_inactive} days on platform")
    if is_langchain:
        concerns.append("AI skills limited to framework wrappers")
    if must_have < 2:
        concerns.append(f"only {must_have}/4 must-have skill clusters")
    if domain < 0.4:
        concerns.append("primarily consulting/services background")
    if skill_score < 0.2:
        concerns.append("limited relevant technical skills")
    if country.lower() not in ("india", ""):
        concerns.append(f"based in {country}")

    if disqualified:
        reason_map = {
            "entire_career_at_consulting_firms": "entire career at consulting firms with no product company experience",
            "zero_must_have_skill_clusters": "no signals for any must-have skill cluster",
            "insufficient_experience": f"only {yoe:.1f} years of experience (minimum 3 required)",
            "non_technical_with_no_tech_history": "non-technical title with no technical career history",
        }
        dq_text = reason_map.get(dq_reason, dq_reason)
        return f"{title} with {yoe:.1f} yrs at {company}; disqualified: {dq_text}."

    # Format by rank tier
    # Ensure some variety and specific candidate details in the output
    comp_str = f" at {company}" if company else ""
    ind_str = f" ({industry})" if industry else ""
    
    strength_desc = "; ".join(strengths) if strengths else f"experience in {industry or 'relevant tech domain'}"
    concern_desc = f"with concerns on {'; '.join(concerns)}" if concerns else "with no major concerns"

    if rank <= 10:
        return f"{title}{comp_str}{ind_str} with {yoe:.1f} YoE; possesses {strength_desc}. Strongly matches Senior AI Engineer requirements."
    elif rank <= 30:
        return f"{title} with {yoe:.1f} YoE{comp_str}; shows {strength_desc}. Note: {'; '.join(concerns) if concerns else 'shows consistent profile alignment'}."
    elif rank <= 60:
        primary = strengths[0] if strengths else f"{yoe:.1f} YoE as {title}"
        secondary = f" and {strengths[1]}" if len(strengths) > 1 else ""
        gap = f", though restricted by {concerns[0]}" if concerns else ", showing reasonable baseline fit"
        return f"Exhibits {primary}{secondary}{gap}. Suitable mid-tier profile match."
    else:
        # Rank 61-100
        primary_sig = strengths[0] if strengths else f"background as {title}"
        primary_gap = f"but held back by {concerns[0]}" if concerns else "but aligns weakly with specific must-have skills"
        return f"Has {yoe:.1f} YoE with {primary_sig}, {primary_gap}. Positioned as a backup candidate."


def generate_all_reasonings(ranked_results: list) -> dict:
    """
    Generate reasoning for all ranked candidates.
    
    Args:
        ranked_results: list of (candidate_id, rank, score, features)
    
    Returns:
        dict mapping candidate_id -> reasoning string
    """
    reasonings = {}
    for cid, rank, score, features in ranked_results:
        reasonings[cid] = generate_reasoning(rank, features)
    return reasonings
