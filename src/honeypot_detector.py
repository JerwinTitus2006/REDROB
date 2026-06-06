"""
honeypot_detector.py — Flag impossible/suspicious candidate profiles.
"""


def honeypot_multiplier(candidate: dict) -> float:
    """
    Check for honeypot signals. Returns 0.50 if 2+ flags, else 1.0.
    """
    flags = 0
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})

    # 1. YoE much greater than sum of career history
    career_years = sum(c.get("duration_months", 0) for c in career) / 12.0
    claimed_yoe = profile.get("years_of_experience", 0)
    if claimed_yoe - career_years > 4:
        flags += 1

    # 2. Expert proficiency in a skill with 0 months duration
    for skill in skills:
        if (skill.get("proficiency") == "expert" and
                skill.get("duration_months", 99) == 0):
            flags += 1
            break

    # 3. Claimed skills far exceed assessment scores (>40 point gap)
    prof_score_map = {
        "beginner": 25, "intermediate": 50,
        "advanced": 75, "expert": 90
    }
    for skill in skills:
        sname = skill.get("name", "")
        prof = skill.get("proficiency", "beginner")
        # Check if there's an assessment for this skill
        for aname, ascore in assessments.items():
            if sname.lower().strip() == aname.lower().strip():
                claimed = prof_score_map.get(prof, 25)
                if claimed - ascore > 40:
                    flags += 1
                break

    # 4. Profile completeness very low but many skills claimed
    completeness = signals.get("profile_completeness_score", 50)
    if completeness < 30 and len(skills) > 15:
        flags += 1

    # 5. Career titles/descriptions don't match industry at all
    # e.g., "Accountant" title but description talks about ML systems
    for job in career:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        non_tech = ["accountant", "hr manager", "marketing manager",
                     "content writer", "graphic designer", "sales"]
        tech_desc = ["machine learning", "neural network", "deep learning",
                      "model training", "pytorch", "tensorflow"]
        if any(nt in title for nt in non_tech):
            if sum(1 for td in tech_desc if td in desc) >= 3:
                flags += 1
                break

    if flags >= 2:
        return 0.50
    return 1.0
