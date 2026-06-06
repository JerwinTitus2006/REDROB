---
title: Redrob Ranker
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: false
---

# Redrob Intelligent Candidate Discovery & Ranking System

## Problem Statement

Given 100,000 candidate profiles and a job description for a Senior AI Engineer role, rank the top 100 most suitable candidates using a deterministic scoring pipeline. The system must run within 5 minutes on CPU with ≤16GB RAM and no network access during ranking.

## Architecture

```
candidates.jsonl ──┐
                   ├─→ Feature Extractor ─→ 5-Pillar Scorer ─→ Ranker ─→ submission.csv
job_description.docx┘        │                    │
                             │                    │
                    ┌────────┘                    │
                    ▼                             ▼
              Fuzzy Skill Match          Honeypot Detector
              Career Desc Scan          Disqualifier Gate
              Must-Have Clusters        Composite Score
```

### Scoring Pillars (weights)
| Pillar | Weight | Key Signals |
|--------|--------|-------------|
| Skill Fit | 30% | Must-have clusters, fuzzy matching, career description scan |
| Experience | 25% | YoE band, industry relevance, title trajectory, narrative |
| Redrob Signals | 25% | Platform activity, responsiveness, reliability, GitHub |
| Education | 10% | Institution tier, field relevance, certifications |
| Logistics | 10% | Notice period, location, work mode compatibility |

## Setup

```bash
pip install -r requirements.txt
```

## Pre-computation (run once, no time limit)

```bash
python precompute.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx
```

## Ranking (≤5 min, CPU, no network)

```bash
python rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out ./outputs/submission.csv
```

## Tests

```bash
pytest tests/ -v
```

## Design Decisions

1. **Pillar weights**: Skill fit (30%) is highest because the JD is highly technical. Redrob signals (25%) match experience weight because availability is critical for hiring.
2. **No LLM during ranking**: All features are pre-extracted deterministically. Reasoning is template-based using actual candidate data — no hallucination risk.
3. **Fuzzy matching**: Uses rapidfuzz for skill name matching to handle variations (e.g., "Sentence Transformers" vs "sentence-transformers").
4. **Career description scanning**: Catches candidates who have relevant experience but don't list specific keywords in their skills array.
5. **Honeypot detection**: Multi-signal approach (YoE gaps, assessment mismatches, impossible timelines) rather than special-casing.

## Compute Environment

- Windows PC, CPU only
- Python 3.11
- 16GB RAM
- No GPU required

## AI Tools Declared

- Claude (architecture design, prompt engineering, code generation)
- GitHub Copilot (code completion)
