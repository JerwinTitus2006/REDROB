"""
tests/test_honeypot.py — Tests for honeypot detection.
"""
import csv
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.honeypot_detector import honeypot_multiplier

SUBMISSION_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "submission.csv")
CANDIDATES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "candidates.jsonl")


def _load_candidates_map():
    cmap = {}
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                cmap[c["candidate_id"]] = c
    return cmap


def _load_top_k(k=10):
    rows = []
    with open(SUBMISSION_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["rank"]) <= k:
                rows.append(row)
    return rows


@pytest.mark.skipif(
    not os.path.exists(SUBMISSION_PATH),
    reason="submission.csv not yet generated"
)
def test_no_honeypot_in_top_10():
    cmap = _load_candidates_map()
    top10 = _load_top_k(10)
    for row in top10:
        cid = row["candidate_id"]
        if cid in cmap:
            mult = honeypot_multiplier(cmap[cid])
            assert mult >= 0.90, \
                f"Honeypot detected in top 10: {cid} (multiplier={mult})"


@pytest.mark.skipif(
    not os.path.exists(SUBMISSION_PATH),
    reason="submission.csv not yet generated"
)
def test_honeypot_rate_below_10_percent_in_top_100():
    cmap = _load_candidates_map()
    rows = []
    with open(SUBMISSION_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    honeypot_count = 0
    for row in rows:
        cid = row["candidate_id"]
        if cid in cmap:
            mult = honeypot_multiplier(cmap[cid])
            if mult < 1.0:
                honeypot_count += 1
    
    assert honeypot_count <= 10, \
        f"Too many honeypots in top 100: {honeypot_count} (max 10)"
