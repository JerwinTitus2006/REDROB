"""
tests/test_format.py — Validate submission.csv format.
"""
import csv
import os
import json
import pytest

SUBMISSION_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "submission.csv")
CANDIDATES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "candidates.jsonl")


def _load_submission():
    rows = []
    with open(SUBMISSION_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_candidate_ids():
    ids = set()
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["candidate_id"])
    return ids


@pytest.fixture
def rows():
    return _load_submission()


def test_exactly_100_rows(rows):
    assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"


def test_ranks_1_to_100_each_once(rows):
    ranks = sorted(int(r["rank"]) for r in rows)
    assert ranks == list(range(1, 101))


def test_no_duplicate_candidate_ids(rows):
    cids = [r["candidate_id"] for r in rows]
    assert len(set(cids)) == len(cids), "Duplicate candidate_id found"


def test_all_ids_exist_in_jsonl(rows):
    valid = _load_candidate_ids()
    for r in rows:
        assert r["candidate_id"] in valid, f"{r['candidate_id']} not in candidates.jsonl"


def test_scores_monotonically_non_increasing(rows):
    scores = [float(r["score"]) for r in rows]
    for i in range(1, len(scores)):
        assert scores[i] <= scores[i-1], \
            f"Score at rank {i+1} ({scores[i]}) > rank {i} ({scores[i-1]})"


def test_no_negative_scores(rows):
    for r in rows:
        assert float(r["score"]) >= 0, f"Negative score at rank {r['rank']}"


def test_scores_between_0_and_1(rows):
    for r in rows:
        s = float(r["score"])
        assert 0 <= s <= 1, f"Score {s} at rank {r['rank']} outside [0, 1]"


def test_encoding_is_utf8():
    with open(SUBMISSION_PATH, "rb") as f:
        raw = f.read()
    # Should not start with BOM
    assert not raw.startswith(b'\xef\xbb\xbf'), "File has UTF-8 BOM"
    # Should be valid UTF-8
    raw.decode("utf-8")


def test_tie_breaking_by_candidate_id(rows):
    """When scores are equal, candidate_id should be ascending."""
    for i in range(1, len(rows)):
        if float(rows[i]["score"]) == float(rows[i-1]["score"]):
            assert rows[i]["candidate_id"] >= rows[i-1]["candidate_id"], \
                f"Tie at ranks {rows[i-1]['rank']}-{rows[i]['rank']} not broken by candidate_id"
