FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-compute artifacts
RUN python precompute.py \
    --candidates ./data/candidates.jsonl \
    --jd ./data/job_description.docx

# Default command: run the ranking pipeline
CMD ["python", "rank.py", \
     "--candidates", "./data/candidates.jsonl", \
     "--jd", "./data/job_description.docx", \
     "--out", "./outputs/submission.csv"]
