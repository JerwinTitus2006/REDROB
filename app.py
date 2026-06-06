import gradio as gr
import subprocess
import tempfile
import os

def rank_candidates(candidates_file):
    if candidates_file is None:
        return None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "submission.csv")
        # Ensure we run in the right directory by running python in rank's directory
        # Let's run rank.py with appropriate arguments
        cmd = [
            "python", "rank.py",
            "--candidates", candidates_file.name,
            "--jd", "./data/job_description.docx",
            "--out", out_path
        ]
        
        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        
        # Copy output to a stable temporary file that Gradio can read
        stable_out = os.path.join(tempfile.gettempdir(), "ranked_submission.csv")
        if os.path.exists(out_path):
            if os.path.exists(stable_out):
                os.remove(stable_out)
            import shutil
            shutil.copy(out_path, stable_out)
            return stable_out
        else:
            raise gr.Error(f"Error executing pipeline: {result.stderr or result.stdout}")

demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.File(label="Upload candidates.jsonl (up to 100,000 candidates)"),
    outputs=gr.File(label="Download submission.csv"),
    title="Redrob Intelligent Candidate Discovery & Ranking System",
    description="Upload a candidates.jsonl file. The pipeline executes offline deterministic scoring (Skills, Experience, Engagement, Education, and Logistics) to rank the top 100 candidates.",
    api_name=False
)

if __name__ == "__main__":
    demo.launch()
