import os
from pathlib import Path
import csv, json, datetime


# Use DATA_DIR if provided (e.g., set to "/data" on Spaces), else local "data/"
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


LEADS_CSV = DATA_DIR / "leads.csv"
FEEDBACK_JSONL = DATA_DIR / "feedback.jsonl"


def record_customer_interest(email: str, name: str, message: str) -> str:
    """Log a potential customer lead to data/leads.csv and echo a short receipt."""
    LEADS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LEADS_CSV.exists()
    with LEADS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "name", "email", "message"])  # header
        writer.writerow([datetime.datetime.utcnow().isoformat(), name, email, message])
    print(f"[LEAD] {name} <{email}> - {message}")
    return "Thanks! We've recorded your interest and will reply soon."


def record_feedback(question: str) -> str:
    """Log an unanswered question to data/feedback.jsonl and echo a short receipt."""
    FEEDBACK_JSONL.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "question": question,
    }
    with FEEDBACK_JSONL.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[FEEDBACK] {question}")
    return "Got it - I've logged your question for the team."
