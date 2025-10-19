---
title: Business Assistant (Gradio)
colorFrom: indigo
colorTo: blue
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Business Assistant (Gradio)

Agent-powered assistant grounded on two local files: `me/about_business.pdf` and `me/summary.txt`. It can log customer leads and unanswered questions.

## Features

- Grounded answers from your business docs (PDF + summary)
- Tool: `record_customer_interest(email, name, message)` writes to `data/leads.csv`
- Tool: `record_feedback(question)` writes to `data/feedback.jsonl`
- Gradio chat UI; works locally and on Hugging Face Spaces

## Project structure

```
.
|-- app.py
|-- tools.py
|-- requirements.txt
|-- system_prompt.txt
|-- me/
|   |-- about_business.pdf
|   |-- summary.txt
|-- README.md
```

## Quickstart (local)

1) Install dependencies: `pip install -r requirements.txt`
2) Create `.env` with your key:
```
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```
3) Place business files under `me/` as shown above
4) Run: `python app.py` and open http://127.0.0.1:7860

## Deploy on Hugging Face Spaces: https://huggingface.co/spaces/LamaMawalwi/Business_Assistant

1) Create a Space (SDK: Gradio)
2) Push this repo to the Space (or connect GitHub)
3) In Space Settings → Variables and secrets add:
   - `OPENAI_API_KEY = sk-...`
   - (optional) `OPENAI_MODEL = gpt-4o-mini`
   - (optional) `DATA_DIR = /data` if you enable Persistent Storage
4) Open the Space URL after build completes

## Notes

- Do not commit `.env` or private data
- On Spaces, enable Persistent Storage to save `/data` across restarts
