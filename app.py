# app.py
import os
import json
from pathlib import Path
from typing import List, Tuple, Union

from dotenv import load_dotenv
from pypdf import PdfReader
import gradio as gr
from openai import OpenAI

# ---- Local tool implementations (imported) ----
# Make sure tools.py sits next to this file and exposes:
#   record_customer_interest(email, name, message) -> str
#   record_feedback(question) -> str
from tools import record_customer_interest, record_feedback

load_dotenv()
client = OpenAI()

ROOT = Path(__file__).parent.resolve()
ME_DIR = ROOT / "me"
ABOUT_PDF = ME_DIR / "about_business.pdf"
SUMMARY_TXT = ME_DIR / "summary.txt"
SYSTEM_PROMPT_FILE = ROOT / "system_prompt.txt"


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _read_pdf_text(pdf_path: Path, max_chars: int = 25000) -> str:
    """Extract text from a PDF, but fail gracefully if the file is missing/corrupt."""
    if not pdf_path.exists():
        return "[about_business.pdf not found]"
    chunks = []
    try:
        reader = PdfReader(str(pdf_path))
        for p in reader.pages:
            chunks.append(p.extract_text() or "")
    except Exception as e:
        chunks.append(f"[PDF read error: {e}]")
    return ("\n".join(chunks))[:max_chars]


def _read_text(path: Path, placeholder: str, max_chars: int = 24000) -> str:
    if not path.exists():
        return placeholder
    try:
        return path.read_text(encoding="utf-8")[:max_chars]
    except Exception as e:
        return f"[read error {path.name}: {e}]"


SYSTEM_PROMPT = _read_text(
    SYSTEM_PROMPT_FILE,
    placeholder=(
        "You are a helpful business assistant. Use the provided business files "
        "to answer questions, stay on-brand, and keep answers concise."
    ),
)

BUSINESS_DOCS = {
    "about_business.pdf": _read_pdf_text(ABOUT_PDF),
    "summary.txt": _read_text(SUMMARY_TXT, "[summary.txt not found]"),
}


# =========================
# Tool wiring for the model
# =========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_customer_interest",
            "description": "Save a potential customer's contact info and notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["email", "name", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_feedback",
            "description": "Log an unanswered question for follow-up by the team.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
]


def _call_local_tool(tool_name: str, args_json: str) -> str:
    """Dispatch tool calls to local Python functions in tools.py, returning a short receipt string."""
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError as e:
        return f"[Tool args parse error] {e}"

    try:
        if tool_name == "record_customer_interest":
            return record_customer_interest(**args)
        if tool_name == "record_feedback":
            return record_feedback(**args)
        return f"[Unknown tool] {tool_name}"
    except Exception as e:
        return f"[Tool error] {e}"


# =========================
# Chat assembly
# =========================
def build_messages(user_msg: str, history: List[Tuple[str, str]]) -> list:
    """
    Compose the message list for the model:
      - A system prompt (persona / behavior)
      - A system 'files' message that contains the business docs (grounding)
      - The prior turns (user/assistant)
      - The latest user message
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Provide business docs every turn for grounding (truncated to stay small).
    files_blob = (
        "Business files (reference only; do not reveal raw text unless asked):\n\n"
        "[about_business.pdf]\n" + BUSINESS_DOCS["about_business.pdf"][:12000] + "\n\n"
        "[summary.txt]\n" + BUSINESS_DOCS["summary.txt"][:8000]
    )
    messages.append({"role": "system", "content": files_blob})

    for u, a in history or []:
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_msg})
    return messages


def chat(user_message: str, history: List[Tuple[str, str]]):
    """Gradio callback: one turn of chat with tool-calling support.

    Handles both legacy tuple history and Gradio messages dict format.
    """

    def _to_tuples_from_messages(msgs: List[dict]) -> Tuple[str, List[Tuple[str, str]]]:
        if not msgs:
            return "", []
        latest = msgs[-1]
        latest_text = latest.get("content", "") if isinstance(latest, dict) else ""
        conv: List[Tuple[str, str]] = []
        pending_user: Union[str, None] = None
        # Pair earlier turns into (user, assistant)
        for m in msgs[:-1]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                conv.append((pending_user, content))
                pending_user = None
        return latest_text, conv

    # Normalize inputs coming from different Gradio versions/configs
    # Case 1: user_message is actually a list of message dicts (type="messages")
    if isinstance(user_message, list) and (not isinstance(user_message, str)):
        user_message, history = _to_tuples_from_messages(user_message)
    # Case 2: history is list of dict messages; convert to tuples
    elif history and isinstance(history, list) and isinstance(history[0], dict):
        _, history = _to_tuples_from_messages(history + [{"role": "user", "content": user_message}])

    messages = build_messages(user_message, history)

    # First pass: let the model decide if it needs tools
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.4,
    )

    msg = resp.choices[0].message

    # If tools are requested, execute them locally and send results back
    if getattr(msg, "tool_calls", None):
        tool_msgs = []
        for tc in msg.tool_calls:
            result = _call_local_tool(tc.function.name, tc.function.arguments)
            tool_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result,
                }
            )

        follow = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages + [{"role": "assistant", "tool_calls": msg.tool_calls}] + tool_msgs,
            temperature=0.4,
        )
        return follow.choices[0].message.content

    # Otherwise return the plain text
    return msg.content


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Business Assistant") as demo:
    gr.Markdown("# Business Assistant")
    gr.Markdown(
        "Ask about services, scheduling, pricing, or leave your contact info.\n\n"
        "_Grounded on your local **me/about_business.pdf** and **me/summary.txt**._"
    )

    # Some Gradio versions don't support retry_btn/undo_btn/clear_btn.
    # Try with them first; fall back if not supported.
    try:
        chat_ui = gr.ChatInterface(
            fn=chat,
            type="messages",
            submit_btn="Send",
            retry_btn="Retry",
            undo_btn="Undo",
            clear_btn="Clear",
        )
    except TypeError:
        chat_ui = gr.ChatInterface(
            fn=chat,
            type="messages",
            submit_btn="Send",
        )

if __name__ == "__main__":
    # Choose binding based on environment: localhost for dev, 0.0.0.0 for Spaces
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    running_in_space = bool(os.getenv("PORT") or os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
    if running_in_space:
        demo.launch(server_name="0.0.0.0", server_port=port)
    else:
        demo.launch(server_port=port)
