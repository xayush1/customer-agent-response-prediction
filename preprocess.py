"""Preprocess raw email conversations into clean turn-pairs.

The raw messages contain deeply nested quoted reply chains (often 80%+ of the
text is duplicated history). We strip those so embeddings capture the actual
new content of each message rather than shared boilerplate.
"""
import json
import re
from typing import Any

# --- Quote / signature markers --------------------------------------------

# "On Wed, Feb 11, 2026 at 1:55 PM <support@hiverhq.com> wrote:" and variants
_ON_WROTE = re.compile(
    r"\n?\s*On\s+.{0,200}?\bwrote:\s*", re.IGNORECASE | re.DOTALL
)
# Portuguese: "Em qua., 11 de fev. de 2026 às 17:53, <...> escreveu:"
_EM_ESCREVEU = re.compile(
    r"\n?\s*Em\s+.{0,200}?\bescreveu:\s*", re.IGNORECASE | re.DOTALL
)
# "---------- Forwarded message ---------"
_FORWARDED = re.compile(r"\n?-{2,}\s*Forwarded message\s*-{2,}.*", re.IGNORECASE | re.DOTALL)
# Feedback / satisfaction footer injected by Hiver
_FEEDBACK = re.compile(
    r"How satisfied are you with our service\?.*",
    re.IGNORECASE | re.DOTALL,
)
# Tracking / marketing URLs (very long resources.hiverhq.com links etc.)
_LONG_URL = re.compile(r"<?https?://\S{40,}>?")
_SHORT_URL_ANGLE = re.compile(r"<https?://\S+?>")
# "Book a meeting with me!" promo block
_PROMO = re.compile(r"\*?Book a meeting with me!\*?.*", re.IGNORECASE | re.DOTALL)
# image placeholders
_IMAGE = re.compile(r"\[image:[^\]]*\]")
# repeated whitespace
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def strip_quoted_replies(text: str) -> str:
    """Remove quoted reply chains, keeping only the new message content."""
    if not text:
        return ""
    # Cut everything from the first quote/forward marker onward.
    cut_points = []
    for pat in (_ON_WROTE, _EM_ESCREVEU, _FORWARDED):
        m = pat.search(text)
        if m:
            cut_points.append(m.start())
    if cut_points:
        text = text[: min(cut_points)]

    # Drop lines that are quoted (start with one or more '>').
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    return "\n".join(lines)


def strip_noise(text: str) -> str:
    """Remove feedback footers, promo blocks, tracking URLs, images."""
    text = _FEEDBACK.sub("", text)
    text = _PROMO.sub("", text)
    text = _IMAGE.sub("", text)
    text = _LONG_URL.sub("", text)
    text = _SHORT_URL_ANGLE.sub("", text)
    return text


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single message."""
    text = strip_quoted_replies(text)
    text = strip_noise(text)
    # normalise whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def normalize_role(role: str) -> str:
    """Treat any non-customer participant as 'agent' for prediction purposes."""
    return "customer" if role == "customer" else "agent"


def format_context(messages: list[dict], window: int, max_chars: int = 6000) -> str:
    """Render the last `window` cleaned messages as a readable context block."""
    recent = messages[-window:] if window > 0 else messages
    parts = []
    for m in recent:
        role = normalize_role(m["role"]).upper()
        body = clean_text(m["text"])
        if body:
            parts.append(f"{role}: {body}")
    ctx = "\n\n".join(parts)
    return _truncate(ctx, max_chars)


def extract_turn_pairs(
    conversations: list[dict], window: int = 3, max_varchar: int = 65000
) -> list[dict[str, Any]]:
    """Extract (context, customer_msg, agent_reply, metadata) training pairs.

    A pair is created wherever a customer message is immediately followed by an
    agent message. The context is the cleaned conversation up to (and including)
    the customer message, limited to the last `window` turns.
    """
    pairs: list[dict[str, Any]] = []
    for conv in conversations:
        msgs = conv.get("messages", [])
        thread_id = conv.get("threadId", "")
        subject = conv.get("subject", "")
        for i in range(len(msgs) - 1):
            cur, nxt = msgs[i], msgs[i + 1]
            if normalize_role(cur["role"]) == "customer" and normalize_role(
                nxt["role"]
            ) == "agent":
                customer_msg = clean_text(cur["text"])
                agent_reply = clean_text(nxt["text"])
                if not customer_msg or not agent_reply:
                    continue
                context = format_context(msgs[: i + 1], window)
                pairs.append(
                    {
                        "thread_id": thread_id,
                        "subject": _truncate(subject, 1000),
                        "turn_position": i,
                        "customer_msg": _truncate(customer_msg, max_varchar),
                        "agent_reply": _truncate(agent_reply, max_varchar),
                        "context": _truncate(context, max_varchar),
                    }
                )
    return pairs


def load_conversations(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from config import get_config

    cfg = get_config()
    convos = load_conversations(cfg.DATASET_PATH)
    pairs = extract_turn_pairs(convos, window=cfg.CONTEXT_WINDOW, max_varchar=cfg.MAX_VARCHAR)
    print(f"Conversations: {len(convos)}")
    print(f"Turn-pairs extracted: {len(pairs)}")
    if pairs:
        ex = pairs[0]
        print("\n--- Example pair ---")
        print(f"Subject: {ex['subject']}")
        print(f"Customer: {ex['customer_msg'][:300]}")
        print(f"Agent: {ex['agent_reply'][:300]}")
