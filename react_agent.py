"""ReAct agent that predicts the next agent reply.

The agent is given the current conversation + the latest customer message.
It uses tools to retrieve similar past interactions (and optionally drill into
a full thread), reasons over up to REACT_MAX_DEPTH rounds, then produces the
predicted agent response via the `finish` tool.
"""
import json
from functools import lru_cache

from openai import AzureOpenAI

from config import get_config
import retriever

# --- Tool schemas (OpenAI function-calling format) -------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_conversations",
            "description": (
                "Search the knowledge base of past customer-support interactions "
                "for turn-pairs whose customer message is semantically similar to "
                "a query. Returns the past customer message, the agent's actual "
                "reply, and metadata. Use this to ground your predicted response "
                "in how agents handled similar situations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The customer's question/issue to find similar cases for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar cases to retrieve (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_thread",
            "description": (
                "Fetch a small set of the most RELATED agent-customer turn-pairs "
                "from a specific thread_id (not the entire conversation), ordered "
                "by position. Use this when a similar case looks highly relevant "
                "and you want the surrounding back-and-forth of how it was resolved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string", "description": "The thread_id to expand."},
                },
                "required": ["thread_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this when you have enough context to produce the final "
                "predicted agent response. Provide the response text the agent "
                "should send to the customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The predicted agent reply to send to the customer.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Your confidence in this prediction.",
                    },
                },
                "required": ["response"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a customer-support agent response predictor.

You are given an ongoing conversation and the customer's latest message. Your \
job is to predict the reply the REAL human support agent would send next, by \
IMITATING how agents actually handled similar situations in the retrieved data \
— NOT by writing a generic best-practice answer.

## Process (be evidence-driven)
1. ALWAYS call `search_similar_conversations` first with the customer's core \
issue to see how agents actually replied in similar cases.
2. Inspect the retrieved `agent_reply` fields. These are your ground truth for \
the expected behaviour, tone, and concrete procedure.
3. If the top results are not clearly relevant (low similarity, different \
topic, or they don't reveal the actual procedure), DO NOT settle for a generic \
answer — call `search_similar_conversations` again with a refined query, and/or \
`get_full_thread` to see the full resolution. You may search up to 3 times \
before you must finish.
4. Only when you have grounded evidence (or have exhausted 3 searches) call \
`finish`.

## How to write the predicted reply (CRITICAL — avoid generic answers)
- ANCHOR on the closest retrieved `agent_reply`. Adapt its actual wording, \
structure, and procedure to the current customer — do not paraphrase it into a \
generic template.
- Follow the PROCEDURE the agents actually follow in the retrieved chunks. For \
example, if agents escalate to an Account Manager and proactively extend the \
trial, do that — do NOT tell the customer to self-serve upgrade unless the \
retrieved replies actually do so.
- Reuse concrete specifics that appear in the retrieved replies (e.g. "extended \
your trial by 3 days", "informed your Account Manager", exact plan names, \
prices, steps). Prefer the dataset's real procedure over your own assumptions.
- Be specific to the customer's message; reference concrete details (amounts, \
dates, plan names, features) when present.
- If NO retrieved case gives a definitive answer, fall back to the closest \
PROCEDURE seen in the retrieved chunks (e.g. acknowledge + escalate/loop in the \
right team + promise follow-up) rather than inventing a generic self-serve flow.
- Keep it professional and concise. Do NOT invent satisfaction-survey footers \
or marketing links.
- Always end by calling `finish`. Do not write the reply as plain assistant text.

## Confidence
- "high": a retrieved reply closely matches this exact situation.
- "medium": you adapted a related procedure but no exact match.
- "low": no relevant evidence found after searching; you used a generic fallback."""


@lru_cache
def _client() -> AzureOpenAI:
    cfg = get_config()
    return AzureOpenAI(
        azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
        api_key=cfg.AZURE_OPENAI_API_KEY,
        api_version=cfg.AZURE_OPENAI_API_VERSION,
    )


def _execute_tool(name: str, args: dict, query: str | None = None) -> str:
    if name == "search_similar_conversations":
        hits = retriever.search(args["query"], top_k=int(args.get("top_k", 5)))
        return json.dumps(hits, ensure_ascii=False)
    if name == "get_full_thread":
        # Bound the context to the most related pairs (ranked against `query`)
        # rather than dumping the entire thread.
        rows = retriever.get_thread_context(args["thread_id"], query=query)
        return json.dumps(rows, ensure_ascii=False)
    return json.dumps({"error": f"unknown tool {name}"})


def _format_conversation(history: list[dict], customer_message: str) -> str:
    lines = []
    for m in history:
        role = "CUSTOMER" if m.get("role") == "customer" else "AGENT"
        lines.append(f"{role}: {m.get('text', '').strip()}")
    block = "\n\n".join(lines)
    if block:
        block += "\n\n"
    block += f"CUSTOMER (latest): {customer_message.strip()}"
    return block


def predict(history: list[dict], customer_message: str) -> dict:
    """Run the ReAct loop and return prediction + reasoning trace.

    history: list of {"role": "customer"|"agent", "text": str} prior turns.
    customer_message: the new customer message to respond to.
    """
    cfg = get_config()
    client = _client()

    conv_block = _format_conversation(history, customer_message)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here is the current conversation. Predict the agent's next reply.\n\n"
                f"{conv_block}"
            ),
        },
    ]

    trace: list[dict] = []
    sources: list[dict] = []
    max_iters = cfg.REACT_MAX_DEPTH + 1  # reasoning rounds + final

    for step in range(max_iters):
        force_finish = step == max_iters - 1
        resp = client.chat.completions.create(
            model=cfg.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            tools=TOOLS,
            tool_choice=(
                {"type": "function", "function": {"name": "finish"}}
                if force_finish
                else "auto"
            ),
            temperature=0.3,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # Model answered in plain text instead of calling finish; accept it.
            trace.append({"type": "thought", "content": msg.content or ""})
            return {
                "predicted_response": (msg.content or "").strip(),
                "confidence": "medium",
                "reasoning_trace": trace,
                "sources": sources,
            }

        # Record assistant turn (with tool calls) into message history.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "finish":
                trace.append({"type": "finish", "args": args})
                return {
                    "predicted_response": args.get("response", "").strip(),
                    "confidence": args.get("confidence", "medium"),
                    "reasoning_trace": trace,
                    "sources": sources,
                }

            # thought/action
            if msg.content:
                trace.append({"type": "thought", "content": msg.content})
            trace.append({"type": "action", "tool": name, "args": args})
            result = _execute_tool(name, args, query=customer_message)

            # Collect sources from searches
            if name == "search_similar_conversations":
                try:
                    for h in json.loads(result):
                        sources.append(
                            {
                                "thread_id": h.get("thread_id"),
                                "subject": h.get("subject"),
                                "score": h.get("score"),
                            }
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

            obs_preview = result if len(result) < 2500 else result[:2500] + "..."
            trace.append({"type": "observation", "tool": name, "content": obs_preview})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

    # Should not reach here (forced finish guarantees return).
    return {
        "predicted_response": "I'll look into this and get back to you shortly.",
        "confidence": "low",
        "reasoning_trace": trace,
        "sources": sources,
    }


if __name__ == "__main__":
    out = predict([], "Hi, my account was suspended this morning. Please help, it's urgent.")
    print("\n=== PREDICTED RESPONSE ===\n")
    print(out["predicted_response"])
    print(f"\nConfidence: {out['confidence']}")
    print(f"Sources: {len(out['sources'])}, Trace steps: {len(out['reasoning_trace'])}")
