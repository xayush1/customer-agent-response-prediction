"""Accuracy evaluation for the Agent Response Predictor.

Strategy: hold out real customer->agent turns from the dataset, predict the
agent reply, and compare against the ground-truth agent reply using:
  1. Embedding cosine similarity (semantic closeness)
  2. An LLM-as-judge score (1-5) for whether the prediction is an acceptable
     substitute for the real agent reply.

Also runs a multi-turn context check (follow-up that depends on earlier turns).
"""
import json
import random
import statistics

import httpx
import numpy as np

from config import get_config
from embeddings import embed_query
from preprocess import clean_text, load_conversations, normalize_role
import react_agent

cfg = get_config()
random.seed(7)


def cosine(a, b) -> float:
    va, vb = np.array(a), np.array(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def llm_judge(customer, context, predicted, actual) -> dict:
    client = react_agent._client()
    prompt = f"""You are evaluating a customer-support reply predictor.

CONVERSATION CONTEXT:
{context or '(none)'}

CUSTOMER MESSAGE:
{customer}

ACTUAL AGENT REPLY (ground truth):
{actual}

PREDICTED AGENT REPLY:
{predicted}

Rate how good the PREDICTED reply is as a substitute for the ACTUAL reply on a
1-5 scale, considering correctness of intent, appropriateness, and tone.
5 = essentially equivalent / would be fine to send
3 = reasonable but misses some specifics
1 = wrong intent or unhelpful
Return STRICT JSON: {{"score": <int>, "reason": "<short>"}}"""
    r = client.chat.completions.create(
        model=cfg.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)


def build_holdout(n_simple=4, n_complex=4):
    """Pick held-out customer->agent turns. 'Simple' = early/short turns,
    'complex' = deep turns in long threads."""
    convos = load_conversations(cfg.DATASET_PATH)
    simple, complex_ = [], []
    for c in convos:
        msgs = c["messages"]
        for i in range(len(msgs) - 1):
            if normalize_role(msgs[i]["role"]) == "customer" and normalize_role(msgs[i + 1]["role"]) == "agent":
                cust = clean_text(msgs[i]["text"])
                reply = clean_text(msgs[i + 1]["text"])
                if not cust or not reply or len(reply) < 40:
                    continue
                history = [
                    {"role": normalize_role(m["role"]), "text": clean_text(m["text"])}
                    for m in msgs[:i]
                ]
                item = {
                    "thread_id": c["threadId"],
                    "subject": c["subject"],
                    "history": history,
                    "customer": cust,
                    "actual": reply,
                    "depth": i,
                }
                if i <= 1 and len(c["messages"]) <= 4:
                    simple.append(item)
                elif i >= 4:
                    complex_.append(item)
    random.shuffle(simple)
    random.shuffle(complex_)
    return simple[:n_simple], complex_[:n_complex]


def run_case(item, label):
    payload = {"conversation_history": item["history"], "customer_message": item["customer"]}
    with httpx.Client(timeout=180) as client:
        resp = client.post(f"{cfg.API_BASE_URL}/predict", json=payload).json()
    predicted = resp["predicted_response"]
    ctx = "\n".join(f"{m['role'].upper()}: {m['text'][:300]}" for m in item["history"])
    sim = cosine(embed_query(predicted), embed_query(item["actual"]))
    judge = llm_judge(item["customer"], ctx, predicted, item["actual"])
    print(f"\n{'='*70}\n[{label}] {item['subject'][:60]} (depth={item['depth']})")
    print(f"CUSTOMER: {item['customer'][:200]}")
    print(f"\nACTUAL : {item['actual'][:300]}")
    print(f"\nPREDICT: {predicted[:300]}")
    print(f"\n  cosine_sim={sim:.3f}  judge={judge['score']}/5  ({judge['reason']})")
    print(f"  tool_calls={sum(1 for t in resp['reasoning_trace'] if t.get('type')=='action')}"
          f"  confidence={resp['confidence']}")
    return {"sim": sim, "judge": judge["score"]}


def multiturn_check():
    """Verify the agent uses prior conversation context on a follow-up."""
    print(f"\n{'#'*70}\n# MULTI-TURN CONTEXT CHECK\n{'#'*70}")
    history = []
    turns = [
        "Hi, I'd like to upgrade my plan from Pro to Elite for my 50 users.",
        "What would the price be per user per month?",
        "Ok and does that include the unlimited AI features you mentioned?",
    ]
    with httpx.Client(timeout=180) as client:
        for t in turns:
            payload = {"conversation_history": history, "customer_message": t}
            resp = client.post(f"{cfg.API_BASE_URL}/predict", json=payload).json()
            reply = resp["predicted_response"]
            print(f"\nCUSTOMER: {t}")
            print(f"AGENT   : {reply[:280]}")
            history.append({"role": "customer", "text": t})
            history.append({"role": "agent", "text": reply})
    # Judge whether the final reply correctly references context (Elite/AI/50 users)
    judge = llm_judge(
        turns[-1],
        "\n".join(f"{m['role'].upper()}: {m['text'][:200]}" for m in history[:-2]),
        history[-1]["text"],
        "A reply that correctly stays on the topic of the Elite plan upgrade and its AI features.",
    )
    print(f"\n  context-coherence judge={judge['score']}/5 ({judge['reason']})")


def main():
    simple, complex_ = build_holdout()
    print(f"Holdout: {len(simple)} simple, {len(complex_)} complex cases")
    results = {"simple": [], "complex": []}
    for it in simple:
        results["simple"].append(run_case(it, "SIMPLE"))
    for it in complex_:
        results["complex"].append(run_case(it, "COMPLEX"))

    print(f"\n{'='*70}\nSUMMARY")
    for k, rs in results.items():
        if rs:
            print(
                f"  {k:8s}: avg cosine={statistics.mean(r['sim'] for r in rs):.3f}  "
                f"avg judge={statistics.mean(r['judge'] for r in rs):.2f}/5  (n={len(rs)})"
            )
    all_rs = results["simple"] + results["complex"]
    if all_rs:
        print(
            f"  OVERALL : avg cosine={statistics.mean(r['sim'] for r in all_rs):.3f}  "
            f"avg judge={statistics.mean(r['judge'] for r in all_rs):.2f}/5"
        )

    multiturn_check()


if __name__ == "__main__":
    main()
