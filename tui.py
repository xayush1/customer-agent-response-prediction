"""Rich terminal UI for the Agent Response Predictor.

Multi-turn: you type as the CUSTOMER, the system predicts the AGENT reply,
both are appended to the running conversation, and the full context is sent
on every subsequent turn. You can keep asking follow-ups.

Commands:
  /load <thread_id>  - preload a conversation from the dataset (up to a customer turn)
  /threads           - list some thread_ids from the dataset
  /trace             - toggle showing the ReAct reasoning trace
  /reset             - clear the conversation
  /quit              - exit
"""
import json

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from config import get_config
from preprocess import clean_text, load_conversations, normalize_role

console = Console()
cfg = get_config()

HISTORY: list[dict] = []  # [{"role": "customer"|"agent", "text": str}]
SHOW_TRACE = False


def render_conversation():
    console.print(Rule("[bold]Conversation[/bold]"))
    if not HISTORY:
        console.print("[dim](empty — type a customer message to begin)[/dim]")
        return
    for m in HISTORY:
        if m["role"] == "customer":
            console.print(
                Panel(m["text"], title="[bold cyan]Customer[/bold cyan]",
                      border_style="cyan", title_align="left")
            )
        else:
            console.print(
                Panel(Markdown(m["text"]), title="[bold green]Agent (predicted)[/bold green]",
                      border_style="green", title_align="left")
            )


def render_trace(trace: list[dict], sources: list[dict], confidence: str):
    if not SHOW_TRACE:
        # Compact summary line only.
        n_search = sum(1 for t in trace if t.get("type") == "action")
        console.print(
            f"[dim]confidence={confidence} | reasoning steps={len(trace)} | "
            f"tool calls={n_search} | sources={len(sources)} "
            f"(use /trace to expand)[/dim]"
        )
        return

    console.print(Rule("[bold yellow]ReAct Reasoning Trace[/bold yellow]"))
    for t in trace:
        kind = t.get("type")
        if kind == "thought":
            if t.get("content"):
                console.print(f"[yellow]Thought:[/yellow] {t['content']}")
        elif kind == "action":
            console.print(
                f"[magenta]Action:[/magenta] {t['tool']}("
                f"{json.dumps(t.get('args', {}), ensure_ascii=False)})"
            )
        elif kind == "observation":
            preview = t.get("content", "")
            if len(preview) > 600:
                preview = preview[:600] + "..."
            console.print(f"[blue]Observation:[/blue] [dim]{preview}[/dim]")
        elif kind == "finish":
            console.print("[green]Finish[/green] -> producing final reply")
    if sources:
        tbl = Table(title="Sources used", show_lines=False)
        tbl.add_column("Score", justify="right")
        tbl.add_column("Thread")
        tbl.add_column("Subject")
        seen = set()
        for s in sources:
            key = s.get("thread_id")
            if key in seen:
                continue
            seen.add(key)
            tbl.add_row(str(s.get("score")), str(s.get("thread_id")), str(s.get("subject")))
        console.print(tbl)
    console.print(f"[dim]Confidence: {confidence}[/dim]")


def call_predict(customer_message: str) -> dict:
    payload = {"conversation_history": HISTORY, "customer_message": customer_message}
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{cfg.API_BASE_URL}/predict", json=payload)
        r.raise_for_status()
        return r.json()


def cmd_threads():
    convos = load_conversations(cfg.DATASET_PATH)
    tbl = Table(title="Sample threads (multi-turn)")
    tbl.add_column("thread_id")
    tbl.add_column("# msgs", justify="right")
    tbl.add_column("subject")
    shown = 0
    for c in convos:
        if len(c.get("messages", [])) >= 4:
            tbl.add_row(c["threadId"], str(len(c["messages"])), c["subject"][:60])
            shown += 1
        if shown >= 15:
            break
    console.print(tbl)


def cmd_load(thread_id: str):
    global HISTORY
    convos = load_conversations(cfg.DATASET_PATH)
    conv = next((c for c in convos if c["threadId"] == thread_id), None)
    if not conv:
        console.print(f"[red]Thread {thread_id} not found.[/red]")
        return
    msgs = conv["messages"]
    # Load up to (but not including) the last customer message, so the user can
    # let the system predict that turn.
    last_customer_idx = None
    for i, m in enumerate(msgs):
        if normalize_role(m["role"]) == "customer":
            last_customer_idx = i
    if last_customer_idx is None:
        console.print("[red]No customer message in this thread.[/red]")
        return

    HISTORY = []
    for m in msgs[:last_customer_idx]:
        HISTORY.append({"role": normalize_role(m["role"]), "text": clean_text(m["text"])})
    pending = clean_text(msgs[last_customer_idx]["text"])
    console.clear()
    console.print(f"[bold]Loaded thread:[/bold] {conv['subject']}")
    render_conversation()
    console.print(
        Panel(pending, title="[bold cyan]Pending customer message (press Enter to predict)[/bold cyan]",
              border_style="cyan")
    )
    if Prompt.ask("Predict agent reply for this message? [Y/n]", default="y").lower().startswith("y"):
        handle_customer_message(pending)


def handle_customer_message(text: str):
    HISTORY.append({"role": "customer", "text": text})
    with console.status("[bold green]Agent thinking (ReAct)...[/bold green]", spinner="dots"):
        try:
            result = call_predict(text)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error calling API: {e}[/red]")
            HISTORY.pop()  # roll back
            return
    reply = result.get("predicted_response", "")
    HISTORY.append({"role": "agent", "text": reply})
    console.print(
        Panel(Markdown(reply), title="[bold green]Agent (predicted)[/bold green]",
              border_style="green", title_align="left")
    )
    render_trace(result.get("reasoning_trace", []), result.get("sources", []),
                 result.get("confidence", "medium"))


def banner():
    console.print(
        Panel(
            Text.from_markup(
                "[bold]Agent Response Predictor[/bold] — ReAct + Milvus + Azure gpt-4.1\n"
                "Type as the [cyan]customer[/cyan]; the system predicts the [green]agent[/green] reply.\n"
                "Conversation context is kept across turns.\n\n"
                "[dim]/load <thread_id>  /threads  /trace  /reset  /quit[/dim]"
            ),
            border_style="white",
        )
    )


def main():
    banner()
    # Health check
    try:
        with httpx.Client(timeout=10) as client:
            h = client.get(f"{cfg.API_BASE_URL}/health").json()
        console.print(f"[dim]Backend OK — {h.get('milvus')}[/dim]\n")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Backend not reachable at {cfg.API_BASE_URL}: {e}[/red]")
        console.print("[yellow]Start it: uvicorn api:app --port 8888[/yellow]")
        return

    global SHOW_TRACE
    while True:
        try:
            user_in = Prompt.ask("\n[bold cyan]Customer[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break
        s = user_in.strip()
        if not s:
            continue
        if s in ("/quit", "/exit", "/q"):
            break
        if s == "/reset":
            HISTORY.clear()
            console.clear()
            banner()
            continue
        if s == "/trace":
            SHOW_TRACE = not SHOW_TRACE
            console.print(f"[dim]Trace display: {'ON' if SHOW_TRACE else 'OFF'}[/dim]")
            continue
        if s == "/threads":
            cmd_threads()
            continue
        if s.startswith("/load"):
            parts = s.split(maxsplit=1)
            if len(parts) == 2:
                cmd_load(parts[1].strip())
            else:
                console.print("[yellow]Usage: /load <thread_id>[/yellow]")
            continue
        handle_customer_message(s)

    console.print("\n[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    main()
