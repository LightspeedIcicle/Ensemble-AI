# core/knowledge.py
# Stage 7 — Knowledge persistence.
# Validated facts are logged to disk and, with user confirmation, promoted into
# a growing "master prompt" that primes the local LLM on future queries. A local
# dedup check keeps the store from accumulating the same fact twice.

import json
from datetime import datetime
from pathlib import Path

from core.clients import ollama_client, LOCAL_MODEL, TEMP_DETERMINISTIC
from core.helpers import parse_json

# ── Store locations ───────────────────────────────────────────────────────────
KNOWLEDGE_DIR = Path("knowledge")
KNOWLEDGE_LOG = KNOWLEDGE_DIR / "log.json"
MASTER_PROMPT_FILE = KNOWLEDGE_DIR / "master_prompt.txt"


# ── Store setup / IO ──────────────────────────────────────────────────────────

def init_knowledge_store():
    """Create the knowledge directory and files if they don't exist."""
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    if not KNOWLEDGE_LOG.exists():
        KNOWLEDGE_LOG.write_text(json.dumps({"entries": []}, indent=2))
    if not MASTER_PROMPT_FILE.exists():
        MASTER_PROMPT_FILE.write_text(
            "You are a helpful assistant with the following accumulated knowledge:\n\n"
        )


def load_knowledge_log():
    return json.loads(KNOWLEDGE_LOG.read_text())


def save_knowledge_log(data):
    KNOWLEDGE_LOG.write_text(json.dumps(data, indent=2))


def load_master_prompt():
    return MASTER_PROMPT_FILE.read_text()


def save_master_prompt(content):
    MASTER_PROMPT_FILE.write_text(content)


# ── Log writes ────────────────────────────────────────────────────────────────

def add_to_knowledge_log(prompt, validated_items, source_topic):
    """Append validated items to the knowledge log. Returns the new entries."""
    data = load_knowledge_log()
    new_entries = []
    for item in validated_items:
        entry = {
            "id": len(data["entries"]) + len(new_entries) + 1,
            "timestamp": datetime.now().isoformat(),
            "source_topic": source_topic,
            "knowledge": item,
            "added_to_master": False,
        }
        new_entries.append(entry)
        data["entries"].append(entry)
    save_knowledge_log(data)
    return new_entries


def check_duplicate(new_item, existing_entries):
    """Ask the local LLM whether this item is already captured in the log."""
    if not existing_entries:
        return False

    existing_knowledge = [e["knowledge"] for e in existing_entries[-20:]]

    result = ollama_client.chat(
        model=LOCAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a deduplication agent. Determine if new information is already captured in existing knowledge.
Return only this JSON with no preamble or markdown:
{"is_duplicate": true or false, "reason": "brief explanation"}""",
            },
            {
                "role": "user",
                "content": f"New item: {new_item}\n\nExisting knowledge:\n{json.dumps(existing_knowledge, indent=2)}",
            },
        ],
        options={"temperature": TEMP_DETERMINISTIC},
    )
    parsed = parse_json(result["message"]["content"])
    if not parsed:
        return False
    return parsed.get("is_duplicate", False)


# ── Master prompt promotion ───────────────────────────────────────────────────

def request_master_prompt_addition(items, source_topic):
    """Ask the user which validated items should be promoted to the master prompt.

    Returns the list to add ([] to log-only), or None to skip entirely.
    """
    print("\n--- KNOWLEDGE LOG UPDATE ---")
    print(f"Source topic: {source_topic}")
    print(f"The following {len(items)} items were validated and are new:")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")

    print("\nOptions:")
    print("  [a] Add all to master prompt")
    print("  [s] Select specific items")
    print("  [l] Log only (don't add to master prompt)")
    print("  [n] Skip")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "a":
        return items
    elif choice == "s":
        selected = []
        for i, item in enumerate(items, 1):
            confirm = input(f"Add item {i}? (y/n): ").strip().lower()
            if confirm == "y":
                selected.append(item)
        return selected
    elif choice == "l":
        return []
    else:
        return None


def add_to_master_prompt(items, source_topic):
    """Append validated knowledge to the master prompt file and mark it in the log."""
    current = load_master_prompt()
    timestamp = datetime.now().strftime("%Y-%m-%d")

    addition = f"\n## {source_topic} ({timestamp})\n"
    for item in items:
        addition += f"- {item}\n"

    save_master_prompt(current + addition)

    data = load_knowledge_log()
    for entry in data["entries"]:
        if entry["knowledge"] in items:
            entry["added_to_master"] = True
    save_knowledge_log(data)

    print(f"[Knowledge] Added {len(items)} items to master prompt")


# ── Orchestration helper ──────────────────────────────────────────────────────

def persist(prompt, validated):
    """Full stage-7 flow: dedup validated facts, then log + optionally promote.

    `validated` is the monitor's list of {"item": ...} dicts. Kept here so the
    pipeline stays a thin orchestrator.
    """
    if not validated:
        return

    validated_knowledge = [v["item"] for v in validated]
    existing = load_knowledge_log()["entries"]

    new_items = [item for item in validated_knowledge if not check_duplicate(item, existing)]

    if not new_items:
        print("[Knowledge] No new information to log")
        return

    # Topic label comes from the original prompt. It used to come from the
    # compressed one, but the escalation path no longer produces a compressed
    # prompt to borrow — a truncation is just as good for a log heading.
    topic = prompt[:50] + "..." if len(prompt) > 50 else prompt
    selected = request_master_prompt_addition(new_items, topic)

    if selected is None:
        print("[Knowledge] Skipped")
        return

    add_to_knowledge_log(prompt, new_items, topic)
    print(f"[Knowledge] Logged {len(new_items)} items")

    if selected:
        add_to_master_prompt(selected, topic)
