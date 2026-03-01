"""
LLM-based Knowledge Graph Triple Extraction from MS MARCO passages.

Usage:
    # Recommended: set env var once, then run without --api_key
    export GROQ_API_KEY="gsk_xxxxx"          # for Groq
    export OPENAI_API_KEY="sk-xxxxx"         # for OpenAI

    # Step 1: Extract (test on 5 passages)
    python extract_kg_triples.py extract \
        --input entity_linking_results/passage_test_with_id_bm25rank1000.jsonl \
        --output kg_triples_test.jsonl \
        --provider groq \
        --n 5

    # Step 2: View results
    python extract_kg_triples.py view --file kg_triples_test.jsonl --n 5

    # Step 3: Build SQLite KG database
    python extract_kg_triples.py build_db --file kg_triples_test.jsonl --db llm_kg.db

Notes:
    - You may still pass --api_key to override env var (not recommended).
    - If you see 403 / error code 1010, User-Agent header usually fixes it.
"""

import json
import argparse
import sqlite3
import time
import os
import re
import sys


EXTRACTION_PROMPT = """You are an entity and relationship extractor. Read the passage below and extract:

1. All important ENTITIES (people, places, organizations, concepts, events)
2. All RELATIONSHIPS between those entities as triples: (subject, relation, object)

Rules:
- Only extract facts stated or clearly implied in the passage
- Use simple, lowercase entity names
- Use short, clear relation names (e.g., governor_of, located_in, is_a, part_of)
- Do NOT invent relationships not supported by the text

Output EXACTLY in this format:
ENTITIES: entity1, entity2, entity3
TRIPLES:
(subject, relation, object)
(subject, relation, object)

Passage:
{passage_text}"""


def _require_api_key(provider: str, cli_key: str | None) -> str:
    """
    Returns API key from (1) CLI --api_key, else (2) env var based on provider.
    Exits with a helpful error if missing.
    """
    if cli_key and cli_key.strip():
        return cli_key.strip()

    if provider == "groq":
        env = "GROQ_API_KEY"
    elif provider == "openai":
        env = "OPENAI_API_KEY"
    else:
        print(f"ERROR: Unknown provider: {provider}")
        sys.exit(1)

    key = os.getenv(env)
    if not key or not key.strip():
        print("ERROR: No API key found.")
        print(f"Set {env} in your shell, e.g.:")
        print(f'  export {env}="YOUR_KEY_HERE"')
        print("Or pass --api_key (not recommended).")
        sys.exit(1)

    return key.strip()


def call_groq(text, api_key, model="llama-3.1-8b-instant"):
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract entities and relationships from text as structured triples."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(passage_text=text)}
        ],
        "temperature": 0.0,
        "max_tokens": 1000,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Helps avoid Cloudflare 1010 blocks for urllib
            "User-Agent": "kg-ore-extractor/1.0 (python urllib)",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP Error {e.code}: {body[:400]}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def call_openai(text, api_key, model="gpt-4o-mini"):
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract entities and relationships from text as structured triples."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(passage_text=text)}
        ],
        "temperature": 0.0,
        "max_tokens": 1000,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "kg-ore-extractor/1.0 (python urllib)",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP Error {e.code}: {body[:400]}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def parse_response(text):
    """Parse LLM response into entities and triples."""
    entities = []
    triples = []

    if not text:
        return entities, triples

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith("ENTITIES:"):
            ent_text = line.split(":", 1)[1].strip()
            entities = [e.strip().lower() for e in ent_text.split(",") if e.strip()]

        match = re.match(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", line)
        if match:
            triples.append((
                match.group(1).strip().lower(),
                match.group(2).strip().lower(),
                match.group(3).strip().lower(),
            ))

    return entities, triples


# ============================================================
# EXTRACT command
# ============================================================

def cmd_extract(args):
    # Resolve key from env if needed
    args.api_key = _require_api_key(args.provider, args.api_key)

    if args.provider == "groq":
        model = args.model or "llama-3.1-8b-instant"
        call_fn = lambda text: call_groq(text, args.api_key, model)
        rate_limit = 1.5
    elif args.provider == "openai":
        model = args.model or "gpt-4o-mini"
        call_fn = lambda text: call_openai(text, args.api_key, model)
        rate_limit = 0.2
    else:
        print(f"Unknown provider: {args.provider}")
        return

    print(f"Loading passages from {args.input}...")
    passages = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line.strip()))
            if args.n and len(passages) >= args.n:
                break

    total = len(passages)
    print(f"Loaded {total} passages")
    print(f"Provider: {args.provider} / {model}")

    if args.n and args.n <= 20 and os.path.exists(args.output):
        os.remove(args.output)
        print(f"Clean start (removed old {args.output})")

    print(f"\n{'='*70}")
    print("Starting extraction...")
    print(f"{'='*70}")

    processed = 0
    errors = 0
    total_triples = 0

    out_f = open(args.output, "w", encoding="utf-8")

    try:
        for i in range(total):
            passage = passages[i]
            pid = str(passage.get("id"))
            text = passage.get("text", "")

            if not text or len(text.strip()) < 10:
                print(f"[{i+1}/{total}] Passage {pid}: SKIP (empty)")
                continue

            print(f"\n[{i+1}/{total}] Passage {pid} ({len(text)} chars)")
            print(f"  Text: {text[:120]}...")

            response = call_fn(text)

            if response is None:
                errors += 1
                print("  ✗ API error")
                time.sleep(3)
                continue

            entities, triples = parse_response(response)

            result = {
                "id": pid,
                "text": text,
                "llm_entities": entities,
                "llm_triples": [list(t) for t in triples],
                "original_entities": list(set(
                    e.lower().strip() for e in passage.get("entity_name", [])
                )),
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            processed += 1
            total_triples += len(triples)

            print(f"  ✓ {len(entities)} entities, {len(triples)} triples")
            print(f"  Original EL:  {result['original_entities']}")
            print(f"  LLM entities: {entities}")
            for s, r, o in triples:
                print(f"    ({s}, {r}, {o})")

            time.sleep(rate_limit)

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    finally:
        out_f.close()

    print(f"\n{'='*70}")
    print(f"DONE: {processed} processed, {total_triples} triples, {errors} errors")
    print(f"Output: {args.output}")


# ============================================================
# VIEW command
# ============================================================

def cmd_view(args):
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    count = 0
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            if count >= args.n:
                break

            item = json.loads(line.strip())

            print(f"\n{'█'*70}")
            print(f"  PASSAGE {item['id']}")
            print(f"{'█'*70}")
            print(f"\n  Text: {item.get('text', '')[:300]}")

            orig = item.get("original_entities", [])
            llm_ents = item.get("llm_entities", [])
            triples = item.get("llm_triples", [])

            print(f"\n  ORIGINAL Entity Linking ({len(orig)}):")
            for e in orig:
                print(f"    - {e}")

            print(f"\n  LLM Entities ({len(llm_ents)}):")
            for e in llm_ents:
                print(f"    - {e}")

            print(f"\n  LLM Triples ({len(triples)}):")
            for t in triples:
                if len(t) == 3:
                    print(f"    ({t[0]}, {t[1]}, {t[2]})")

            orig_set = set(orig)
            llm_set = set(llm_ents)
            only_orig = orig_set - llm_set
            only_llm = llm_set - orig_set

            print("\n  COMPARISON:")
            print(f"    Shared:        {orig_set & llm_set or 'none'}")
            print(f"    Only original: {only_orig or 'none'}")
            print(f"    Only LLM:      {only_llm or 'none'}")

            count += 1

    print(f"\n{'='*70}")
    print(f"Showed {count} passages")


# ============================================================
# BUILD_DB command
# ============================================================

def cmd_build_db(args):
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    db_path = args.db
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("CREATE TABLE passage_entities (docno TEXT PRIMARY KEY, entities TEXT)")
    conn.execute("CREATE TABLE kg_edges (subject TEXT, relation TEXT, object TEXT, source_passage TEXT)")

    pc = 0
    tc = 0

    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            pid = str(item["id"])
            entities = item.get("llm_entities", [])
            triples = item.get("llm_triples", [])

            if entities:
                conn.execute(
                    "INSERT OR REPLACE INTO passage_entities VALUES (?, ?)",
                    (pid, json.dumps(entities))
                )
                pc += 1

            for t in triples:
                if len(t) == 3:
                    conn.execute("INSERT INTO kg_edges VALUES (?, ?, ?, ?)", (t[0], t[1], t[2], pid))
                    conn.execute("INSERT INTO kg_edges VALUES (?, ?, ?, ?)", (t[2], f"rev_{t[1]}", t[0], pid))
                    tc += 1

    conn.execute("CREATE INDEX idx_pe ON passage_entities(docno)")
    conn.execute("CREATE INDEX idx_s ON kg_edges(subject)")
    conn.execute("CREATE INDEX idx_o ON kg_edges(object)")
    conn.commit()
    conn.close()

    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"Built: {db_path}")
    print(f"  Passages: {pc:,}")
    print(f"  Triples: {tc:,} ({tc*2:,} with reverse)")
    print(f"  Size: {size_mb:.1f} MB")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM KG Triple Extraction")
    sub = parser.add_subparsers(dest="command")

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--input", required=True)
    p_ext.add_argument("--output", required=True)
    p_ext.add_argument("--provider", required=True, choices=["groq", "openai"])
    p_ext.add_argument("--api_key", default=None)  # optional; env var preferred
    p_ext.add_argument("--model", default=None)
    p_ext.add_argument("--n", type=int, default=None)

    p_view = sub.add_parser("view")
    p_view.add_argument("--file", required=True)
    p_view.add_argument("--n", type=int, default=5)

    p_db = sub.add_parser("build_db")
    p_db.add_argument("--file", required=True)
    p_db.add_argument("--db", default="llm_kg.db")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "view":
        cmd_view(args)
    elif args.command == "build_db":
        cmd_build_db(args)
    else:
        parser.print_help()