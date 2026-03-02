import sqlite3, json, ast, re

def parse_list_field(x: str):
    if x is None:
        return []
    x = x.strip()
    if not x:
        return []
    # Try JSON list
    try:
        v = json.loads(x)
        if isinstance(v, list):
            return [str(t) for t in v]
    except Exception:
        pass
    # Try python literal list
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return [str(t) for t in v]
    except Exception:
        pass
    # Try comma-separated fallback
    if "," in x:
        return [t.strip() for t in x.split(",") if t.strip()]
    # Last resort: find m.xxx patterns
    return re.findall(r"m\.[A-Za-z0-9_]+|/m/[A-Za-z0-9_]+", x)

def to_slash_mid(mid: str) -> str:
    mid = mid.strip()
    if mid.startswith("m."):
        return "/m/" + mid[2:]
    return mid

conn = sqlite3.connect("../passage_entities.db")
cur = conn.cursor()

cur.execute("SELECT mids FROM passage_entities WHERE mids IS NOT NULL LIMIT 5000;")
mids_set = set()

for (mids_txt,) in cur.fetchall():
    for mid in parse_list_field(mids_txt):
        mids_set.add(to_slash_mid(mid))

mids_list = sorted(list(mids_set))
print("Unique mids extracted:", len(mids_list))
print("Sample 30 mids:", mids_list[:30])