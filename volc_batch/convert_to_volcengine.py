import json
from pathlib import Path
import sys
from typing import List, Dict
# convert origin version to one suitable for volcengine
# input
TEXT_JSON = Path("")
IMAGE_JSON = Path("")

# output
OUT_TEXT_JSONL = Path("")
OUT_IMAGE_JSONL = Path("")

# Replace local file path prefix with HTTP prefix (access from volcengine TOS service)
LOCAL_PREFIX = ""
HTTP_PREFIX  = "" 

def load_json_array(path: Path) -> List[Dict]:
    if not path.is_file():
        print(f"input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"read JSON failed: {path}\n{e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"{path} The top level must be an array", file=sys.stderr)
        sys.exit(1)
    return data

def to_str(x) -> str:
    return "" if x is None else str(x)

def ensure_custom_id(prefix: str, rec: Dict, idx: int) -> str:
    rid = to_str(rec.get("id")).strip()
    if rid == "":
        return f"{prefix}-{idx:06d}"
    return f"{prefix}-{rid}"

def replace_prefix(p: str) -> str:
    return p.replace(LOCAL_PREFIX, HTTP_PREFIX)

def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

def build_text_jsonl(records: List[Dict]) -> List[Dict]:
    """
    Text generation task:
    The order of messages must strictly be:
    - system: system_input (if any)
    - user: user_input (if any)
    - user: text (if any)
    """
    out = []
    for i, r in enumerate(records):
        sys_inp = to_str(r.get("system_input")).strip()
        user_inp = to_str(r.get("user_input")).strip()
        text_inp = to_str(r.get("text")).strip()

        messages = []
        if sys_inp:
            messages.append({"role": "system", "content": sys_inp})
        if user_inp:
            messages.append({"role": "user", "content": user_inp})
        if text_inp:
            messages.append({"role": "user", "content": text_inp})

        body = {"messages": messages}
        custom_id = ensure_custom_id("text", r, i)
        out.append({"custom_id": custom_id, "body": body})
    return out

def build_image_jsonl(records: List[Dict]) -> List[Dict]:
    """
    Multi-image understanding task:
    - system: system_input (if any)
    - user: content array:
        First, add a text block with user_input (if any)
        Then, convert each local path in the images list to an HTTP URL,
        and insert them sequentially as image_url
    """
    out = []
    for i, r in enumerate(records):
        sys_inp = to_str(r.get("system_input")).strip()
        user_inp = to_str(r.get("user_input")).strip()
        images = r.get("images") or []

        content = []
        if user_inp:
            content.append({"type": "text", "text": user_inp})

        for p in images:
            if not isinstance(p, str):
                continue
            url = replace_prefix(p)
            content.append({"type": "image_url", "image_url": {"url": url}})

        messages = []
        if sys_inp:
            messages.append({"role": "system", "content": sys_inp})
        messages.append({"role": "user", "content": content})

        body = {"messages": messages}
        custom_id = ensure_custom_id("img", r, i)
        out.append({"custom_id": custom_id, "body": body})
    return out

def main():
    text_records = load_json_array(TEXT_JSON)
    image_records = load_json_array(IMAGE_JSON)

    text_rows = build_text_jsonl(text_records)
    img_rows  = build_image_jsonl(image_records)

    write_jsonl(OUT_TEXT_JSONL, text_rows)
    write_jsonl(OUT_IMAGE_JSONL, img_rows)

    print("Successfully Generated:")
    print(f"- Text Input JSONL：{OUT_TEXT_JSONL} , totally {len(text_rows)} lines")
    print(f"- Image Input JSONL：{OUT_IMAGE_JSONL} , totally {len(img_rows)} lines")

if __name__ == "__main__":
    main()
