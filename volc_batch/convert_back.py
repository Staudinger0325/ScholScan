import json
from pathlib import Path
from typing import Dict, List, Optional

SRC_TEXT_JSON      = Path("")
SRC_IMAGE_JSON     = Path("")
BATCH_TEXT_JSONL   = Path("")    
BATCH_IMAGE_JSONL  = Path("")   
OUT_TEXT           = Path("")
OUT_IMAGE          = Path("")

def to_str(x) -> str:
    return "" if x is None else str(x)

def load_json_array(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path} The top level must be an array")
    return data

def resolve_jsonl(p: Path) -> Optional[Path]:
    if p is None:
        return None
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
        return cands[0] if cands else None
    return None

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield ln, json.loads(line)

def ensure_custom_id(prefix: str, rec: Dict, idx: int) -> str:
    rid = to_str(rec.get("id")).strip()
    return f"{prefix}-{(rid if rid else f'{idx:06d}')}"

def build_source_map(source: List[Dict], prefix: str) -> Dict[str, Dict]:
    return {ensure_custom_id(prefix, rec, i): rec for i, rec in enumerate(source)}

def extract_content(obj: Dict) -> str:
    try:
        ch = obj.get("response", {}).get("body", {}).get("choices", [])
        if ch and isinstance(ch, list):
            return to_str(ch[0].get("message", {}).get("content")).strip()
        return ""
    except Exception:
        return ""

def merge_one(source_json: Path, batch_jsonl: Path, out_path: Path, prefix: str):
    source = load_json_array(source_json)
    src_map = build_source_map(source, prefix)

    out_rows, total, missing_in_source, empty_content = [], 0, 0, 0
    for ln, obj in iter_jsonl(batch_jsonl):
        total += 1
        cid = to_str(obj.get("custom_id")).strip()
        src = src_map.get(cid)
        content = extract_content(obj)
        if content == "":
            empty_content += 1

        if src is None:
            missing_in_source += 1
            out_rows.append({
                "question": "",
                "explanation": "",
                "execution_outcome": {"extracted_content": content},
                "_meta": {"custom_id": cid, "note": "source_not_found", "jsonl_line": ln}
            })
        else:
            out_rows.append({
                "question": to_str(src.get("question")),
                "explanation": to_str(src.get("explanation")),
                "execution_outcome": {"extracted_content": content},
                "_meta": {
                    "id": to_str(src.get("id")),
                    "custom_id": cid,
                    "Error Type": to_str(src.get("Error Type")),
                    "Type": to_str(src.get("Type"))
                }
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{prefix}] Total lines: {total}  Unmatched sources: {missing_in_source}  Empty content: {empty_content}  -> {out_path}")


def main():
    # Text
    text_jsonl = resolve_jsonl(BATCH_TEXT_JSONL)
    if text_jsonl:
        print(f"[text] Via JSONL: {text_jsonl}")
        merge_one(SRC_TEXT_JSON, text_jsonl, OUT_TEXT, prefix="text")
    else:
        print("[text] No batch JSONL found, skipped.")

    # Image
    img_jsonl = resolve_jsonl(BATCH_IMAGE_JSONL)
    if img_jsonl:
        print(f"[img] 使用 JSONL: {img_jsonl}")
        merge_one(SRC_IMAGE_JSON, img_jsonl, OUT_IMAGE, prefix="img")
    else:
        print("[img] No batch JSONL found, skipped.")

if __name__ == "__main__":
    main()
