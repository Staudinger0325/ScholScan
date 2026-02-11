import json
import math
from pathlib import Path
from collections import defaultdict

INFILE = ""

P = 2
LAMBDA = 0.8
BETA = 2
WE_LOC = 1
WE_REA = 1
MU = 0.9
GAMMA = 0.6
Q = 1.5

def safe_div(a, b):
    return a / b if b else 0.0

def compute_loc(m):
    gold = int(m.get('location', {}).get('gold_steps', 0) or 0)
    hit = int(m.get('location', {}).get('hit_steps', 0) or 0)
    extra = int(m.get('location', {}).get('extra_steps', 0) or 0)
    pred = hit + extra
    if gold == 0 and pred == 0:
        dice = 1.0
    elif gold == 0 and pred > 0:
        dice = 0.0
    else:
        dice = safe_div(2 * hit, gold + pred)
    q = safe_div(extra, max(pred, 1))
    L_extra = LAMBDA * (q ** P)
    return max(0.0, dice - L_extra)

def compute_rea(m):
    gold = int(m.get('reasoning', {}).get('gold_steps', 0) or 0)
    reached = int(m.get('reasoning', {}).get('reached_steps', 0) or 0)
    if gold == 0:
        return 1.0
    return (safe_div(reached, gold)) ** BETA

def geo_mean(a, b, wa=1, wb=1):
    if a < 0: a = 0
    if b < 0: b = 0
    if a == 0 or b == 0:
        return 0.0
    return (a**wa * b**wb) ** (1/(wa + wb))

def compute_unrel(n):
    n = int(n or 0)
    if n <= 2:
        return (MU ** n)
    else:
        return (MU ** 2) * math.exp(-GAMMA * ((n-2) ** Q))

def compute_core(m):
    loc = compute_loc(m)
    rea = compute_rea(m)
    return geo_mean(loc, rea, wa=WE_LOC, wb=WE_REA)

def compute_score(m):
    exist = int(m.get("existance", 0) or 0) 
    contains = int(m.get("contains_target_error", 0) or 0)
    if exist == 0 or contains == 0:
        return 0.0

    loc_gold = int(m.get("location", {}).get("gold_steps", 0) or 0)
    rea_gold = int(m.get("reasoning", {}).get("gold_steps", 0) or 0)
    if loc_gold == 0 and rea_gold == 0:
        return 0.0

    core = compute_core(m)
    penalty = compute_unrel(int(m.get("unrelated_errors", 0) or 0))
    return core * penalty

def load_samples(infile):
    p = Path(infile)
    with p.open("r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        try: 
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            pass

    samples = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                continue
    return samples

def _try_parse_json(s):
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:
        return None

def extract_metrics(sample):
    m = None
    eo = sample.get("extraction_outcome", {}) or {}
    # 1) parsed_json
    parsed = eo.get("parsed_json")
    if isinstance(parsed, dict):
        m = parsed

    # 2) raw_text
    if m is None:
        m = _try_parse_json(eo.get("raw_text"))

    # 3) raw_response
    if m is None and isinstance(eo.get("raw_response"), str):
        try:
            r = json.loads(eo["raw_response"])
            content = (r.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            m = _try_parse_json(content)
        except Exception:
            pass

    # 4) execution_outcome.extracted_content
    if m is None:
        m = _try_parse_json((sample.get("execution_outcome", {}) or {}).get("extracted_content"))

    if m is None:
        m = {}

    m = {
        "existance": int(m.get("existance", 0) or 0),
        "contains_target_error": int(m.get("contains_target_error", 0) or 0),
        "location": {
            "gold_steps": int((m.get("location", {}) or {}).get("gold_steps", 0) or 0),
            "hit_steps": int((m.get("location", {}) or {}).get("hit_steps", 0) or 0),
            "extra_steps": int((m.get("location", {}) or {}).get("extra_steps", 0) or 0),
        },
        "reasoning": {
            "gold_steps": int((m.get("reasoning", {}) or {}).get("gold_steps", 0) or 0),
            "reached_steps": int((m.get("reasoning", {}) or {}).get("reached_steps", 0) or 0),
            "missing_steps": int((m.get("reasoning", {}) or {}).get("missing_steps", 0) or 0),
        },
        "unrelated_errors": int(m.get("unrelated_errors", 0) or 0),
    }
    return m

def get_error_type(sample):
    meta = sample.get("_meta", {}) or {}
    t = meta.get("Error Type") or sample.get("Error Type") or sample.get("error_type") or sample.get("type")
    if not t:
        return None
    if isinstance(t, str) and len(t) > 0:
        c = t.strip().upper()[0]
        if c in "ABCDEFGHI":
            return c
    return None

# ============ 主流程 ==============
def main():
    samples = load_samples(INFILE)

    total_scores = []
    type_scores = defaultdict(list)

    for sample in samples:
        m = extract_metrics(sample)
        score = compute_score(m)
        total_scores.append(score)

        et = get_error_type(sample)
        if et is not None:
            type_scores[et].append(score)

    n_all = len(total_scores)
    print(f"Total number of samples: {n_all}")

    if n_all > 0:
        avg_total = sum(total_scores) / n_all
        print(f"Total Average Score: {avg_total:.6f}")
    else:
        print("Total Average Score: N/A")

    for c in "ABCDEFGHI":
        scores = type_scores.get(c, [])
        if scores:
            avg_c = sum(scores) / len(scores)
            print(f"Error Type {c} Avg Score: {avg_c:.6f} ( Number of Samples: {len(scores)})")
        else:
            print(f"Error Type {c} Avg Score: N/A")


if __name__ == "__main__":
    main()
