#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Set, Any
from statistics import mean

# ========= 路径与参数 =========
DATA_JSON = "/home/lirongjin/ScholEval/final/final_image_rag.json"
PER_ID_DIR = "/home/lirongjin/ScholEval/results/rag_images/VisRAG-Ret"
K_MRR = 5
RECALL_KS = [5, 10]
# rag_pages 是黄金页号，通常是 1-based（图片页号常见如此）
GOLD_ONE_BASED = True
# =============================

def _to_int_list_any(x: Any) -> List[int]:
    """把任意结构尽量转换为 int 列表：拍平一层、过滤无法转 int 的元素。"""
    if x is None:
        return []
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    if isinstance(x, list):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    return []

def _best_rank_list_from_obj(obj: Dict[str, Any]) -> List[int]:
    """
    从 per-id JSON 对象中尽力提取“排名列表”（越前越相似）。
    优先顺序：
      1) sorted_page_indices_by_similarity
      2) 键名包含 'rank'/'sorted' 且同时包含 'sim/similarity/page/index/indices'
      3) 第一个看起来是 int 列表的键
    """
    if "sorted_page_indices_by_similarity" in obj:
        lst = _to_int_list_any(obj["sorted_page_indices_by_similarity"])
        if lst:
            return lst

    candidates = []
    for k, v in obj.items():
        kl = k.lower()
        if any(s in kl for s in ["rank", "sorted"]) and any(s in kl for s in ["sim", "similar", "similarity", "page", "index", "indices"]):
            lst = _to_int_list_any(v)
            if lst:
                candidates.append((k, lst))
    if candidates:
        candidates.sort(key=lambda kv: len(kv[1]), reverse=True)
        return candidates[0][1]

    for _, v in obj.items():
        lst = _to_int_list_any(v)
        if lst:
            return lst

    return []

def _load_rank_for_id(per_id_dir: str, sid: str) -> List[int]:
    """读取 <per_id_dir>/<id>.json 并解析出排名列表。失败返回空。"""
    p = os.path.join(per_id_dir, f"{sid}.json")
    if not os.path.isfile(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        lst = _best_rank_list_from_obj(obj)
        # 去重但保持相对顺序
        seen = set()
        dedup = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                dedup.append(x)
        return dedup
    except Exception:
        return []

def _to_int_list_strict(x) -> List[int]:
    """把 rag_pages 转成正整数列表。"""
    out = []
    if not x:
        return out
    for v in x:
        try:
            iv = int(v)
            if iv >= 1:
                out.append(iv)
        except Exception:
            continue
    return out

def _align_gold(gold: Set[int], zero_based_rank: bool, gold_one_based: bool) -> Set[int]:
    """
    把 gold 映射到与 rank_list 同口径的索引空间：
    - 如果 rank 是 0 基且 gold 是 1 基，则 gold-1
    - 如果 rank 是 1 基且 gold 是 0 基（几乎不会），则 gold+1
    - 否则不变
    """
    if zero_based_rank and gold_one_based:
        return set(g - 1 for g in gold if g >= 1)
    elif (not zero_based_rank) and (not gold_one_based):
        return set(g + 1 for g in gold)
    else:
        return set(gold)

def reciprocal_rank_at_k(rank_list: List[int], gold_aligned: Set[int], k: int) -> float:
    """MRR@k 的单样本：找到前 k 内第一个命中的排名位次 i，则贡献 1/(i+1)；否则 0。"""
    if not rank_list or not gold_aligned:
        return 0.0
    topk = rank_list[:k]
    for i, idx in enumerate(topk):
        if idx in gold_aligned:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(rank_list: List[int], gold_aligned: Set[int], k: int) -> float:
    """Recall@k 的单样本：top-k 命中数 / gold 数量（gold 为空则 0）。"""
    if not gold_aligned:
        return 0.0
    topk_set = set(rank_list[:k])
    hit = len(gold_aligned & topk_set)
    return hit / len(gold_aligned)

def golden_at_k(rank_list: List[int], gold_aligned: Set[int], k: int) -> float:
    """Golden@k（Hit@k / Success@k）：前 k 内是否命中任意 gold，命中=1，否则=0。"""
    if not gold_aligned:
        return 0.0
    topk_set = set(rank_list[:k])
    return 1.0 if len(gold_aligned & topk_set) > 0 else 0.0

def main():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    skip_no_id = 0
    skip_no_gold = 0
    skip_empty_rank = 0

    # 双口径统计：rank=0基 vs rank=1基
    mrr_0b, mrr_1b = [], []
    rec_0b = {k: [] for k in RECALL_KS}
    rec_1b = {k: [] for k in RECALL_KS}
    # 新增：golden@K（命中率）
    golden_0b, golden_1b = [], []

    # 诊断
    rank_len_stats = []
    empty_rank_ids = []

    for item in data:
        total += 1
        sid = str(item.get("id", "")).strip()
        if not sid:
            skip_no_id += 1
            continue

        gold_pages = _to_int_list_strict(item.get("rag_pages", []))
        if not gold_pages:
            skip_no_gold += 1
            continue
        gold_set_1b = set(gold_pages)  # 原始 gold（通常 1 基页号）

        rank_list = _load_rank_for_id(PER_ID_DIR, sid)
        if not rank_list:
            skip_empty_rank += 1
            empty_rank_ids.append(sid)
            continue

        rank_len_stats.append(len(rank_list))

        # 口径 A：rank 当 0-based（把 gold 对齐到 0 基）
        gold_for_0b = _align_gold(gold_set_1b, zero_based_rank=True, gold_one_based=GOLD_ONE_BASED)
        mrr_0b.append(reciprocal_rank_at_k(rank_list, gold_for_0b, K_MRR))
        golden_0b.append(golden_at_k(rank_list, gold_for_0b, K_MRR))
        for k in RECALL_KS:
            rec_0b[k].append(recall_at_k(rank_list, gold_for_0b, k))

        # 口径 B：rank 当 1-based（把 gold 对齐到 1 基）
        gold_for_1b = _align_gold(gold_set_1b, zero_based_rank=False, gold_one_based=GOLD_ONE_BASED)
        mrr_1b.append(reciprocal_rank_at_k(rank_list, gold_for_1b, K_MRR))
        golden_1b.append(golden_at_k(rank_list, gold_for_1b, K_MRR))
        for k in RECALL_KS:
            rec_1b[k].append(recall_at_k(rank_list, gold_for_1b, k))

    eval_n = len(mrr_0b)

    print(f"Total Number: {total}")
    print(f"Evaluated Number: {eval_n}")
    print(f"skipped (no ids found）: {skip_no_id}")
    print(f"skipped (golden answer is blank）: {skip_no_gold}")
    print(f"skipped (input error): {skip_empty_rank}")
    if rank_len_stats:
        print(f"liest length: avg={mean(rank_len_stats):.2f}  min={min(rank_len_stats)}  max={max(rank_len_stats)}")

    if eval_n == 0:
        print("\n No Sample found for evaluating")
        return

    print(f"Golden@{K_MRR}: {mean(golden_0b):.6f}")
    print(f"MRR@{K_MRR}: {mean(mrr_0b):.6f}")
    for k in RECALL_KS:
        print(f"Recall@{k}: {mean(rec_0b[k]):.6f}")

if __name__ == "__main__":
    main()
