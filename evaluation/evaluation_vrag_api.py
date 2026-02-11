# /home/lirongjin/ScholEval/evaluation_api.py

import copy
import json
import asyncio
import aiolimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import os
import logging
from typing import Dict, List, Any, Optional
import runpy
import re


class Config:
    # 输入
    dataset_file: str = ""  
    utils_file: str = ""
    errtype_file: str = ""

    # 输出到你指定的目录
    outcome_dir: str = ""
    model_name: str = ""

    # 速率与流程
    rpm: int = 1800
    max_no_improve_round_count: int = 100
    process_count: int = 0  # 0 表示处理全部

    def __init__(self):
        api_key = ""
        base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def setup_logging(output_dir: str):
        log_file = os.path.join(output_dir, "execution.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)


def create_output_dirs(config: Config) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name_part = config.model_name.split("/")[-1]
    output_dir = os.path.join(
        config.outcome_dir,
        os.path.basename(config.dataset_file).split(".")[0],
        model_name_part,
        f"{timestamp}_process_count_{config.process_count or 'all'}",
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rounds_outcome"), exist_ok=True)
    return output_dir


def load_extractor_prompt(utils_path: str) -> str:
    """从 utils.py 加载 extractor_prompt 字符串"""
    ns = runpy.run_path(utils_path)
    if "extractor_prompt" not in ns or not isinstance(ns["extractor_prompt"], str):
        raise ValueError(f"From {utils_path} not found extractor_prompt")
    return ns["extractor_prompt"]


def _normalize_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None


def load_error_type_map(errtype_path: str) -> Dict[str, Any]:
    with open(errtype_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, Any] = {}
    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            rid = _normalize_id(rec.get("id"))
            if rid is None:
                continue
            mapping[rid] = rec.get("Error Type", None)
    else:
        logging.warning(f"Error Type File {errtype_path} is not a list, ignored.")
    logging.info(f"Successfully Loading Error Type Mapping {len(mapping)}.")
    return mapping


def _safe_get_final_answer(item: Dict[str, Any]) -> str:
    final_val = item.get("final", None)
    if isinstance(final_val, list) and len(final_val) >= 3 and isinstance(final_val[2], str):
        return final_val[2]
    return ""


def build_messages(extractor_prompt: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    question = item.get("question", "")
    explanation = item.get("explanation", "")
    ans = _safe_get_final_answer(item)

    user_text = f"Q: {question}\n\nE: {explanation}\n\nA: {ans}"

    return [
        {"role": "system", "content": extractor_prompt},
        {"role": "user", "content": user_text},
    ]


def _strip_code_fences(text: str) -> str:
    fenced = re.match(r"^```(?:json|JSON)?\s*(.*)```$", text.strip(), re.S)
    return fenced.group(1).strip() if fenced else text


def _find_first_json_object(text: str) -> Optional[str]:
    s = text
    start_idx = s.find("{")
    if start_idx == -1:
        return None
    depth = 0
    for i in range(start_idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start_idx : i + 1]
    return None


def parse_model_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        pass
    stripped = _strip_code_fences(content)
    if stripped != content:
        try:
            return json.loads(stripped)
        except Exception:
            pass
    candidate = _find_first_json_object(content)
    if candidate:
        return json.loads(candidate)
    raise ValueError("Cannot parse valid JSOn from outputs.")


async def process_one_item(
    data: Dict[str, Any],
    extractor_prompt: str,
    config: Config,
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    async with limiter:
        response = None
        try:
            messages = build_messages(extractor_prompt, data)
            response = await config.client.chat.completions.create(
                model=config.model_name,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            parsed = parse_model_json(content)

            data["extraction_outcome"] = {
                "is_success": True,
                "raw_response": response.model_dump_json(),
                "raw_text": content,
                "parsed_json": parsed,  
                "error_type": data.get("error_type", None),
            }
        except Exception as e:
            error_message = str(e)
            if response is not None:
                try:
                    error_message += "\t" + response.model_dump_json()
                except Exception:
                    pass
            logging.error(f"Error while parsing data: {error_message}")
            data["extraction_outcome"] = {
                "is_success": False,
                "error": error_message,
                "error_type": data.get("error_type", None),  # >>> added
            }
        return data


def _needs_processing(sample: Dict[str, Any]) -> bool:
    eo = sample.get("extraction_outcome")
    if not isinstance(eo, dict):
        return True
    return not bool(eo.get("is_success", False))


async def process_round(
    config: Config,
    extractor_prompt: str,
    limiter: aiolimiter.AsyncLimiter,
    pending_items_map: Dict[int, Dict[str, Any]],
    final_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tasks = [
        process_one_item(item, extractor_prompt, config, limiter)
        for item in pending_items_map.values()
    ]
    current_round_results = await tqdm_asyncio.gather(*tasks)

    for i, result in enumerate(current_round_results):
        index = list(pending_items_map.keys())[i]
        final_results[index] = result

    return final_results


async def main_async():
    config = Config()
    output_dir = create_output_dirs(config)
    config.setup_logging(output_dir)
    round_outcome_dir = os.path.join(output_dir, "rounds_outcome")

    limiter = aiolimiter.AsyncLimiter(1, 60.0 / config.rpm)

    # 数据加载
    with open(config.dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        if not isinstance(dataset, list):
            raise ValueError("Input file should be a dic list")

    err_map = load_error_type_map(config.errtype_file)
    for rec in dataset:
        rid = _normalize_id(rec.get("id"))
        rec["error_type"] = err_map.get(rid, None)

    if config.process_count and config.process_count > 0:
        dataset = dataset[: config.process_count]

    extractor_prompt = load_extractor_prompt(config.utils_file)

    final_results: List[Dict[str, Any]] = copy.deepcopy(dataset)
    pending_items_map: Dict[int, Dict[str, Any]] = {
        i: dataset[i] for i in range(len(dataset)) if _needs_processing(dataset[i])
    }

    round_count = 0

    while (
        len(pending_items_map) > 0 and round_count < config.max_no_improve_round_count
    ):
        round_count += 1
        logging.info(
            f"Round {round_count}, Remaining: {len(pending_items_map)}"
        )

        final_results = await process_round(
            config, extractor_prompt, limiter, pending_items_map, final_results
        )

        with open(
            os.path.join(round_outcome_dir, f"round_{round_count}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        pending_items_map = {
            i: final_results[i]
            for i in range(len(final_results))
            if _needs_processing(final_results[i])
        }

        if len(pending_items_map) == 0:
            break

    with open(
        os.path.join(output_dir, "final_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main_async())
