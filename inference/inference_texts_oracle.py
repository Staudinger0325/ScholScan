import copy
import json
import asyncio
import aiolimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import os
import logging
from typing import Dict, List, Optional


class Config:
    dataset_file: str = ""
    outcome_dir: str = ""
    model_name: str = ""

    rpm: int = -1
    max_no_improve_round_count: int = -1
    process_count: int = -1

    # aliyun
    client: AsyncOpenAI = AsyncOpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # openrouter
    #client: AsyncOpenAI = AsyncOpenAI(
    #    api_key="",
    #    base_url="https://openrouter.ai/api/v1",
    #)

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
        config.dataset_file.split("/")[-1].split(".")[0],
        model_name_part,
        f"{timestamp}_process_count_{config.process_count}",
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rounds_outcome"), exist_ok=True)
    return output_dir

_TEXT_BASE_DIR = ""

def _json_path_from_pdf(pdf_path: Optional[str]) -> Optional[str]:
    if not pdf_path:
        return None
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(_TEXT_BASE_DIR, f"{base}.json")

def _load_page_texts(json_path: Optional[str]) -> List[str]:
    if not json_path:
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [x if isinstance(x, str) else str(x) for x in data]
    except Exception as e:
        logging.error(f"Reading Text Input JSON Error: {json_path}: {e}")
        return []

def _normalize_pages(pages) -> List[int]:
    if not pages:
        return []
    out = []
    for x in pages:
        try:
            v = int(x)
            if v >= 1:
                out.append(v)
        except Exception:
            continue
    return out

def _pick_pages_zero_based(page_texts: List[str], one_based_pages: List[int]) -> List[str]:
    if not page_texts or not one_based_pages:
        return []
    res = []
    n = len(page_texts)
    for p in one_based_pages:
        idx = p - 1
        if 0 <= idx < n:
            res.append(page_texts[idx])
    return res

def _build_text_by_rules(data: Dict) -> str:
    type_value = data.get("Type")
    source_pdf = data.get("source")
    evidence_pdf = data.get("evidence_file")
    rag_pages = _normalize_pages(data.get("rag_pages"))

    parts: List[str] = []

    if type_value == "Inter-Generate":
        ev_json = _json_path_from_pdf(evidence_pdf)
        ev_texts = _load_page_texts(ev_json)
        if ev_texts:
            parts.append("\n".join(ev_texts))
        src_json = _json_path_from_pdf(source_pdf)
        src_texts = _load_page_texts(src_json)
        picked = _pick_pages_zero_based(src_texts, rag_pages)
        if picked:
            if parts:
                parts.append("")
            parts.append("\n".join(picked))
    else:
        src_json = _json_path_from_pdf(source_pdf)
        src_texts = _load_page_texts(src_json)
        picked = _pick_pages_zero_based(src_texts, rag_pages)
        if picked:
            parts.append("\n".join(picked))

    return "\n".join(parts).strip()

async def process_one_item(
    data: Dict, config: Config, limiter: aiolimiter.AsyncLimiter
) -> Dict:
    async with limiter:
        response = None
        try:
            messages = []

            # 1) system_input
            if data.get("system_input"):
                messages.append({"role": "system", "content": data["system_input"]})

            # 2) user_input + 3) text
            user_content = []
            if data.get("user_input"):
                user_content.append({"type": "text", "text": data["user_input"]})

            assembled_text = _build_text_by_rules(data)
            if assembled_text:
                user_content.append({"type": "text", "text": assembled_text})

            if user_content:
                messages.append({"role": "user", "content": user_content})

            response = await config.client.chat.completions.create(
                model=config.model_name,
                messages=messages,
            )

            data["execution_outcome"] = {
                "is_success": True,
                "raw_response": response.model_dump_json(),
                "extracted_content": response.choices[0].message.content,
            }
        except Exception as e:
            error_message = str(e)
            if response is not None:
                try:
                    error_message += "\t" + response.model_dump_json()
                except Exception:
                    pass
            logging.error(f"Error While Parsing Data: {error_message}")
            data["execution_outcome"] = {
                "is_success": False,
                "error": error_message,
            }
        return data


async def process_round(
    config: Config,
    limiter: aiolimiter.AsyncLimiter,
    pending_items_map: Dict,
    final_results: Dict,
) -> Dict:
    tasks = []
    for index, item in pending_items_map.items():
        tasks.append(process_one_item(item, config, limiter))

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

    with open(config.dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if config.process_count > 0:
        dataset = dataset[: config.process_count]

    final_results = copy.deepcopy(dataset)
    pending_items_map = {i: dataset[i] for i in range(len(dataset))}
    round_count = 0

    while (
        len(pending_items_map) > 0 and round_count < config.max_no_improve_round_count
    ):
        round_count += 1
        logging.info(f"Round {round_count}, Remaining: {len(pending_items_map)}")

        final_results = await process_round(
            config, limiter, pending_items_map, final_results
        )

        with open(
            os.path.join(round_outcome_dir, f"round_{round_count}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        pending_items_map = {
            i: final_results[i]
            for i in range(len(final_results))
            if not final_results[i]["execution_outcome"]["is_success"]
        }

        if len(pending_items_map) == 0:
            break

    with open(
        os.path.join(output_dir, "final_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main_async())
