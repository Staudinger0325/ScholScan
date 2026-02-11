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
import base64
import re

per_id_json_dir: str = ""
topk_pages: int = 10 

class Config:
    dataset_file: str = ""
    outcome_dir: str = ""
    model_name: str = ""
    rpm: int = 200 
    max_no_improve_round_count: int = 200
    process_count: int = -1

    # openrouter
    client: AsyncOpenAI = AsyncOpenAI(
        api_key="",
        base_url="https://openrouter.ai/api/v1",
    )
    max_input_images: int = -1

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


def encode_image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error while encoding images {image_path}: {str(e)}")
        return ""


def _pdf_basename(pdf_path: Optional[str]) -> str:
    if not pdf_path:
        return ""
    return os.path.splitext(os.path.basename(pdf_path))[0]


def _is_from_base(image_path: str, base: str) -> bool:
    if not base:
        return False
    fname = os.path.basename(image_path)
    return (f"/{base}/" in image_path) or fname.startswith(f"{base}_")

_page_num_regex = re.compile(r"_(\d+)(?:\D.*)?\.png$", re.IGNORECASE)

def _extract_page_num(image_path: str) -> Optional[int]:
    fname = os.path.basename(image_path)
    m = _page_num_regex.search(fname)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _read_topk_indices_zero_based(sample_id: str, k: int) -> List[int]:
    path = os.path.join(per_id_json_dir, f"{sample_id}.json")
    if not os.path.isfile(path):
        logging.warning(f"[per-id] Missing {path}, no image will be used.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        arr = obj.get("sorted_page_indices_by_similarity", [])
        first_k_zero: List[int] = []
        for x in arr:
            try:
                first_k_zero.append(int(x))
            except Exception:
                continue
            if len(first_k_zero) >= k:
                break
        return first_k_zero
    except Exception as e:
        logging.error(f"[per-id] Error while reading {path}: {e}")
        return []


def _select_images_by_similarity_indices(data: Dict, config: Config) -> List[str]:
    images: List[str] = data.get("images") or []
    source_pdf = data.get("source")
    src_base = _pdf_basename(source_pdf)
    if not src_base:
        return []

    sample_id = str(data.get("id", "")).strip()
    if not sample_id:
        logging.warning("[per-id] Missing id, no image will be used.")
        return []

    zero_idx_topk = _read_topk_indices_zero_based(sample_id, topk_pages)
    if not zero_idx_topk:
        return []

    wanted_pages_1 = sorted([i + 1 for i in zero_idx_topk if i >= 0])

    candidates: List[tuple] = []
    for p in images:
        if _is_from_base(p, src_base):
            pn = _extract_page_num(p)
            if pn is not None and pn in wanted_pages_1:
                candidates.append((pn, p))

    candidates.sort(key=lambda x: x[0])
    selected = [p for _, p in candidates]

    if len(selected) > config.max_input_images:
        selected = selected[: config.max_input_images]
    return selected


async def process_one_item(
    data: Dict, config: Config, limiter: aiolimiter.AsyncLimiter
) -> Dict:
    async with limiter:
        response = None
        try:
            messages = []

            if data.get("system_input"):
                messages.append({"role": "system", "content": data["system_input"]})

            user_content = []
            if data.get("user_input"):
                user_content.append({"type": "text", "text": data["user_input"]})

            images_to_use = _select_images_by_similarity_indices(data, config)
            for image_path in images_to_use:
                b64 = encode_image_to_base64(image_path)
                if b64:
                    user_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    )

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
            msg = str(e)
            if response is not None:
                try:
                    msg += "\t" + response.model_dump_json()
                except Exception:
                    pass
            logging.error(f"Error while parsing data: {msg}")
            data["execution_outcome"] = {"is_success": False, "error": msg}
        return data


async def process_round(
    config: Config,
    limiter: aiolimiter.AsyncLimiter,
    pending_items_map: Dict,
    final_results: Dict,
) -> Dict:
    tasks = [process_one_item(item, config, limiter) for _, item in pending_items_map.items()]
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

    while len(pending_items_map) > 0 and round_count < config.max_no_improve_round_count:
        round_count += 1
        logging.info(f"Round {round_count}, Remaining: {len(pending_items_map)}")

        final_results = await process_round(config, limiter, pending_items_map, final_results)

        with open(os.path.join(round_outcome_dir, f"round_{round_count}.json"), "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        pending_items_map = {
            i: final_results[i]
            for i in range(len(final_results))
            if not final_results[i]["execution_outcome"]["is_success"]
        }

        if len(pending_items_map) == 0:
            break

    with open(os.path.join(output_dir, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main_async())
