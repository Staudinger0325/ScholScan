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


class Config:
    dataset_file: str = ""
    outcome_dir: str = ""
    model_name: str = ""
    rpm: int = -1  
    max_no_improve_round_count: int = -1 
    process_count: int = -1 

    # openrouter
    client: AsyncOpenAI = AsyncOpenAI(
        api_key="",
        base_url="https://openrouter.ai/api/v1",
    )
    # aliyun
    # client: AsyncOpenAI = AsyncOpenAI(
    #     api_key="",
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    max_input_images: int = -1 # 最大输入图片数量

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


def _normalize_pages(pages) -> List[int]:
    result = []
    if not pages:
        return result
    for x in pages:
        if isinstance(x, int):
            result.append(x)
        elif isinstance(x, str) and x.isdigit():
            result.append(int(x))
    return result


def _select_images_by_rules(data: Dict, config: Config) -> List[str]:
    images: List[str] = data.get("images") or []
    type_value = data.get("Type") 
    source_pdf = data.get("source")
    evidence_pdf = data.get("evidence_file")
    rag_pages = set(_normalize_pages(data.get("rag_pages")))

    src_base = _pdf_basename(source_pdf)
    ev_base = _pdf_basename(evidence_pdf)

    selected: List[str] = []

    if type_value == "Inter-Generate":
        if ev_base:
            for p in images:
                if _is_from_base(p, ev_base):
                    selected.append(p)

        if src_base and rag_pages:
            for p in images:
                if _is_from_base(p, src_base):
                    pn = _extract_page_num(p)
                    if pn is not None and pn in rag_pages:
                        selected.append(p)
    else:
        if src_base and rag_pages:
            for p in images:
                if _is_from_base(p, src_base):
                    pn = _extract_page_num(p)
                    if pn is not None and pn in rag_pages:
                        selected.append(p)

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

            images_to_use = _select_images_by_rules(data, config)

            for image_path in images_to_use:
                base64_image = encode_image_to_base64(image_path)
                if base64_image:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        }
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
            error_message = str(e)
            if response is not None:
                try:
                    error_message += "\t" + response.model_dump_json()
                except Exception:
                    pass
            logging.error(f"Error while parsing data: {error_message}")
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
    # 基础配置
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
