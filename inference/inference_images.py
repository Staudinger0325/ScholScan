import copy
import json
import asyncio
import aiolimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import os
import logging
from typing import Dict
import base64


class Config:
    dataset_file: str = ""   # (hidden path)
    outcome_dir: str = ""    # (hidden path)
    model_name: str = ""
    rpm: int = -1
    max_no_improve_round_count: int = -1
    process_count: int = -1

    # OpenRouter
    client: AsyncOpenAI = AsyncOpenAI(
        api_key="",
        base_url="https://openrouter.ai/api/v1",
    )
    # Aliyun
    # client: AsyncOpenAI = AsyncOpenAI(
    #     api_key="",
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    max_input_images: int = -1

    @staticmethod
    def setup_logging(output_dir: str):
        """Initialize logging to both a file and the console."""
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
    """Create timestamped output directories, including a subfolder for per-round results."""
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
    """Encode an image file to a base64 data URL payload (content only)."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Image encoding error {image_path}: {str(e)}")
        return ""


async def process_one_item(
    data: Dict, config: Config, limiter: aiolimiter.AsyncLimiter
) -> Dict:
    """
    Process a single data item. Expected input fields:
    - system_input: str (optional)
    - user_input: str (optional)
    - images: [path1, path2, ...] (optional; strictly keep given order; send as base64)
    """
    async with limiter:
        response = None
        try:
            messages = []

            # Add system message
            if data.get("system_input"):
                messages.append({"role": "system", "content": data["system_input"]})

            # Build user message content: text first, then images (preserve order)
            user_content = []

            if data.get("user_input"):
                user_content.append({"type": "text", "text": data["user_input"]})

            images = (data.get("images") or [])[: config.max_input_images]
            for image_path in images:
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

            # Call OpenAI-compatible Chat Completions API
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
            logging.error(f"Error while processing data: {error_message}")
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
    """Run one round over all pending items."""
    tasks = []
    for index, item in pending_items_map.items():
        tasks.append(process_one_item(item, config, limiter))

    current_round_results = await tqdm_asyncio.gather(*tasks)

    for i, result in enumerate(current_round_results):
        index = list(pending_items_map.keys())[i]
        final_results[index] = result

    return final_results


async def main_async():
    # Basic configuration
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

    # Main loop: each iteration represents one round processing all not-yet-successful items
    while (
        len(pending_items_map) > 0 and round_count < config.max_no_improve_round_count
    ):
        round_count += 1
        logging.info(
            f"Round {round_count}, remaining unprocessed items: {len(pending_items_map)}"
        )

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
