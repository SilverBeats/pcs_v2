#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
from functools import partial
from typing import List

from lwj_tools.llms.client import APIConfig, LLMClientGroup
from lwj_tools.utils.concurrent import MultiThreadingRunner
from lwj_tools.utils.io import FileReader
from lwj_tools.utils.tools import get_file_name_and_ext, get_logger
from transformers import set_seed

from debate import DebateFramework

CONFIG_FILE_PATH = "./configs/role_play.yaml"
LOGGER = get_logger(__name__)


def load_config() -> dict:
    config = FileReader.read(CONFIG_FILE_PATH, return_dict=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    file_name = get_file_name_and_ext(config["event_relation_pair_file_path"])[0]

    config["output_file_path"] = os.path.join(
        config["output_dir"], file_name + "_role_play.jsonl"
    )
    config["history_file_path"] = os.path.join(
        config["output_dir"], file_name + "_history.jsonl"
    )

    return config


def load_data(
    event_file_path: str,
    output_file_path: str,
    role_file_path: str,
    sample_role: bool = False,
) -> List[dict]:
    """
    Args:
        event_file_path: create by `build_event_relation_pairs/main.py`
            each sample's format is {event_idx: int, event: str, relations: List[str]}
        output_file_path:
        role_file_path: jsonl file.
            each line format is {role_idx: int, proportion: float, description: str}
        sample_role: whether to sample the role by proportion
    Returns:
        [{event_idx, event, relations, role_idx, mbti, description, proportion}, ... ]
    """
    existed_idxes = set()
    if os.path.exists(output_file_path):
        for item in FileReader.read(output_file_path, return_iter=True):
            existed_idxes.add((item["event_idx"], item["role_idx"]))

    roles = FileReader.read(role_file_path)
    proportion = None
    if sample_role:
        proportion = [role["proportion"] for role in roles]

    samples = []
    for sample in FileReader.read(event_file_path, return_iter=True):
        selected_role_idxes = [role["role_idx"] for role in roles]
        if sample_role:
            assert proportion is not None
            selected_role_idxes = set(
                random.choices(
                    selected_role_idxes, weights=proportion, k=len(selected_role_idxes)
                )
            )

        for role_idx in selected_role_idxes:
            role = roles[role_idx]
            if (sample["event_idx"], role["role_idx"]) in existed_idxes:
                continue
            samples.append({**sample, **role})
    return samples


def chat(sample, model: DebateFramework):
    try:
        result, records = model(sample)
        data_dict = {
            "event_idx": sample["event_idx"],
            "role_idx": sample["role_idx"],
            "event": sample["event"],
            "inference": result,
        }
        history = {
            "event_idx": sample["event_idx"],
            "role_idx": sample["role_idx"],
            "records": records,
        }
        return data_dict, history
    except Exception as e:
        LOGGER.error(e)
        LOGGER.error(traceback.format_exc())
        return None


def collate_fn(result, sample, out_fp, history_fp):
    if result is None:
        return
    out_fp.write(json.dumps(result[0], ensure_ascii=False) + "\n")
    history_fp.write(json.dumps(result[1], ensure_ascii=False) + "\n")
    out_fp.flush()
    history_fp.flush()


def main():
    config = load_config()
    set_seed(config["seed"])

    # Print Config
    max_key_length = max(len(k) for k in config)
    for k, v in config.items():
        if k == "llm_config":
            v = config["llm_config"][config["llm_name"]]
        LOGGER.info(f"{k:<{max_key_length}}: {v}")

    LOGGER.info("Load data")
    samples = load_data(
        config["event_relation_pair_file_path"],
        config["output_file_path"],
        config["role_file_path"],
        config["sample_role"],
    )

    if len(samples) == 0:
        LOGGER.info("No available data need to role play.")
        return

    LOGGER.info("Create DebateFramework")
    model = DebateFramework(
        writer_client=LLMClientGroup(
            [
                APIConfig(model, api_base, api_key)
                for model, api_base, api_key in zip(
                    config["writer_client"]["models"],
                    config["writer_client"]["api_bases"],
                    config["writer_client"]["api_keys"],
                )
            ]
        ),
        affirmative_client=LLMClientGroup(
            [
                APIConfig(model, api_base, api_key)
                for model, api_base, api_key in zip(
                    config["affirmative_client"]["models"],
                    config["affirmative_client"]["api_bases"],
                    config["affirmative_client"]["api_keys"],
                )
            ]
        ),
        negative_client=LLMClientGroup(
            [
                APIConfig(model, api_base, api_key)
                for model, api_base, api_key in zip(
                    config["negative_client"]["models"],
                    config["negative_client"]["api_bases"],
                    config["negative_client"]["api_keys"],
                )
            ]
        ),
        chairman_client=LLMClientGroup(
            [
                APIConfig(model, api_base, api_key)
                for model, api_base, api_key in zip(
                    config["chairman_client"]["models"],
                    config["chairman_client"]["api_bases"],
                    config["chairman_client"]["api_keys"],
                )
            ]
        ),
        debate_rounds=config["debate_rounds"],
        max_rewrite_num=config["max_rewrite_num"],
        writer_generation_config=config["writer_client"]["generation_config"],
        affirmative_generation_config=config["affirmative_client"]["generation_config"],
        negative_generation_config=config["negative_client"]["generation_config"],
        chairman_generation_config=config["chairman_client"]["generation_config"],
    )

    LOGGER.info("Start role play")
    worker = MultiThreadingRunner(config["max_workers"])

    with open(
        config["output_file_path"], "a+", encoding="utf-8", buffering=1
    ) as out_fp, open(
        config["history_file_path"], "a+", encoding="utf-8"
    ) as history_fp:
        worker(
            samples=samples,
            worker_func=partial(
                chat,
                model=model,
            ),
            callback_func=partial(collate_fn, writer=out_fp, history_writer=history_fp),
        )


if __name__ == "__main__":
    main()
