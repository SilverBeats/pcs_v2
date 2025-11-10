#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset Construction"""
import threading
from functools import partial
from collections import defaultdict, Counter
from easy_tools.utils.io import FileReader, FileWriter
import os
from easy_tools.utils.tools import get_dir_file_path, get_logger, get_file_name_and_ext
from tools import get_unprocessed_data
from easy_tools.llms.client import APIConfig, LLMClientGroup
from easy_tools.llms.chain import LLMChain
from prompts import RateEventRelationPrompt
from easy_tools.utils.concurrent import MultiThreadingRunner
import json
CONFIG_FILE_PATH = "./config.yaml"
LOCK = threading.Lock()

def load_config() -> dict:
    config = FileReader.read(CONFIG_FILE_PATH, return_dict=True)

    atomic_file_path = config["atomic_file_path"]
    file_name = get_file_name_and_ext(atomic_file_path)[0]
    os.makedirs(config["output_dir"], exist_ok=True)

    llm_name = config['llm_name']
    config['llm_config'] = config['llm_config'][llm_name]

    config["rate_file_path"] = os.path.join(
        config["output_dir"], f"{file_name}_{llm_name}_rate.jsonl"
    )
    return config

def _worker_func(sample: dict, model: LLMChain, fp):
    try:
        result = model(sample['event'])
        with LOCK:
            # {'event_idx': xx, 'event': xx, 'xAttr': score, 'xReact': score, ...}
            fp.write(json.dumps( result, ensure_ascii=False) + '\n')
            fp.flush()
    except Exception as e:
        print(e)

def stat_scores(data_file_path: str):
    scores = defaultdict(list)
    for sample in FileReader.read(data_file_path):
        try:
            for relation in sample:
                if relation in ['event_idx' , 'event']:
                    continue
                scores[relation].append(float(sample[relation]))
        except Exception as e:
            print(e)

    stats_results = defaultdict(dict)
    for relation, rel_scores in scores.items():
        stats_results[relation] = {
            "mean": sum(rel_scores) / len(rel_scores),
            "max": max(rel_scores),
            "min": min(rel_scores),
            "dist": Counter(rel_scores),
            "samples": len(rel_scores),
        }

    score_file_path = data_file_path.replace('rate.jsonl', 'rate_score.json')
    FileWriter.dump(stats_results, score_file_path)

def select_event_relation_pairs(output_dir: str, threshold: int):
    score_file_paths = get_dir_file_path(
        dir_path=output_dir,
        file_exts=['jsonl'],
        should_skip_file=lambda x: not x.endswith('_rate.jsonl')
    )

    # {
    #   (event_idx, event): {
    #       'relation': [score1, score2, score3]
    #   }
    # }
    records = defaultdict(lambda: defaultdict(list))
    relations = None
    for score_file_path in score_file_paths:
        for row in FileReader.read(score_file_path):
            event_idx, event = row["event_idx"], row["event"]
            if relations is None:
                relations = list(row.keys())
                relations.remove('event_idx')
                relations.remove('event')

            for relation in relations:
                if relation not in row:
                    continue
                records[(event_idx, event)][relation].append(row[relation])

    results = []
    for (event_idx, event), rel_scores in records.items():
        good_relations = []
        for relation, scores in rel_scores.items():
            if all(score > threshold for score in scores):
                good_relations.append(relation)
        results.append( {
            "event_idx": event_idx,
            "event": event,
            "relations": good_relations
        })

    output_file_path = os.path.join(
        output_dir, f"event_relation_pairs_{threshold}.jsonl"
    )
    FileWriter.dump(results, output_file_path)

def main():
    config = load_config()
    print(config)

    samples = get_unprocessed_data(
        input_file_path=config["atomic_file_path"], output_file_path=config["rate_file_path"],
        id_key='event_idx'
    )

    if len(samples) > 0:
        llm_name = config['llm_name']
        api_configs = [
            APIConfig(model, api_base, api_key)
            for model, api_base, api_key in zip(
                config['llm_config']['models'],
                config['llm_config']['api_bases'],
                config['llm_config']['api_keys']
            )
        ]
        chain = LLMChain(
            client_group=LLMClientGroup(api_configs),
            prompt_template=RateEventRelationPrompt(),
            **config['generation_config']
        )

        runner = MultiThreadingRunner(config['num_workers'])
        with open(config['rate_file_path'], 'a+', encoding='utf-8') as fp:
            runner(
                samples=samples,
                worker_func=partial(_worker_func, model=chain, fp=fp),
                desc=f'{llm_name} Rating'
            )

    print("Calculate score statistics")
    stat_scores(config["output_file_path"])

    if config['select_event_relation_pairs']:
        print("Select event relation pairs")
        select_event_relation_pairs(
            output_dir=config['output_dir'],
            threshold=config['threshold']
        )

if __name__ == "__main__":
    main()
