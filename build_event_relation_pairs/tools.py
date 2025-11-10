import os.path

from easy_tools.utils.io import FileReader
from typing import List
import json
from json_repair import repair_json
def get_unprocessed_data(
    input_file_path: str,
    output_file_path: str,
    id_key: str = 'index'
) -> List[dict]:
    """
    This function is used for continuous evaluation, filtering out data that has already been completed
    Args:
        event_file_path: The path of the jsonl file of the event to be evaluated
        output_file_path: The path to the output file (jsonl) of the evaluation results
    Returns:
        [{'event_idx': xx, 'event': xx}, ...]
    """
    existed_indexes = set()

    if os.path.exists(output_file_path):
        for item in FileReader.read(output_file_path, return_iter=True):
            existed_indexes.add(item[id_key])

    samples = []
    for item in FileReader.read(input_file_path, return_iter=True):
        if item[id_key] not in existed_indexes:
            samples.append(item)

    return samples



def parse_to_json(s: str):
    return json.loads(repair_json(s, ensure_ascii=False))