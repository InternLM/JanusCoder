import os
import shutil
from tqdm import tqdm
import multiprocessing as mp
import argparse
from typing import *

from utils import load_batches, screenshot, make_html_offline

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="output/generation")
args = parser.parse_args()

DATA_FILES = [f"{args.input_path}/{i:05d}.parquet" for i in range(2, 600)]

SAMPLE_NUM = 400000
BATCH_SIZE = 1000
PROCESS_NUM = 10
OUTPUT_DIR = args.output_path


def rollout(idx: int, item: Dict):
    save_path = os.path.join(OUTPUT_DIR, f"{idx}")
    if os.path.exists(save_path):
        return
    os.makedirs(save_path)

    try:
        html_code = item["text"]
        screenshot(html_code, os.path.join(save_path, "screenshot_init.png"))
        with open(os.path.join(save_path, "code_init.html"), "w", encoding="utf-8") as f:
            f.write(html_code)

        html_code_1 = make_html_offline(html_code)
        screenshot(html_code_1, os.path.join(save_path, "screenshot.png"))
        with open(os.path.join(save_path, "code.html"), "w", encoding="utf-8") as f:
            f.write(html_code_1)
    except Exception as e:
        shutil.rmtree(save_path)


def process_chunk(chunk: Dict):
    dataset = chunk["dataset"]
    offset = chunk["offset"]

    item_idx = 0
    for item in tqdm(dataset):
        idx = item_idx + offset
        rollout(idx, item)
        item_idx += 1


if __name__ == "__main__":
    chunks = load_batches(DATA_FILES, SAMPLE_NUM, BATCH_SIZE)

    with mp.Pool(PROCESS_NUM) as pool:
        pool.map(process_chunk, chunks)


