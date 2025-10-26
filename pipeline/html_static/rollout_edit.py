import os
import json
import time
import shutil
import argparse
from typing import *
import multiprocessing as mp

from utils import load_batches, screenshot, make_html_offline
from editor import generate, verify


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="output/edit")
parser.add_argument("--start", type=int, default=600, help="parquet begin index")
parser.add_argument("--end", type=int, default=900, help="parquet end index")
parser.add_argument("--mode", type=str, default="image")
parser.add_argument("--use_preprocess", type=int, default=0)
args = parser.parse_args()
assert args.mode in ["image", "both"]
assert args.use_preprocess in [0, 1]

# 
DATA_FILES = [f"{args.input_path}/{i:05d}.parquet" for i in range(args.start, args.end)]
OUTPUT_DIR = args.output_path
SAMPLE_NUM = 200000
BATCH_SIZE = 1000
PROCESS_NUM = 30


def rollout(idx: int, item: Dict):
    save_path = os.path.join(OUTPUT_DIR, f"{idx}")
    if os.path.exists(os.path.join(save_path, "instruction.json")):
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        begin_time = time.time()
        if args.use_preprocess == 1:
            code_init = item["text"]
            screenshot(code_init, os.path.join(save_path, "init.png"))
            with open(os.path.join(save_path, "init.html"), "w", encoding="utf-8") as f:
                f.write(code_init)

            code_0 = make_html_offline(code_init)
            screenshot(code_0, os.path.join(save_path, "0.png"))
            with open(os.path.join(save_path, "0.html"), "w", encoding="utf-8") as f:
                f.write(code_0)
        else:
            code_0 = item["text"]
            screenshot(code_0, os.path.join(save_path, "0.png"))
            with open(os.path.join(save_path, "0.html"), "w", encoding="utf-8") as f:
                f.write(code_0)

        instruct, code_1, chat_history = generate(code_0, os.path.join(save_path, "0.png"), args.mode)
        screenshot(code_1, os.path.join(save_path, "1.png"))
        with open(os.path.join(save_path, "1.html"), "w", encoding="utf-8") as f:
            f.write(code_1)

        judgement = verify(code_0, os.path.join(save_path, "0.png"), code_1, os.path.join(save_path, "1.png"), instruct["human"], chat_history, args.mode)
        instruct["judgement"] = judgement

        json.dump(instruct, open(os.path.join(save_path, "instruction.json"), "w"), ensure_ascii=False, indent=4)
        json.dump(chat_history, open(os.path.join(save_path, "messages.json"), "w"), ensure_ascii=False, indent=4)
        print(f"{idx} time: {int(time.time() - begin_time)}")

    except Exception as e:
        shutil.rmtree(save_path)


def process_batch(batch: Dict):
    dataset = batch["dataset"]
    offset = batch["offset"]
    sample_idx = 0
    for item in dataset:
        idx = sample_idx + offset
        rollout(idx, item)
        sample_idx += 1


if __name__ == "__main__":    
    batches = load_batches(DATA_FILES, SAMPLE_NUM, BATCH_SIZE)

    with mp.Pool(PROCESS_NUM) as pool:
        pool.map(process_batch, batches)
