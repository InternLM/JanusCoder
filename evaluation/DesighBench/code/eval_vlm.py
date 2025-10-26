import os
import re
import random
random.seed(42)
import json
import argparse
from collections import defaultdict
import numpy as np
from pprint import pprint
from openai import OpenAI

from runner.main import Runner
from utils import Framework, Task, Mode
from evaluator.main import evaluate_generation, evaluate_edit, evaluate_repair




def compute_avg_scores(result_path: str):

    data = json.load(open(result_path, "r"))
    output_dict = defaultdict(list)
    for web_name in data:
        for metric, score in data[web_name].items():
            if metric == "llm score":
                try:
                    response = score
                    matches = re.findall(r'"[sS]core"\s*:\s*([0-9]|10),', response)
                    score = int(matches[0])
                except:
                    score = 0
            output_dict[metric].append(score)
    for k, v in output_dict.items():
        output_dict[k] = np.round(np.mean(np.array(v)), 4)
    
    for k, v in output_dict.items():
        print(f"{k}: {v:.4f}")
    return output_dict


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct")
parser.add_argument("--base_url", default="", type=str)
parser.add_argument("--api_key", default="", type=str)
args = parser.parse_args()



generate_range = list(range(1, 121))
edit_range = list(range(1, 81))

generate_save_dir = f"results/generation/{args.model_name}/"
edit_save_dir = f"results/edit/{args.model_name}/"

generation_score_file = f"scores/generation/{args.model_name}.json"
edit_score_file = f"scores/edit/{args.model_name}_{args.mode}.json"



mode = Mode.BOTH 

runner = Runner(args.model_name, framework=Framework.VANILLA, stream=False, print_content=False, base_url=args.base_url, api_key=args.api_key)
runner.run(task=Task.GENERATION, output_framework=Framework.VANILLA, execution_range=generate_range, max_workers=20, save_dir=generate_save_dir, mode=Mode.IMAGE)

runner.run(task=Task.EDIT, output_framework=Framework.VANILLA, execution_range=edit_range, max_workers=20, save_dir=edit_save_dir, mode=mode)



if not os.path.exists(generation_score_file):
    evaluate_generation(generate_range, generate_save_dir, generation_score_file)
if not os.path.exists(edit_score_file):
    evaluate_edit(edit_range, mode, edit_save_dir, edit_score_file, llm_judge_flag=True)


print("Generation")
compute_avg_scores(generation_score_file)
print("="*80)
print("Edit")
compute_avg_scores(edit_score_file)











