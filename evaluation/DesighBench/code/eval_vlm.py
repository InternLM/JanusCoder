import os
import re
# os.environ["HOME"] = "/root"  # playwright启动firefox浏览器需要这个环境变量
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




def compute_avg_scores(result_path: str, task=None):
    if task == "generation":
        task_range = [82, 15, 4, 95, 36, 32, 29, 18, 117, 14, 87, 112, 70, 12, 76, 55, 5, 118, 107, 28, 30, 65, 78, 103, 72, 26, 92, 84, 90, 108]
    elif task == "edit":
        task_range = [54, 29, 58, 76, 36, 1, 21, 55, 44, 77, 20, 28, 72, 14, 12, 49, 13, 23, 73, 63]
    elif task == "repair":
        task_range = [20, 9, 26, 2, 24, 15, 18, 4]
    else:
        task_range = None

    data = json.load(open(result_path, "r"))
    output_dict = defaultdict(list)
    for web_name in data:
        if task_range is not None and int(web_name) not in task_range:
            continue
        for metric, score in data[web_name].items():
            if metric == "llm score":
                try:
                    # score = json.loads(score.replace("```json", "").replace("```", ""))["score"]
                    response = score
                    matches = re.findall(r'"[sS]core"\s*:\s*([0-9]|10),', response)
                    score = int(matches[0])
                except:
                    print(web_name)
                    print(response)
                    print("\n\n")
                    score = 0
            output_dict[metric].append(score)
    for k, v in output_dict.items():
        # print(len(v)) 
        output_dict[k] = np.round(np.mean(np.array(v)), 4)
    
    for k, v in output_dict.items():
        print(f"{k}: {v:.4f}")
    return output_dict


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct")
parser.add_argument("--base_url", default="http://10.130.128.114:8994/v1", type=str)
parser.add_argument("--api_key", default="skk", type=str)
parser.add_argument("--is_lite", default=0, type=int)
parser.add_argument("--use_runner", default=1, type=int)
parser.add_argument("--mode", default="both", type=str)
args = parser.parse_args()
assert args.is_lite in [0, 1]
assert args.use_runner in [0, 1]
assert args.mode in ["both", "code", "image"]

if args.is_lite == 1:
    generate_range = [82, 15, 4, 95, 36, 32, 29, 18, 117, 14, 87, 112, 70, 12, 76, 55, 5, 118, 107, 28, 30, 65, 78, 103, 72, 26, 92, 84, 90, 108]
    edit_range = [54, 29, 58, 76, 36, 1, 21, 55, 44, 77, 20, 28, 72, 14, 12, 49, 13, 23, 73, 63]
    repair_range = [20, 9, 26, 2, 24, 15, 18, 4]

    generate_save_dir = f"results/generation_lite/{args.model_name}/"
    edit_save_dir = f"results/edit_lite/{args.model_name}/"
    repair_save_dir = f"results/repair_lite/{args.model_name}/"

    generation_score_file = f"scores/generation_lite/{args.model_name}.json"
    edit_score_file = f"scores/edit_lite/{args.model_name}_{args.mode}.json"
    repair_score_file = f"scores/repair_lite/{args.model_name}_{args.mode}.json"
else:
    generate_range = list(range(1, 121))
    edit_range = list(range(1, 81))
    repair_range = list(range(1, 29))

    generate_save_dir = f"results/generation/{args.model_name}/"
    edit_save_dir = f"results/edit/{args.model_name}/"
    repair_save_dir = f"results/repair/{args.model_name}/"

    generation_score_file = f"scores/generation/{args.model_name}.json"
    edit_score_file = f"scores/edit/{args.model_name}_{args.mode}.json"
    repair_score_file = f"scores/repair/{args.model_name}_{args.mode}.json"


if args.mode == "both":
    mode = Mode.BOTH 
elif args.mode == "code":
    mode = Mode.CODE
elif args.mode == "image":
    mode = Mode.IMAGE
else:
    assert 0, args.mode

if args.use_runner == 1:
    runner = Runner(args.model_name, framework=Framework.VANILLA, stream=False, print_content=False, base_url=args.base_url, api_key=args.api_key)
    runner.run(task=Task.GENERATION, output_framework=Framework.VANILLA, execution_range=generate_range, max_workers=20, save_dir=generate_save_dir, mode=Mode.IMAGE)
    print("edit...")
    runner.run(task=Task.EDIT, output_framework=Framework.VANILLA, execution_range=edit_range, max_workers=20, save_dir=edit_save_dir, mode=mode)
    runner.run(task=Task.REPAIR, output_framework=Framework.VANILLA, execution_range=repair_range, max_workers=20, save_dir=repair_save_dir, mode=mode)

print("开始验证...")
if not os.path.exists(generation_score_file):
    evaluate_generation(generate_range, generate_save_dir, generation_score_file)
if not os.path.exists(edit_score_file):
    evaluate_edit(edit_range, mode, edit_save_dir, edit_score_file, llm_judge_flag=True)
if not os.path.exists(repair_score_file):
    evaluate_repair(repair_range, mode, repair_save_dir, repair_score_file, llm_judge_flag=True)

print("计算平均指标...")
print("生成")
compute_avg_scores(generation_score_file)
print("="*80)
print("编辑")
compute_avg_scores(edit_score_file)
print("="*80)
print("修复")
compute_avg_scores(repair_score_file)
print("="*80)



# if args.is_lite == 0:
#     print("计算验证集的平均指标...")
#     print("生成")
#     compute_avg_scores(generation_score_file, "generation")
#     print("="*80)
#     print("编辑")
#     compute_avg_scores(edit_score_file, "edit")
#     # print("="*80)
#     # print("修复")
#     # compute_avg_scores(repair_score_file, "repair")
#     # print("="*80)








