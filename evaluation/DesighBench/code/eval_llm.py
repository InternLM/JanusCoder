import os
# os.environ["HOME"] = "/root"  # playwright启动firefox浏览器需要这个环境变量
import random
random.seed(42)
import json
import argparse
from collections import defaultdict
import numpy as np
from pprint import pprint

from runner.main import Runner
from utils import Framework, Task, Mode
from evaluator.main import evaluate_generation, evaluate_edit, evaluate_repair



def compute_avg_scores(result_path: str):
    data = json.load(open(result_path, "r"))
    output_dict = defaultdict(list)
    for web_name in data:
        for metric, score in data[web_name].items():
            output_dict[metric].append(score)
    for k, v in output_dict.items():
        output_dict[k] = np.round(np.mean(np.array(v)), 4)
    
    for k, v in output_dict.items():
        print(f"{k}: {v:.4f}")
    return output_dict


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Qwen3-8B", type=str)
parser.add_argument("--base_url", default="http://10.130.128.114:8000/v1", type=str)
parser.add_argument("--api_key", default="EMPTY", type=str)
args = parser.parse_args()

# 30 20 8
generate_range = [82, 15, 4, 95, 36, 32, 29, 18, 117, 14, 87, 112, 70, 12, 76, 55, 5, 118, 107, 28, 30, 65, 78, 103, 72, 26, 92, 84, 90, 108]
edit_range = [54, 29, 58, 76, 36, 1, 21, 55, 44, 77, 20, 28, 72, 14, 12, 49, 13, 23, 73, 63]
repair_range = [20, 9, 26, 2, 24, 15, 18, 4]

# generate_range = list(range(1, 121))
# edit_range = list(range(1, 81))
# repair_range = list(range(1, 29))

runner = Runner(
    args.model_name,  # Name of the model to use (see mllm/__init__.py)
    framework=Framework.VANILLA,
    stream=False,  # Enable streaming output for stability (default: True)
    print_content=False,  # Print model outputs to console (default: False)
    base_url=args.base_url,
    api_key=args.api_key,
)


runner.run(
    task=Task.EDIT, 
    output_framework=Framework.VANILLA,
    mode=Mode.CODE,  # CODE, IMAGE, BOTH
    max_workers=20,
    execution_range=edit_range
)
runner.run(
    task=Task.REPAIR, 
    output_framework=Framework.VANILLA,
    mode=Mode.CODE,  # CODE, IMAGE, BOTH, MARK
    max_workers=20,
    execution_range=repair_range
)

evaluate_edit(
    models=[args.model_name],
    frame_works = ["vanilla"],
    modes=[Mode.CODE],
    llm_judge_flag=False,
    iterate_range=edit_range,
)
evaluate_repair(
    models=[args.model_name],
    frame_works = ["vanilla"], 
    modes=[Mode.CODE],
    llm_judge_flag=False,
    iterate_range=repair_range,
)


print("计算平均指标...")
print("编辑")
compute_avg_scores(f"/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench/res/DesignEdit/{args.model_name}_code.json")
print("="*80)
print("修复")
compute_avg_scores(f"/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench/res/DesignRepair/{args.model_name}_code.json")
print("="*80)


"""
Qwen2.5-VL-7B
generation
MAE: 61.124234885336044,
clip_similarity: 0.7414300026992957
structure_similarity: 0.6536884143221579

edit
MAE: 20.782273250367588
clip_similarity: 0.9192307189106941
code_score: 0.2229820263040545
structure_similarity: 0.8844046413816923

repair
MAE: 9.621132648245629
clip_similarity: 0.9395465829542705
code_score: 0.055494150173886876
issue accuracy: 0.34523809523809523
structure_similarity: 0.9299276221869901



qwen3_8b_mclm_mix_0819_1e5_xpuyu

edit
MAE: 14.3049
clip_similarity: 0.9446
structure_similarity: 0.8951
code_score: 0.2341

repair
MAE: 10.1495
clip_similarity: 0.9431
structure_similarity: 0.9269
code_score: 0.0117
issue accuracy: 0.0000




Qwen3-8B
edit
MAE: 14.6189
clip_similarity: 0.9404
structure_similarity: 0.8948
code_score: 0.3111

repair
MAE: 24.6323
clip_similarity: 0.8803
structure_similarity: 0.8889
code_score: 0.0711
issue accuracy: 0.2542
"""





