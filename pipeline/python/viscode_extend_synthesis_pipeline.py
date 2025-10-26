from multiprocessing import Pool, cpu_count
import multiprocessing
from tqdm import tqdm
from typing import Tuple, List, Dict
from functools import partial
from openai import OpenAI
import sys
import tempfile
import subprocess
import random
import time
import httpx
import json
import os
import re
import base64
import psutil
import threading

def build_client(_base_url, _api_key, _timeout, _httpx_client) -> OpenAI:
    return OpenAI(
        base_url=_base_url,
        api_key=_api_key,
        default_headers={"Authorization": f"Bearer {_api_key}"},
        http_client=_httpx_client
    )

def extract_python_code_block(text: str):
    pattern = r"```python\s+([\s\S]*?)\s+```"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        code_block = match.group(1).strip()
        return code_block, True
    else:
        return None, False

def convert_image_to_url(image_path: str, image_type: str="png") -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        image_url = f"data:image/{image_type};base64,{img_b64}"
    return image_url

def chat_response(_client: OpenAI, _messages: List, _max_tokens: int, _temperature: float, _model: str) -> Tuple[str, str]:
    response = _client.chat.completions.create(
        model=_model,
        messages=_messages,
        temperature=_temperature,
        max_tokens=_max_tokens,
        stream=False,
        timeout=None
    )
    return response.choices[0].message.content, str(response.usage)

def run_script(_identifier, _code: str, _save_root: str, _timeout: int) -> Dict:
    full_name = _identifier + str(int(time.time()))
    script_path = os.path.join(_save_root, "{0}.py".format(full_name))

    return_dict = {'has_code': False, 'success': False, 'plot': False, 'path': None}

    extracted_code, has_code = extract_python_code_block(_code)
    # print(extracted_code)
    if not has_code:
        return return_dict
    else:
        return_dict['has_code'] = True

    full_code = (
        "import plotly.graph_objects as go\n"
        "import pandas as pd\n"
        "import numpy as np\n\n"
        + extracted_code
    )

    if "fig.show()" in full_code:
        full_code = full_code.replace("fig.show()", "fig.write_image(\"{0}.png\")".format(full_name))
        
    if "plt.show()" in full_code:
        full_code = full_code.replace("plt.show()", "plt.savefig(\"{0}.png\")".format(full_name))

    with open(script_path, "w") as f:
        f.write(full_code)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=_save_root,
            capture_output=True,
            text=True,
            timeout=_timeout,
            check=False,
            start_new_session=True,
            close_fds=True
        )
        if result.returncode == 0:
            return_dict['success'] = True
            img_path = os.path.join(_save_root, '{0}.png'.format(full_name))
            if os.path.exists(img_path):
                return_dict['plot'] = True
                return_dict['path'] = img_path
        return return_dict
    except subprocess.TimeoutExpired as e:
        try:
            if hasattr(e, "pid"):
                os.killpg(e.pid, 9)
        except Exception:
            pass
        return return_dict

def select_directions(_candidate_directions, _rng: random.Random) -> str:
    _num_directions = _rng.randint(1, 3)
    _target_directions = _rng.sample(_candidate_directions, k=_num_directions)
    _directions_str = ", ".join(_target_directions)
    return _directions_str

def process_sample(_rng_seed: int, _dataset, _gen_url, _gen_key, _gen_model, _judge_url, _judge_key, _judge_model, _api_timeout, _exe_timeout, _gen_const_pack):
    cid = str(_rng_seed)

    rand_generator = random.Random(_rng_seed)

    sample_size = 2
    sampled_list = rand_generator.sample(_dataset, sample_size)

    old_description_list = []
    for sample in sampled_list:
        old_description = sample["task_info"]["description"]
        old_description_list.append(old_description)

    with httpx.Client(verify=False, timeout=_api_timeout) as gen_http_client, httpx.Client(verify=False, timeout=_api_timeout) as judge_http_client:
        gen_client = build_client(_gen_url, _gen_key, _api_timeout, gen_http_client)
        judge_client = build_client(_judge_url, _judge_key, _api_timeout, judge_http_client)

        new_item = {
            "id": cid,
            "new_pass": False,
            "new_has_plot": False,
            "new_judge_response": None,
            "new_pass_judge": False,
            "new_description": None,
            "new_code": None
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            gen_max_length = _gen_const_pack['gen_max_length']
            gen_temperature = _gen_const_pack['gen_temperature']
            judge_max_length = _gen_const_pack['judge_max_length']
            judge_temperature = _gen_const_pack['judge_temperature']
            system_prompt_for_desc = _gen_const_pack['system_prompt_for_desc']
            user_prompt_for_desc_template = _gen_const_pack['user_prompt_for_desc_template']
            system_prompt_for_judge = _gen_const_pack['system_prompt_for_judge']
            user_prompt_for_judge_template = _gen_const_pack['user_prompt_for_judge_template']

            try:
                gen_user_prompt = user_prompt_for_desc_template.format(
                    old_description_list[0], old_description_list[1]
                )
                gen_desc_messages = [
                    {'role': 'system', 'content': system_prompt_for_desc},
                    {'role': 'user', 'content': gen_user_prompt}
                ]

                new_desc, _ = chat_response(gen_client, gen_desc_messages, gen_max_length, gen_temperature, _gen_model)
                final_new_desc = new_desc.strip()

                new_item["new_description"] = new_desc

                gen_code_prompt = (
                    f"You will be given some instructions on data visualization. Write python codes that strictly follows those instructions. Use matplotlib library to do the plotting.\n"
                    f"Instuctions:"
                    f"{final_new_desc}"
                )

                gen_code_messages = [
                    {'role': 'system', 'content': "You are a code assistant on data visualization. You will only write python code block."},
                    {'role': 'user', 'content': gen_code_prompt}
                ]

                code_content, _ = chat_response(gen_client, gen_code_messages, gen_max_length, gen_temperature, _gen_model)

                # if "```python" in code_content:
                #     final_new_code = code_content.split("```python", 1)[1].split("```", 1)[0].strip()
                # else:
                #     final_new_code = code_content.strip()

                new_item["new_code"] = code_content

                new_runpack = run_script(cid, code_content, tmp_dir, _exe_timeout)
                if new_runpack['success']:
                    new_item['new_pass'] = True
                if new_runpack['plot']:
                    new_item['new_has_plot'] = True
                    new_plot_fp = new_runpack['path']
                    
                    judge_user_prompt = user_prompt_for_judge_template.format(new_desc, code_content)
                    judge_messages = [
                        {'role': 'system', 'content': system_prompt_for_judge},
                        {'role': 'user', 'content': [
                            {"type": "image_url", "image_url": {"url":convert_image_to_url(new_plot_fp)}},
                            {"type": "text", "text": judge_user_prompt},
                        ]}
                    ]
                    judge_response, _ = chat_response(judge_client, judge_messages, judge_max_length, judge_temperature, _judge_model)
                    if "```json" in judge_response:
                        judge_json = judge_response.split("```json", 1)[1].split("```", 1)[0].strip()
                    else:
                        judge_json = judge_response.strip()
                    new_item["new_judge_response"] = judge_json
                    
                    try:
                        judge_dict = json.loads(judge_json)
                        if "Total Score" in judge_dict and (judge_dict["Total Score"] >= 3 or float(judge_dict["Total Score"]) >= 3.0):
                            new_item["new_pass_judge"] = True
                        else:
                            new_item["new_pass_judge"] = False
                    except Exception as e:
                        new_item["new_pass_judge"] = False
            except Exception as e:
                print(e)
        gen_client.close()
        judge_client.close()
    return new_item

if __name__ == "__main__":
    source_fp = '/path/to/source.jsonl'
    dump_fp = '/path/to/output.jsonl'
    api_timeout = 60
    execution_timeout = 20
    concurrency = 32

    gen_url = "https://generation/api/address"
    gen_key = "generation/api/key"
    gen_model = "generation/model/name"

    judge_url = "https://judge/api/address"
    judge_key = "generation/api/key"
    judge_model = "generation/model/name"

    gen_trial = int(1e5)

    gen_constant_pack = {}
    gen_constant_pack['gen_max_length'] = 100072
    gen_constant_pack['gen_temperature'] = 0.5
    gen_constant_pack['judge_max_length'] = 12800
    gen_constant_pack['judge_temperature'] = 0.0

    gen_constant_pack['system_prompt_for_desc'] = (
        "You are an expert data visualization assistant."
        "Your primary task is to generate clear, actionable descriptions for creating or editing data visualizations in Python using matplotlib."
        "You must carefully analyze the user's input to determine if it's already a visualization task or something else."
    )
    gen_constant_pack['user_prompt_for_desc_template'] = """
You will be given two example descriptions of data visualization. Your task is to generate a new visualization instruction.
Here is your generation logic:
1.  If the given description is about data visualization (charts, plots, maps), create a new instruction that can visualize a similar problem or make a different kind of plot;
2.  If the given description is NOT about data visualization, create a brand new visualization instruction based on the core topic of the original description.

Your output should have two part: plot description and plot style description, and you should follow the following format:
1. Plot Description: Your new plot description
2. Plot Style Description: Your new description for the plotting style

The two example descriptions are:
Example 1:
{0}

Example 2:
{1}
""".strip()
    gen_constant_pack['system_prompt_for_judge'] = "You are a Senior AI Data Visualization Synthesis Quality Assurance Expert. Your mission is to provide a rigorous, objective, and multi-faceted evaluation of AI-generated data visualizations."
    gen_constant_pack['user_prompt_for_judge_template'] = """
You will be given a triplet of information:
1.  A natural language `Instruction`;
2.  The `Code` generated to fulfill it;
3.  The resulting `Image`.

Your evaluation must follow a detailed Chain of Thought process, analyzing each component before assigning a score.

---
### **Evaluation Framework**

**Stage 1: Comprehensive Task Understanding**
* **Analyze the Instruction:** Deconstruct the user's request to identify all explicit requirements (e.g., chart type, title, colors) and implicit intents (e.g., the information to be conveyed).
* **Analyze the Code:** Review the generated code for correctness, logic, and quality.
* **Analyze the Image:** Inspect the final visual output to assess its accuracy and clarity.

**Stage 2: Multi-dimensional Rating & Scoring**
Based on your analysis, you will rate the triplet across four dimensions. Then, you will provide a final score based on the detailed guidelines below.

#### **Evaluation Dimensions**

1.  **Task Completion:** This measures the extent to which the final image and code successfully fulfill all aspects of the instructed task.
    * **Accuracy:** Does the image accurately represent the data and adhere to all specified chart types, labels, and titles?
    * **Completeness:** Are all parts of the instruction addressed? Are any requirements missing?

2.  **Solution Coherence & Code Quality:** 
    * **Logic & Efficiency:** Does the code follow a logical and efficient sequence of operations to generate the visualization? 
    * **Correctness & Readability:** Is the code syntactically correct and executable? Does it follow standard programming best practices for clarity?

3.  **Visual Clarity:** This assesses the aesthetic and communicative quality of the final image.
    * **Readability:** Is the chart easy to read and interpret? Are fonts, colors, and labels clear?
    * **Aesthetics & Layout:** Is the visualization well-designed and visually appealing? Is the layout balanced, free of clutter and overlapping elements?

4.  **Task Relevance:** This measures the practical, real-world value of the assigned task.
    * **Practicality:** Does the instruction represent a realistic and useful data visualization scenario?
    * **Value:** Does the task serve as a meaningful benchmark for a valuable AI capability?

#### **Scoring Guidelines (1-5 Scale)**

* **5 (Excellent):** The task is perfectly completed with no flaws. The code is efficient, clean, and logical. The visual output is clear, accurate, and aesthetically excellent. A flawless submission.
* **4 (Good):** The task is mostly completed and achieves the core objective, but with minor, non-critical issues. This could be a small element missing from the chart, slight code inefficiency, or minor visual imperfections.
* **3 (Fair):** The task is only partially completed, or the output has significant flaws. For example, the chart type is wrong, the data is misrepresented, the code is highly inefficient, or the visual is cluttered and hard to read.
* **2 (Poor):** The solution attempts the task but deviates significantly from the instructions. The code may run, but the resulting image is largely incorrect, misleading, or irrelevant to the user's request.
* **1 (Failed):** The task fails completely. The code is non-executable, produces an error, or the output is completely unusable.
---

### **Output Specification**

Your final output must be a single JSON object. It must include your detailed `Chain of Thought` reasoning, a score for each of the four dimensions, and a final `Total Score` (the average of the dimensional scores).

**Illustrative Example:**

**Data Triplet:**
* **Instruction:** "Generate a horizontal bar chart showing the projected 2024 revenue for 'Product Alpha', 'Product Beta', and 'Product Gamma'. Revenues are $4M, $5.5M, and $3.2M respectively. Use a blue color palette and title the chart 'Projected Revenue 2024'."
* **Code:**
    ```python
    import matplotlib.pyplot as plt

    products = ['Product Alpha', 'Product Beta', 'Product Gamma']
    revenues = [4, 5.5, 3.2]

    plt.figure(figsize=(10, 6))
    plt.barh(products, revenues, color='skyblue')
    plt.xlabel('Projected Revenue (in Millions)')
    plt.ylabel('Product')
    # Note: The title was forgotten in the code.
    plt.tight_layout()
    plt.show()
    ```
* **Image:** [An image of a horizontal bar chart with the correct data, labels, and blue color. However, the chart has no title.]

**Output:**
```json
{{
  "Chain of Thought": "1. **Task Understanding:** The instruction requires a horizontal bar chart for three products with specific revenue figures. It explicitly asks for a blue palette and a specific title, 'Projected Revenue 2024'. 2. **Code Analysis:** The Python code uses matplotlib correctly. It defines the correct data and uses `barh` for a horizontal chart. The color 'skyblue' fits the 'blue color palette' requirement. However, the line to add the title (`plt.title(...)`) is missing. The code is clean and executable. 3. **Image Analysis:** The image shows the correct chart type and data. The axes are labeled correctly. The color is blue. The only missing element is the title specified in the instruction. 4. **Rating:** Task Completion is flawed because the title is missing. Solution Coherence is good as the code is logical, just incomplete. Visual Clarity is good but could be better with a title. Task Relevance is high as this is a very common business chart.",
  "Task Completion": "3",
  "Solution Coherence & Code Quality": "4",
  "Visual Clarity": "4",
  "Task Relevance": "5",
  "Total Score": "4.0"
}}
```
---
The resulting image is given at the beginning.
The natural language instruction is: {0}
The code generated is:
{1}
""".strip()

    with open(source_fp, "r") as f_in:
        all_samples = [json.loads(l) for l in f_in]

    # total = min(len(all_samples), process_end_idx)
    total = gen_trial
    # all_samples = all_samples[process_start_idx: total]

    # rng_seed = int(time.time())
    rng_seeds = [int(time.time()) + i for i in range(total)]

    multiprocessing.set_start_method("spawn")

    worker = partial(process_sample, _dataset=all_samples, _gen_url=gen_url, _gen_key=gen_key, _gen_model=gen_model, _judge_url=judge_url, _judge_key=judge_key, _judge_model=judge_model, _api_timeout=api_timeout, _exe_timeout=execution_timeout, _gen_const_pack=gen_constant_pack)

    pass_cnt = 0
    fail_cnt = 0

    with open(dump_fp, "w") as outfile, Pool(processes=concurrency, maxtasksperchild=200) as pool:
        with tqdm(total=total, desc="Processing ") as pbar:
            for new_item in pool.imap_unordered(worker, rng_seeds):
                json.dump(new_item, outfile, ensure_ascii=False)
                outfile.write("\n")
                if new_item['new_pass_judge']:
                    pass_cnt += 1
                else:
                    fail_cnt += 1

                pbar.set_postfix({"Passed": pass_cnt, "Failed": fail_cnt})
                pbar.update(1)

    print(f"Total: {total}, Passed: {pass_cnt}, Failed: {fail_cnt}")