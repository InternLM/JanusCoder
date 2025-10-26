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

def process_sample(_sample: Dict, _rng_seed: int, _gen_url, _gen_key, _gen_model, _judge_url, _judge_key, _judge_model, _api_timeout, _exe_timeout, _gen_const_pack):
    cid = _sample["id"]

    rand_generator = random.Random(cid + str(_rng_seed))

    with httpx.Client(verify=False, timeout=_api_timeout) as gen_http_client, httpx.Client(verify=False, timeout=_api_timeout) as judge_http_client:
        gen_client = build_client(_gen_url, _gen_key, _api_timeout, gen_http_client)
        judge_client = build_client(_judge_url, _judge_key, _api_timeout, judge_http_client)

        old_desc = _sample["task_info"]["description"]
        old_code = _sample["turns"][0]["output"]["content"]["value"]

        new_item = {
            "id": cid,
            "old_has_code": False,
            "old_pass": False,
            "old_has_plot": False,
            "new_pass": False,
            "new_has_plot": False,
            "new_judge_response": None,
            "new_pass_judge": False,
            "edit_directions": None,
            "old_description": old_desc,
            "old_code": old_code,
            "new_description": None,
            "new_code": None
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            old_runpack = run_script(cid, old_code, tmp_dir, _exe_timeout)
            new_item['old_has_code'] = old_runpack['has_code']
            new_item['old_pass'] = old_runpack['success']
            new_item['old_has_plot'] = old_runpack['plot']

            if not new_item['old_pass'] or not new_item['old_has_plot']:
                gen_client.close()
                judge_client.close()
                return new_item
            
            old_plot_fp = old_runpack['path']

            explore_retries = _gen_const_pack['max_retries']
            gen_max_length = _gen_const_pack['gen_max_length']
            gen_temperature = _gen_const_pack['gen_temperature']
            judge_max_length = _gen_const_pack['judge_max_length']
            judge_temperature = _gen_const_pack['judge_temperature']
            modify_directions = _gen_const_pack['modify_directions']
            system_prompt_for_desc = _gen_const_pack['system_prompt_for_desc']
            user_prompt_for_desc_template = _gen_const_pack['user_prompt_for_desc_template']
            system_prompt_for_judge = _gen_const_pack['system_prompt_for_judge']
            user_prompt_for_judge_template = _gen_const_pack['user_prompt_for_judge_template']

            for _ in range(explore_retries):
                try:
                    chosen_directions = select_directions(modify_directions, rand_generator)

                    gen_user_prompt = user_prompt_for_desc_template.format(
                        old_desc=old_desc, directions_str=chosen_directions
                    )
                    gen_desc_messages = [
                        {'role': 'system', 'content': system_prompt_for_desc},
                        {'role': 'user', 'content': gen_user_prompt}
                    ]

                    new_desc, _ = chat_response(gen_client, gen_desc_messages, gen_max_length, gen_temperature, _gen_model)
                    final_new_desc = new_desc.strip()
                    final_directions = chosen_directions

                    new_item["edit_directions"] = final_directions
                    new_item["new_description"] = new_desc

                    gen_code_prompt = (
                        f"The following is a Python Plotly code snippet:\n\n{old_code}\n\n"
                        f"Based on the new instruction below, modify the code accordingly. "
                        f"If the instruction describes a brand new visualization, you should replace the old code entirely.\n\n"
                        f"{final_new_desc}\n\n"
                        "Provide only the modified or new Python code in your response, inside a single code block. Keep it executable."
                    )

                    gen_code_messages = [
                        {'role': 'system', 'content': "You are a helpful code assistant."},
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
                        
                        judge_user_prompt = user_prompt_for_judge_template.format(new_desc)
                        judge_messages = [
                            {'role': 'system', 'content': system_prompt_for_judge},
                            {'role': 'user', 'content': [
                                {"type": "image_url", "image_url": {"url": convert_image_to_url(old_plot_fp)}},
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
                            if "Final Result" in judge_dict and (judge_dict["Final Result"] == 1 or judge_dict["Final Result"] == "1"):
                                new_item["new_pass_judge"] = True
                                break
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
    # concurrency = max(int(cpu_count() / 2), 1)
    concurrency = 32

    gen_url = "https://generation/api/address"
    gen_key = "generation/api/key"
    gen_model = "generation/model/name"

    judge_url = "https://judge/api/address"
    judge_key = "generation/api/key"
    judge_model = "generation/model/name"

    process_start_idx = 0
    process_end_idx = 300000

    gen_constant_pack = {}
    gen_constant_pack['max_retries'] = 3
    gen_constant_pack['gen_max_length'] = 100072
    gen_constant_pack['gen_temperature'] = 0.5
    gen_constant_pack['judge_max_length'] = 12800
    gen_constant_pack['judge_temperature'] = 0.0

    gen_constant_pack['modify_directions'] = [
        "change marker color", "change marker shape", "change marker size", "adjust marker opacity",
        "change line style", "adjust line width", "change bar color", "set x-axis title",
        "set y-axis range", "change axis to logarithmic scale", "format axis tick labels",
        "add a main title", "add annotations to a data point", "change plot theme/template",
        "adjust figure size", "add a legend title", "change legend position",
        "apply a continuous color scale", "apply a sequential color scale", "switch map style",
        "change map zoom level or center", "change chart type from bar to line",
        "change chart type from scatter to line",
        "convert to a pie chart", "change to a box plot", "create a histogram", "generate a heatmap",
        "make the plot 3D", "group data and create subplots", "calculate and plot a moving average",
        "set y-axis title", "invert the y-axis", "toggle grid lines on/off", "change grid line color",
        "rotate axis tick labels", "show axis lines",
        "hide the legend", "add a border to the legend", "change legend font size",
        "reverse legend item order", "set color bar title",
        "add a horizontal line for the average", "add a vertical line at a specific date/value",
        "highlight a region with a rectangle", "add error bars to the data points",
        "customize the hover text format", "add a range slider for the x-axis",
        "add a dropdown menu to filter data", "disable the mode bar"
    ]
    gen_constant_pack['system_prompt_for_desc'] = (
        "You are an expert data visualization assistant. "
        "Your primary task is to generate clear, actionable descriptions for creating or editing data visualizations in Python using Plotly. "
        "You must carefully analyze the user's input to determine if it's already a visualization task or something else."
    )
    gen_constant_pack['user_prompt_for_desc_template'] = """
You will be given an 'Original Description' and 'Editing Directions'. Your task is to generate a new visualization instruction.

Here is your decision logic:
1.  If the 'Original Description' is about data visualization (charts, plots, maps), create an instruction to **modify** the visualization using the 'Editing Directions'.
2.  If the 'Original Description' is NOT about data visualization, **ignore** the 'Editing Directions' and create a **brand new** visualization instruction based on the core topic of the original description.

**--- STRICT OUTPUT FORMAT ---**
Your final response MUST adhere strictly to the following template.
- Your response MUST begin directly with `Objective:`.
- Do NOT include any other text, explanations, reasoning, or conversational phrases like "Sure, here is the new instruction" or "Since the description is about...".
- Your entire output must ONLY be the text matching the template below.

**TEMPLATE:**
Objective: [A single, concise sentence summarizing the final goal of the visualization.]
Requirements:
- [Requirement 1 as a clear, imperative statement.]
- [Requirement 2 as a clear, imperative statement.]
- [... and so on for all requirements.]

---
**Provided Input:**

**Original Description:**
{old_desc}

**Editing Directions:**
{directions_str}
---
""".strip()
    gen_constant_pack['system_prompt_for_judge'] = "You are a professional AI visual editing evaluation expert. Your mission is to rigorously assess the quality of an image edit based on a user's instruction."
    gen_constant_pack['user_prompt_for_judge_template'] = """
You will be given a triplet of information:
1.  The `Initial Image` (before the edit).
2.  The `Edit Instruction` (a natural language command).
3.  The `Edited Image` (the result after applying the instruction).

Your evaluation must follow a strict, three-step process to determine a final binary outcome.

---
### **Evaluation Framework**

**Step 1: Comprehensive Analysis**
* **Analyze the Initial Image & Instruction:** First, understand the content of the `Initial Image` and deconstruct the `Edit Instruction` to identify the user's core intent. What object needs to be changed, added, or removed? What style or attribute needs to be modified?
* **Analyze the Edited Image:** Carefully compare the `Edited Image` with the `Initial Image`. Identify all changes that were made and assess their fidelity to the instruction.

**Step 2: Dimensional Scoring (Internal Thought Process)**
As part of your reasoning, you will mentally score the edit across three critical dimensions on a 1-5 scale. This scoring is part of your thought process to reach the final judgment.

#### **Evaluation Dimensions (1-5 Scale)**

1.  **Instruction Adherence:** How well did the edit follow the user's command?
    * **5 (Perfect):** The instruction was followed perfectly, including all nuances.
    * **4 (Good):** The main goal of the instruction was achieved, but with minor deviations (e.g., "make the car red" results in a slightly orange car).
    * **3 (Fair):** The instruction was only partially followed (e.g., "remove the two people" only removes one).
    * **2 (Poor):** The edit attempts the instruction but fundamentally misunderstands or fails to execute it.
    * **1 (Failed):** The edit completely ignores or acts contrary to the instruction.

2.  **Edit Quality & Realism:** How high is the technical and artistic quality of the edited portion?
    * **5 (Excellent):** The edit is seamless, photorealistic, and indistinguishable from a real photograph. No artifacts.
    * **4 (Good):** The edit is high quality but has very minor, barely noticeable artifacts or imperfections.
    * **3 (Fair):** The edit is noticeable. There are visible artifacts, unnatural textures, or slight inconsistencies in lighting/shadows.
    * **2 (Poor):** The edit is of low quality, looking obviously fake or "pasted on." Contains significant, distracting artifacts.
    * **1 (Failed):** The edited area is a chaotic mess of pixels, completely broken, or nonsensical.

3.  **Preservation of Unrelated Areas:** How well were the parts of the image *not* meant to be edited preserved?
    * **5 (Excellent):** Only the targeted area was modified. The rest of the image is completely untouched and pristine.
    * **4 (Good):** The edit mostly contained itself to the target area but caused tiny, insignificant changes elsewhere.
    * **3 (Fair):** The edit bled into other areas, causing noticeable but not catastrophic changes to the background or other objects.
    * **2 (Poor):** The edit significantly damaged or altered other important parts of the image.
    * **1 (Failed):** The entire image is distorted or corrupted as a result of the edit.

---
### **Step 3: Final Judgment (Fail/Success)**

Based on your dimensional scores, you will make a final binary judgment.

* **Rule:** The final result is `1` (Success) **if and only if all three dimensional scores are 3 or higher**. If *any* dimension scores 1 or 2, the final result must be `0` (Fail). This ensures that any significant flaw constitutes a failure.

---
### **Output Specification**

Your final output must be a single JSON object containing your detailed `Chain of Thought` and the final binary `Final Result`. You should also include your internal scores for transparency.

**Illustrative Example:**

* **Initial Image:** [A photo of a brown dog sitting on green grass next to a white fence.]
* **Edit Instruction:** "Change the grass to snow."
* **Edited Image:** [The grass is now white, but the dog's paws are blurry and partially erased, and a patch of snow incorrectly covers part of the white fence.]

**Output:**
```json
{{
  \"Chain of Thought\": "1. **Analysis:** The user wants to replace the 'green grass' with 'snow' while keeping the dog and fence intact. 2. **Dimensional Scoring:** a) **Instruction Adherence:** The grass was indeed changed to snow, so the main instruction was followed. Score: 4. b) **Edit Quality & Realism:** The edit on the dog's paws is poor, with noticeable blurring and erasure. This makes the edit look fake. Score: 2. c) **Preservation of Unrelated Areas:** The edit incorrectly spilled onto the white fence, altering an object that should have been preserved. Score: 2. 3. **Final Judgment:** Since two dimensions scored below 3, the edit is a failure.",
  \"Instruction Adherence Score\": 4,
  \"Edit Quality & Realism Score\": 2,
  \"Preservation of Unrelated Areas Score\": 2,
  \"Final Result\": 0
}}
```

---
The Initial image and Edited image are given at the beginning.
Edit Instruction is: {0}
""".strip()

    with open(source_fp, "r") as f_in:
        all_samples = [json.loads(l) for l in f_in]

    total = min(len(all_samples), process_end_idx)
    all_samples = all_samples[process_start_idx: total]

    rng_seed = int(time.time())

    multiprocessing.set_start_method("spawn")

    worker = partial(process_sample, _rng_seed=rng_seed, _gen_url=gen_url, _gen_key=gen_key, _gen_model=gen_model, _judge_url=judge_url, _judge_key=judge_key, _judge_model=judge_model, _api_timeout=api_timeout, _exe_timeout=execution_timeout, _gen_const_pack=gen_constant_pack)

    pass_cnt = 0
    fail_cnt = 0

    with open(dump_fp, "a") as outfile, Pool(processes=concurrency, maxtasksperchild=200) as pool:
        with tqdm(total=total, desc="Processing ") as pbar:
            for new_item in pool.imap_unordered(worker, all_samples):
                json.dump(new_item, outfile, ensure_ascii=False)
                outfile.write("\n")
                if new_item['new_pass_judge']:
                    pass_cnt += 1
                else:
                    fail_cnt += 1

                pbar.set_postfix({"Passed": pass_cnt, "Failed": fail_cnt})
                pbar.update(1)

    print(f"Total: {total}, Passed: {pass_cnt}, Failed: {fail_cnt}")