import os
import json
import shutil
import tqdm
import numpy as np
from typing import Tuple, List, Union, Dict
from pprint import pprint
import random



def remove_special_tokens(text: str) -> str:
    return text.replace("<image>", "`image`").replace("<video>", "`video`").replace("<audio>", "`audio`")

def get_generation_prompt() -> Tuple[str, str]:
    system_prompt = """You are an expert HTML/CSS developer.
You take screenshots of a reference web page from the user, and then build single page apps using HTML/CSS.

- Make sure the app looks exactly like the screenshot.
- Pay close attention to background color, text color, font size, font family, padding, margin, border, etc. Match the colors and sizes exactly.
- Use the exact text from the screenshot.
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.
- Repeat elements as needed to match the screenshot. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like "<!-- Repeat for each news item -->" or bad things will happen.
- For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an image generation AI can generate the image later.

Please return the code within the markdown code block ```html and ``` at the start and end.
Do not output any extra information or comments.
"""
    prompt = """
The screenshot:
<image>
"""
    return system_prompt.strip(), prompt.strip()

def get_edit_prompt(mode: str, instruction: str, code: str) -> Tuple[str, str]:
    system_prompt = """You are an expert HTML/CSS developer.
{provide_items}
You need to modify the code according to the user's instruction to make the webpage satisfy user's demands.

Requirements:
- Do not modify any part of the web page other than the parts covered by the instructions.
- For images, use placeholder images from https://placehold.co
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.

You MUST wrap your entire code output inside the following markdown fences: ```html and ```.

Do not output any extra information or comments.

"""

    if mode == "BOTH":
        provide_items = "You take a screenshot, a piece of code of a reference web page, and an instruction from the user."
        prompt = f"Instruction:\n{instruction}\n\nCode:\n{code}\n\nThe webpage screenshot:\n<image>"
    elif mode == "IMAGE":
        provide_items = "You take a screenshot, and an instruction from the user."
        prompt = f"Instruction:\n{instruction}\n\nThe webpage screenshot:\n<image>"
    elif mode == "CODE":
        provide_items = "You take a piece of code of a reference web page, and an instruction from the user."
        prompt = f"Instruction:\n{instruction}\n\nCode:\n{code}"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    system_prompt = system_prompt.format(provide_items=provide_items)
    user_prompt = prompt
        
    return system_prompt.strip(), user_prompt.strip()

def get_repair_prompt(mode: str, code: str) -> Tuple[str, str]:
    system_prompt = """You are an expert HTML/CSS developer. You are proficient in UI repair.
You take a screenshot, a piece of code of a reference web page with design issues.
You need to repair the UI display issues.

Here are the issue types and explanations:

- occlusion: Elements are hidden or partially covered by other elements, making content inaccessible or invisible to users. This includes overlapping components, modal dialogs blocking content, or elements positioned behind others.
- crowding: Too many elements are packed into a small space without adequate spacing, making the interface feel cluttered and difficult to navigate. This affects readability and user experience.
- text overlap: Text content overlaps with other text or UI elements, making it unreadable or causing visual confusion. This often occurs due to improper positioning, z-index issues, or responsive design problems.
- alignment: Elements are not properly aligned with each other or the overall layout grid, creating a disorganized and unprofessional appearance. This includes misaligned text, buttons, images, or containers.
- color and contrast: Poor color choices that affect readability or accessibility, including insufficient contrast between text and background, or color combinations that are difficult for users with visual impairments to distinguish.
- overflow: Content extends beyond its intended container boundaries, causing horizontal scrollbars, cut-off text, or elements appearing outside their designated areas.

Requirements:
- Do not modify the code except for the part with display issues.
- For images, use placeholder images from https://placehold.co
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.


Output Format:

Please provide the following information and output them using the tags: [ISSUES] ... [/ISSUES] , [REASONING] ... [/REASONING], and [CODE] ... [/CODE].

1. Display issues: occlusion/crowding/text overlap/alignment/color and contrast/overflow, you should output all the issues in a list [].

2. Reasoning:
   - Explain your rationale about the display issues.
   - Describe specific elements that involving design issues

3. Repaired Code: The complete fixed code

You MUST wrap your entire code output inside the following markdown fences: ```html and ```.

Please follow the format of the example response below:

[ISSUES]
["text overlap"]
[/ISSUES]

[REASONING]
The main heading text overlaps with the navigation menu due to absolute positioning without proper z-index management. The h1 element with 'position: absolute; top: 16px; left: 16px' is positioned behind the navigation bar, making the title partially unreadable. Additionally, the navigation items are too close together without proper spacing, causing readability issues on smaller screens.
[/REASONING]

[CODE]
```html
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Header Component</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        .header {{
            position: relative;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        .nav {{
            background-color: #2563eb;
            color: white;
            padding: 12px 16px;
            position: relative;
            z-index: 10;
        }}
        
        .nav ul {{
            display: flex;
            list-style: none;
            gap: 24px;
        }}
        
        .nav a {{
            color: white;
            text-decoration: none;
        }}
        
        .nav a:hover {{
            text-decoration: underline;
        }}
        
        .title-section {{
            padding: 32px 16px;
        }}
        
        .title {{
            font-size: 1.875rem;
            font-weight: bold;
            color: #1f2937;
            position: relative;
            z-index: 0;
        }}
    </style>
</head>
<body>
    <header class=\"header\">
        <nav class=\"nav\">
            <ul>
                <li><a href=\"#\">Home</a></li>
                <li><a href=\"#\">About</a></li>
                <li><a href=\"#\">Services</a></li>
                <li><a href=\"#\">Contact</a></li>
            </ul>
        </nav>
        <div class=\"title-section\">
            <h1 class=\"title\">Welcome to Our Website</h1>
        </div>
    </header>
</body>
</html>
```
[/CODE]

Do not output any extra information or comments.

"""
    
    code_message = f"The code is:\n{code}."
    image_message = "The screenshot:\n<image>"
    
    if mode == "CODE":
        prompt = code_message
    elif mode == "IMAGE":
        prompt = image_message
    elif mode in ["BOTH", "MARK"]:
        prompt = f"{code_message}\n\n{image_message}"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return system_prompt.strip(), prompt.strip()


def all_files_exist(file_path_list: List[str]):
    for file_path in file_path_list:
        if os.path.exists(file_path):
            continue
        else:
            return False
    return True

def is_all_target(values: List, target: int) -> bool:
    for value in values:
        if value != target:
            return False
    return True

def save_files(data: List, save_path: str):
    data_num = round(len(data) / 1000)
    save_path += f"_{data_num}K"
    if len(data) <= 50000:
        json.dump(data, open(f"{save_path}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    else:
        for idx, start in enumerate(range(0, len(data), 50000)):
            json.dump(data[start:start+50000], open(f"{save_path}_{idx}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def make_generation_data(input_dir: str, sample_num: int=20, init=False):
    output_data = []
    data_cnt = 0
    for number in tqdm.tqdm(os.listdir(input_dir)):
        if init:
            code_path = os.path.join(input_dir, number, "code_init.html")
            image_path = os.path.join(input_dir, number, "screenshot_init.png")
        else:
            code_path = os.path.join(input_dir, number, "code.html")
            image_path = os.path.join(input_dir, number, "screenshot.png")
        if not all_files_exist([code_path, image_path]):
            continue

        with open(code_path, "r") as f:
            code = f.read()
        code = remove_special_tokens(code)
        system_prompt, user_prompt = get_generation_prompt()
        response = f"```html\n{code}\n```"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]

        images = [image_path]
        output_data.append({"messages": messages, "images": images})
        
        data_cnt += 1
        if sample_num != -1 and sample_num == data_cnt:
            break

    return output_data

def make_edit_data(input_dir: str, sample_num: int=20):
    output_data = []
    data_cnt = 0
    bad_cnt = 0
    for number in tqdm.tqdm(os.listdir(input_dir)):
        vanilla_code_path = os.path.join(input_dir, number, "0.html")
        vanilla_image_path = os.path.join(input_dir, number, "0.png")
        new_code_path = os.path.join(input_dir, number, "1.html")
        new_image_path = os.path.join(input_dir, number, "1.png")
        instruct_path = os.path.join(input_dir, number, "instruction.json")
        if not all_files_exist([vanilla_code_path, vanilla_image_path, new_code_path, new_image_path, instruct_path]):
            continue
        
        try:
            instruct = json.load(open(instruct_path, "r"))
        except:
            continue
        
        # judgement = instruct["judgement"]
        # values = [v for k, v in judgement.items() if k != "Chain of Thought"]
        # if not is_all_target(values, 5):
        #     bad_cnt += 1
        #     continue

        with open(vanilla_code_path, "r") as f:
            code_0 = f.read()
        with open(new_code_path, "r") as f:
            code_1 = f.read()
        
        instruction = remove_special_tokens(instruct["human"])
        code_0 = remove_special_tokens(code_0)
        code_1 = remove_special_tokens(code_1)
        system_prompt, user_prompt = get_edit_prompt("BOTH", instruction, code_0)
        response = f"```html\n{code_1}\n```"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
        images = [vanilla_image_path]
        output_data.append({"messages": messages, "images": images})

        data_cnt += 1
        if sample_num != -1 and sample_num == data_cnt:
            break
        
    print(f"[edit] data number: {data_cnt}; bad rate: {bad_cnt/(data_cnt+bad_cnt):.3f}")
    return output_data

def make_repair_data(input_dir: str, sample_num: int=20):
    output_data = []
    data_cnt = 0
    bad_cnt = 0
    for number in tqdm.tqdm(os.listdir(input_dir)):
        vanilla_code_path = os.path.join(input_dir, number, "0.html")
        vanilla_image_path = os.path.join(input_dir, number, "0.png")
        new_code_path = os.path.join(input_dir, number, "1.html")
        new_image_path = os.path.join(input_dir, number, "1.png")
        instruct_path = os.path.join(input_dir, number, "instruction.json")
        if not all_files_exist([vanilla_code_path, vanilla_image_path, new_code_path, new_image_path, instruct_path]):
            continue
        data_cnt += 1

        instruct = json.load(open(instruct_path, "r"))
        judgement = instruct["judgement"]
        values = [v for k, v in judgement.items() if k != "Chain of Thought"]
        if not is_all_target(values, 5):
            bad_cnt += 1
            continue

        with open(vanilla_code_path, "r") as f:
            code_0 = f.read()
        with open(new_code_path, "r") as f:
            code_1 = f.read()
        
        code_0 = remove_special_tokens(code_0)
        code_1 = remove_special_tokens(code_1)
        reasoning = remove_special_tokens(instruct["reasoning"])
        issues = remove_special_tokens(json.dumps(instruct["issues"], ensure_ascii=False))

        system_prompt, user_prompt = get_repair_prompt("BOTH", code_1)     
        response = f"""[ISSUES]\n{issues}\n[/ISSUES]\n\n[REASONING]\n{reasoning}\n[/REASONING]\n\n[CODE]\n```html\n{code_0}\n```\n[/CODE]"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]

        images = [new_image_path]
        output_data.append({"messages": messages, "images": images})
 
        if sample_num != -1 and sample_num == data_cnt:
            break

    print(f"[repair] data number: {data_cnt}; bad rate: {bad_cnt/data_cnt:.3f}")
    return output_data



with open("/cpfs01/shared/XNLP_H800/liuyang/data_generation/output_20250827/edit_96K.jsonl", "r", encoding="utf-8") as f:
    items = f.readlines()
    items = random.sample(items, 69501)

with open("/cpfs01/shared/XNLP_H800/liuyang/data_generation/output_20250827/edit_random_70K.jsonl", "w", encoding="utf-8") as f:
    for line in items:
        f.write(line)


# generation_dir = "/cpfs01/shared/XNLP_H800/liuyang/data_generation/output_generation"
# generation_data = make_generation_data(generation_dir, sample_num=1000000, init=True)
# with open("output_20250825/generation_vanilla_400K.json", "w") as f:
#     json.dump(generation_data, f, indent=4, ensure_ascii=False)
# print(f"[generation] data cnt: {len(generation_data)}")


# edit_dir = "/cpfs01/shared/XNLP_H800/liuyang/data_generation/output_20250827/edit"
# edit_data = make_edit_data(edit_dir, sample_num=210000)
# save_files(edit_data, "output_20250827/edit_100K")

# repair_dir = "/cpfs01/shared/XNLP_H800/liuyang/data_generation/output_20250825/repair"
# repair_data = make_repair_data(repair_dir, sample_num=30000)
# with open("output_20250825/repair.json", "w") as f:
#     json.dump(repair_data, f, indent=4, ensure_ascii=False)
