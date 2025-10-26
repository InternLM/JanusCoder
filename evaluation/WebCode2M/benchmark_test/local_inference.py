import os
import io
import time
import base64
import hashlib
import argparse
from tqdm import tqdm
from openai import OpenAI
from datasets import Dataset, load_dataset
from typing import Union, List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

def image2md5(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_data = image_bytes.getvalue()
    md5_hash = hashlib.md5(image_data)
    md5_hex = md5_hash.hexdigest()
    return str(md5_hex)

def save_result(result_path, image, answer, prediction, duration):
    md5 = image2md5(image)
    os.makedirs(os.path.join(result_path,f'{md5}'), exist_ok=True)
    image.save(os.path.join(result_path,f'{md5}/image.png'))

    with open(os.path.join(result_path,f'{md5}/answer.html'),'w') as f:
        f.write(answer)
    with open(os.path.join(result_path,f'{md5}/prediction.html'),'w') as f:
        f.write(prediction)
    with open(os.path.join(result_path,f'{md5}/time.csv'), 'a+') as f:
        f.write(f'{duration}\n')


class VLMChat:    
    def __init__(self, model_name: str, url: str) -> None:
        self.model_name = model_name
        self.max_tokens = 8192 * 2
        self.temperature = 0.7
        self.top_p = 0.95
        self.seed = 42
        print(f"Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Seed: {self.seed}")
        self.client =OpenAI(api_key="EMPTY", base_url=url)
        
    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def chat(self, system_prompt: str, user_prompt: str, image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        encoded_image = base64.b64encode(buffered.getvalue()).decode("ascii")
        # encoded_image = self.encode_image(image)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        # print("call API")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            timeout=None,
        )
        # print("finished")
        return response.choices[0].message.content.strip()


prompt_system = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps 
using Tailwind, HTML and JS.

- Make sure the app looks exactly like the screenshot.
- Make sure the app has the same page layout like the screenshot, i.e., the gereated html elements should be at the same place with the correspondingpart in the screenshot and the generated  html containers should have the same hierachy structure as the screenshot.
- Pay close attention to background color, text color, font size, font family, 
padding, margin, border, etc. Match the colors and sizes exactly.
- Use the exact text from the screenshot.
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writingthe full code. WRITE THE FULL CODE.
- Repeat elements as needed to match the screenshot. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like"<!-- Repeat for each news item -->" or bad things will happen.
- For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an imagegeneration AI can generate the image later.

In terms of libraries,

- Use this script to include Tailwind: <script src="https://cdn.tailwindcss.com"></script>
- You can use Google Fonts
- Font Awesome for icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

Return only the full code in <html></html> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

prompt_user = "Turn this into a single html file using tailwind."


parser = argparse.ArgumentParser(description='Process two path strings.')
parser.add_argument('--model_name', type=str, default="final0_internvl3.5-4b_0820")
parser.add_argument('--url', type=str, default="")
parser.add_argument('--workers', type=int, default=20)
args = parser.parse_args()

model = VLMChat(model_name=args.model_name, url=args.url)
for data_name in ["short", "mid", "long"]:
    # TODO: Replace with your data path
    data_path = f"WebCode2M_test/{data_name}.parquet"
    result_path = f"results/{args.model_name}_{data_name}"
    data_num = 256

    print(data_name)
    ds = load_dataset('parquet', data_files=data_path)['train']

    def worker(item):
        image = item['image']
        md5 = image2md5(image)
        if os.path.exists(os.path.join(result_path, md5)) and len(os.listdir(os.path.join(result_path, md5))) == 4:
            return

        t_start = time.time()
        html = model.chat(prompt_system, prompt_user, image)

        duration = time.time() - t_start
        save_result(result_path, image, item.get('text') or item.get('html'), html, duration)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, item) for item in ds.select(range(0, data_num))]
        for f in tqdm(as_completed(futures), total=len(futures)):
            try:
                f.result()
            except Exception as e:
                print(f"Error: {e}")
