import os
import re
import httpx
import base64

from openai import OpenAI
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import *


from datasets import load_dataset
from playwright.sync_api import sync_playwright, Browser


"""
gpt-4o
"""
GPT_CLIENT = OpenAI(
    base_url='',
    api_key='',
    max_retries=3,
)

def gpt4o_api(
    messages: List[Dict], 
    max_tokens: int = 20000,
    temperature: float = 0.7
) -> str:
    response = GPT_CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        timeout=None
    )

    return response.choices[0].message.content


API_URL = ""
API_KEY = "EMPTY"
MODEL_ID = ""
API_CLIENT = OpenAI(
    base_url=API_URL, 
    api_key=API_KEY, 
    http_client=httpx.Client(verify=False)
)

def chat_api(
    messages: List[Dict], 
    max_tokens: int = 30000,
    temperature: float = 1.0,
) -> str:
    response = API_CLIENT.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        timeout=None
    )

    return response.choices[0].message.content


GEMINI_CLIENT = OpenAI(
    base_url="",
    api_key="",
)
def gemini_api(
    messages: List[Dict], 
    max_tokens: int = 2000,
    temperature: float = 1.0,
):
    response = GEMINI_CLIENT.chat.completions.create(
        model="gemini-2.5-pro",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        timeout=None
    )

    return response.choices[0].message.content


VLM_URL = ""
VLM_KEY = "EMPTY"
VLM_ID = ""
VLM_CLIENT = OpenAI(
    base_url=VLM_URL,
    api_key=VLM_KEY,
    http_client=httpx.Client(verify=False)
)

def chat_vlm(
    messages: List[Dict], 
    max_tokens: int = 20000,
    temperature: float = 0.7
) -> str:
    response = VLM_CLIENT.chat.completions.create(
        model=VLM_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        timeout=None
    )

    return response.choices[0].message.content

def convert_image_to_url(image_path: str, image_type: str="png") -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        image_url = f"data:image/{image_type};base64,{img_b64}"
    return image_url


def print_msgs(messages: List) -> None:
    for msg in messages:
        print("="*40 + f" {msg['role']} " + "="*40)
        print(msg["content"])
        print("="*80)
        print("\n\n")


def load_batches(
    data_files: Union[str|List], 
    sample_num: int, 
    batch_size: int
):
    dataset = load_dataset("parquet", data_files=data_files)["train"]
    dataset = dataset.select(range(sample_num))

    batches = []
    for i in range(0, sample_num, batch_size):
        batches.append({
            "dataset": dataset.select(range(i, min(i+batch_size, sample_num))), "offset": i
        })

    return batches


def screenshot( html: str, save_path: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.set_content(html, wait_until="domcontentloaded")

            page.wait_for_timeout(2000) 

            page.screenshot(path=save_path, full_page=True)

            print(f"Screenshot saved: {save_path}")
        except Exception as e:
            print(f"Error in screenshot (utils.py): {e}")
            raise e
        finally:
            page.close()
            context.close()
            browser.close()

def is_icon_image(img_tag):
    src = img_tag.get("src", "").lower()
    classes = " ".join(img_tag.get("class", [])).lower()
    keywords = ["icon", "logo", "avatar", "facebook", "twitter", "instagram", "linkedin", "social"]
    return any(k in src for k in keywords) or any(k in classes for k in keywords)

def get_placeholder_size(img_tag):
    width = img_tag.get("width")
    height = img_tag.get("height")
    
    if width and height:
        return width, height

    if is_icon_image(img_tag):
        return "32", "32"

    return "50", "40"

def make_html_offline(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all("div"):
        if "src" in tag.attrs:
            del tag.attrs["src"]

    HIDDEN_PATTERNS = ["display:none", "visibility:hidden", "opacity:0", "width:0", "height:0"]
    tags_to_remove = []
    for tag in soup.find_all(style=True):
        style = tag.get("style", "")
        style_clean = style.replace(" ", "").lower().replace("\\9", "")
        if any(p in style_clean for p in HIDDEN_PATTERNS):
            tags_to_remove.append(tag)

    for tag in tags_to_remove:
        try:
            tag.decompose()
        except Exception as e:
            print(f"cannot delete the tag: {e}")
        
    for tag in soup.find_all([
        "script", "noscript", "audio", "embed", "image", 
        "amp-img", "amp-audio", "amp-video", "amp-iframe", "amp-pixel", "amp-install-serviceworker", "amp-anim", 
        "mip-img", "mip-video", "mip-iframe"
        ]):
        tag.decompose()
    
    img_counter = 0
    for img_tag in soup.find_all(["img", "input"]):
        src = img_tag.get("src", "")
        if not src:
            continue

        img_counter += 1
        label = f"Image{img_counter}"
        width, height = get_placeholder_size(img_tag)

        img_tag["src"] = f"https://placehold.co/{width}x{height}/a7adb2/ffffff?text={label}"
        img_tag["alt"] = "placeholder"

    media_counter = 0
    for media_tag in soup.find_all(["video", "iframe"]):
        media_counter += 1
        label = f"Video{media_counter}"

        width, height = get_placeholder_size(media_tag)

        media_tag["src"] = f"https://placehold.co/{width}x{height}/a7adb2/ffffff?text={label}"
        

    return str(soup)
