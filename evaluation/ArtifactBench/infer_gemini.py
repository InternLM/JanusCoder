import argparse
import os
import json
import uuid
import time
import base64
import requests
import multiprocessing
from functools import partial
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union
from extract_ans import extract_mllm_overall
from prompts.prompt_mllm_check import get_prompt_mllm_checklist
from transformers import AutoTokenizer
from utils import *


def get_args() -> argparse.Namespace:
    """
    Parses command line arguments for input file, output file, screenshot directory,
    count of screenshots, API key, model marker, tokenizer directory, and number of processes.

    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="This program processes data and generates screenshots."
    )

    # Command-line arguments
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("save_path", help="Path to save the processed output JSON.")
    parser.add_argument("screenshots_dir", help="Directory to store the screenshots.")
    parser.add_argument("count", type=int, help="Number of screenshots to capture.")
    parser.add_argument("api_key", help="API key for making requests.")
    parser.add_argument("model_marker", help="Model marker for identifying the model.")
    parser.add_argument("api_url", help="The URL endpoint for making the POST request to the server.")
    parser.add_argument("tokenizer_dir", help="Directory for the tokenizer.")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes to use (default is 4).",
    )

    return parser.parse_args()


def make_request(
    messages: Union[str, List[Dict[str, Any]]],
    api_key: str,
    model_marker: str,
    api_url: str,
    session_id: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 0.5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Makes a request to the specified model with the provided messages and other parameters.
    Retries the request in case of failure.

    Parameters:
    - messages: The message or list of message objects to send to the model.
    - api_key: The API key to authenticate the request.
    - model_marker: The model marker used to identify the model.
    - api_url: The URL endpoint for making the POST request to the server.
    - session_id: Optional session identifier for the request.
    - max_retries: Maximum number of retry attempts (default is 3).
    - retry_delay: Delay between retries (default is 0.5 seconds).

    Returns:
    - dict: The response from the model or an empty string if all retries fail.
    """
    request_id = uuid.uuid4().hex[:24]
    session_id = session_id or uuid.uuid4().hex[:24]

    if isinstance(messages, str):
        messages = [{"role": "user", "content": [{"type": "text", "value": messages}]}]

    payload = {
        "messages": messages,
        "model": model_marker,
    }

    # Try the request with retries if an error occurs
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                timeout=30000,
            )
            response.raise_for_status()

            response_json = response.json()
            value = response_json.get("choices", [{}])[0].get("message", "").get("content", "")
            if value:
                return value
            else:
                raise ValueError("The return value is empty!")

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries:
                delay = retry_delay * (2**attempt)
                time.sleep(min(delay, 5))
                continue

    print("All reattempts failed!")
    return ""


def build_data(query, answer, checklist, tokenizer):
    """
    Builds the data to be sent to the model by formatting the query, answer, and checklist
    according to the required prompt structure.

    Parameters:
    - query: The query to be asked to the model.
    - answer: The answer to the query.
    - checklist: Optional checklist to guide the model's response.
    - tokenizer: The tokenizer used to process the data.

    Returns:
    - List[Dict]: The formatted data ready for the model request.
    """
    data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": get_prompt_mllm_checklist(checklist, query, answer),
                }
            ],
        }
    ]

    # Truncate the content to fit within the model's token limit
    data[0]["content"][0]["text"] = truncate_content(
        data[0]["content"][0]["text"], tokenizer
    )

    return data


def add_images_to_data(img_path, data):
    """
    Adds images to the request data by encoding the images to base64 format.

    Parameters:
    - img_path: List of paths to the image files to be included.
    - data: The original data to which the images will be appended.

    Returns:
    - dict: The modified data with the images added.
    """
    try:
        for i in range(len(img_path)):
            with open(img_path[i], "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                new_image = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    },
                }
                data[0]["content"].append(new_image)
        return data
    except Exception as e:
        print(f"Could not add image to data: {e}")
        return data


def make_api_request(data, api_key, model_marker, api_url, index, line_data):
    """
    Makes the API request to the model with the provided data and updates the line data with the model's response.

    Parameters:
    - data: The formatted data to be sent to the model.
    - api_key: The API key to authenticate the request.
    - model_marker: The model marker used to identify the model.
    - api_url: The URL endpoint for making the POST request to the server.
    - index: The index of the current item in the data.
    - line_data: The data for the current item.

    Returns:
    - dict: The updated line data with the model's response.
    """
    try:
        for i in range(4):  # Try the request up to 4 times
            output_text = make_request(data, api_key, model_marker, api_url)
            output_text = output_text.split("</think>")[-1].strip()
            line_data["gemini_reason"] = output_text
            line_data["gemini_ans"] = extract_mllm_overall(index, output_text)
            if line_data["gemini_ans"] is not None:
                return line_data
        return line_data
    except Exception as e:
        print(f"{index} request failed: {e}")
        return line_data


def process_item(item_data, count, screenshots_dir, tokenizer, api_key, model_marker, api_url):
    """
    Processes a single item by handling the request and image processing.

    Parameters:
    - item_data: The data for the item to be processed.
    - count: The number of screenshots to capture.
    - screenshots_dir: The directory to store the screenshots.
    - tokenizer: The tokenizer used to process the data.
    - api_key: The API key for making requests.
    - model_marker: The model marker for identifying the model.
    - api_url: The URL endpoint for making the POST request to the server.

    Returns:
    - tuple: A tuple containing the index and the updated line data for the item.
    """
    try:
        _, line_data = item_data
        index = line_data["index"]
        img_path, query, ans = extract_information(
            index, line_data, count, screenshots_dir
        )
        checklist = line_data.get("checklist")
        data = build_data(query, ans, checklist, tokenizer)
        data = add_images_to_data(img_path, data)
        line_data = make_api_request(data, api_key, model_marker, api_url, index, line_data)
        return index, line_data
    except Exception as e:
        _, line_data = item_data
        index = line_data["index"]
        print(f"Mistakes in process_item (index={index}): {e}")
        return index, line_data


def main():
    """
    The main function to orchestrate the data processing and screenshot generation.

    It parses command line arguments, loads data, processes the items in parallel,
    and saves the results to the output file.
    """
    args = get_args()  # Get arguments from the command line

    start_time = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True
    )

    # Create screenshots directory
    os.makedirs(args.screenshots_dir, exist_ok=True)

    # Load data from the provided path
    all_data = load_data_from_file(args.input_file)

    print(f"A total of {len(all_data)} items to process.")

    items_to_process = [(i, item) for i, item in enumerate(all_data)]
    process_func = partial(
        process_item,
        count=args.count,
        screenshots_dir=args.screenshots_dir,
        tokenizer=tokenizer,
        api_key=args.api_key,
        model_marker=args.model_marker,
        api_url = args.api_url,
    )

    num_processes = args.num_processes
    print(f"Starting parallel processing with {num_processes} processes.")

    # Process items in parallel
    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_func, items_to_process),
            total=len(all_data),
            desc="Processing",
        ):
            results.append(result)
            index, item = result
            with open(args.save_path, "a", encoding="utf-8") as w:
                w.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save processed data to the output file
    processed_data = [item for _, item in sorted(results, key=lambda x: x[0])]
    with open(args.save_path, "w", encoding="utf-8") as w:
        for item in processed_data:
            w.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processing completed in {time.time() - start_time:.2f} seconds.")

    # Calculate average scores and lengths
    avg_mllm_score = 0
    error_count = 0
    avg_len = 0
    answer_count = 0
    for data in processed_data:
        try:
            avg_mllm_score += float(data['gemini_ans'])
            answer_str = get_answer(data)
            if 'model_infer_think' in data:
                answer_str = data['model_infer_think'] + answer_str
            avg_len += len(answer_str)
            answer_count += 1
        except:
            error_count += 1
    print(f"Average MLLM score: {avg_mllm_score/answer_count:.4f}")
    print(f"Average answer length: {avg_len/answer_count:.2f}")
    print(f"Error count: {error_count}")


if __name__ == "__main__":
    main()
