# 🧪 ArtifactsBench Quick Start Guide

The evaluation of artifacts is based on [Tencent-Hunyuan/ArtifactsBenchmark](https://github.com/Tencent-Hunyuan/ArtifactsBenchmark), which provides a benchmark for evaluating multimodal and interactive reasoning capabilities of models.

 The dataset file `artifacts_bench_lite.json` is the **lite subset** mentioned in the paper.

## 🚀 1. Clone the Repository and Environment Setup

First, clone the original repository and enter the project directory:

```
git clone https://github.com/Tencent-Hunyuan/ArtifactsBenchmark.git
cd ArtifactsBenchmark
```

Then install the required dependencies:

```
pip install vllm==0.8.3
pip install pytest-playwright
playwright install
playwright install-deps
pip install transformers
pip install requests
pip install tqdm
```

## 📘 2. Data Format

You can use your own model to perform inference based on the **question** field in `dataset/artifacts_bench.json` and save the corresponding output in the **answer** field.

 Each record follows this format:

```
{
    "index": "unique identifier in the dataset that corresponds one-to-one with 'question'",
    "question": "each 'question' in ArtifactsBench",
    "answer": "the answer inferred by your model based on the 'question'"
}
```

## ⚙️ 3. Evaluation with Gemini

Run the following command to evaluate your model outputs using the Gemini model:

```
api_key=xxx
model_marker=xxx
api_url=xxx
screenshots_count=3
path_with_index=xxx
save_path=xxx
screenshots_dir=xxx
tokenizer_dir=xxx
num_processes=16

python3 src/infer_gemini.py \
    $path_with_index \
    $save_path \
    $screenshots_dir \
    $screenshots_count \
    $api_key \
    $model_marker \
    $api_url \
    $tokenizer_dir \
    --num_processes $num_processes
```

## 🧩 4. Parameter Description

Parameters:

**api_key**

Your API key for accessing the Gemini model.

**model_marker**

 The specific model marker used within Gemini.

**api_url**

The API endpoint for sending POST requests.

**screenshots_count**

Number of screenshots to feed into Gemini.

**path_with_index**

Input file path containing entries with `index`, `question`, and `answer`.

**save_path**

Path to save the output results (adds `gemini_reason` and `gemini_ans` fields).

**screenshots_dir**

Directory where corresponding screenshots are stored.

**tokenizer_dir**

Path to the tokenizer model to control token length.

**num_processes**

Number of parallel processes for inference (e.g., 16).



## 📊 5. Output Description

After evaluation, each entry in the saved file will include:

-  `gemini_reason`: Gemini’s reasoning explanation

-  `gemini_ans`: The score provided by Gemini

This completes the quick start for running inference and evaluation with ArtifactsBenchmark.

 You can begin by using the lite subset `artifacts_bench_lite.json` for a faster evaluation process.
