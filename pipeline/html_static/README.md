## Environment Installation

Install the required dependencies via pip:

```bash
pip install pytest-playwright
```

---

## Data Source

The WebCode2M dataset is available at:  
👉 [https://webcode2m.github.io](https://webcode2m.github.io)

You can also download the dataset from Hugging Face using the following script:

```python
from huggingface_hub import snapshot_download, login

login(token="YOUR_HUGGINGFACE_ACCESS_TOKEN")

snapshot_download(
    repo_id="xcodemind/webcode2m",
    repo_type="dataset",
    local_dir="YOUR_LOCAL_DIRECTORY",
    local_dir_use_symlinks=False
)
```

---

## Before Running

If you plan to synthesize training data for the **HTML editing task**, please configure the following fields in `utils.py` before running the scripts.

> 💡 If you only need to synthesize data for **HTML generation**, you can skip this step.

### Required Configurations

| Variable   | Description                                                                             |
| ---------- | --------------------------------------------------------------------------------------- |
| `API_URL`  | API endpoint of the model used for webpage editing                                      |
| `API_KEY`  | API key for the editing model                                                           |
| `MODEL_ID` | Model identifier for webpage editing                                                    |
| `VLM_URL`  | API endpoint of the VLM used to generate editing instructions and evaluate data quality |
| `VLM_KEY`  | API key for the VLM                                                                     |
| `VLM_ID`   | Model identifier for the VLM                                                            |

---

## Data Synthesis for HTML Generation

Run the following command to synthesize training data for the HTML generation task:

```bash
python rollout_generation.py \
    --input_path [path to WebCode2M parquet files] \
    --output_path [path to save synthesized data]
```

### Optional Arguments

```python
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="output/edit")
parser.add_argument("--start", type=int, default=600, help="start index of parquet files")
parser.add_argument("--end", type=int, default=900, help="end index of parquet files")
parser.add_argument("--mode", type=str, default="image", help="input mode: image / both")
parser.add_argument("--use_preprocess", type=int, default=0, help="whether to preprocess webpages")
```

---

## Data Synthesis for HTML Editing

To generate training data for the HTML editing task, run:

```bash
python rollout_edit.py \
    --input_path [path to WebCode2M parquet files] \
    --output_path [path to save synthesized data] \
    --start [start index of parquet files] \
    --end [end index of parquet files] \
    --mode [image | both] \
    --use_preprocess [0 or 1]
```

### Parameter Explanation

* **`--mode`**

  * `image`: use only webpage screenshots
  * `both`: use screenshot + source code for instruction synthesis and VLM judgment

* **`--use_preprocess`**

  * `1`: preprocess webpages to remove dependencies on online resources
  * `0`: use raw webpages without preprocessing

