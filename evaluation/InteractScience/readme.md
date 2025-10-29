This guide outlines the step-by-step process for setting up the environment, running model inference, and evaluating the results for Artifactsbench.

## 🚀 1. Clone Repository and Environment Setup

First, clone the original repository and enter the project directory:

```
git clone https://github.com/open-compass/InteractScience.git
cd InteractScience
```

Then install the dependencies:

```
# Install project dependencies
npm install

# Install Playwright browsers
npx playwright install
```

## ⚙️ 2. Model Inference

Use the `run_generation.sh` script to perform model inference:

```
# Edit the script to configure model path and parameters
vim run_generation.sh

# Run inference (requires model path configuration)
bash run_generation.sh
```

-  Starts the vLLM API server

-  Calls `test_llm.py` for model inference

-  Saves results to the `eval/` directory

## 🧩 3. Automated Testing

Use the `run_benchmark.sh` script for the full automated testing pipeline:

```
# Set the model name to be tested
export MODEL="your_model_name"

# Run the benchmark
bash run_benchmark.sh
```

### Testing workflow

1. Extract HTML code from inference results (`extract_and_save_code.py`)
2. Run Program Functionality Testing (PFT) using `playwright_PFT.config.js`
3. Run Visual Quality Testing (VQT) using `playwright_VQT.config.js`
4. Compute CLIP similarity scores (`clip_score.py`)
5. Save all results to the `results/` directory

## 🧠 4. VLM-as-Judge Evaluation

Use `run_vlm_as_judge.sh` for automatic scoring with a vision-language model:

```
# Edit model and path configurations in the script
vim run_vlm_as_judge.sh

# Run VLM-based evaluation
bash run_vlm_as_judge.sh
```

Evaluation description:

-  Uses a vision-language model to score generated results

-  Compares generated screenshots with reference screenshots

-  Evaluates based on predefined checklists

## 📊 5. Results Analysis

Finally, calculate the overall metrics and scores using the following scripts:

```
python cal_metrics.py
python cal_vlm_as_judege_score.py
```

All final evaluation results will be saved in the `results/` directory.



