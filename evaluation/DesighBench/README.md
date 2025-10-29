# DesignBench Evaluation

## Installation and Dataset
Please refer to [DesignBench](https://webpai.github.io/DesignBench/) for more details.

## Model Deployment
1. For models from Qwen family, change the configurations in the first three lines of `scripts/vllm_qwen.sh`. For models from Intern Family, use `scripts/vllm_intern.sh` instead.
2. Run the deployment script, e.g., `bash scripts/vllm_qwen.sh` or `bash scripts/vllm_intern.sh`.

### Evaluation
1. Configure the url and api key of GPT-4o at the beginning of `code/evaluator/metric.py`.
2. Configure `scripts/eval.sh`
3. Run the evaluation script: `bash scripts/eval.sh`.