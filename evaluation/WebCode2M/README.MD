## Environment Installation and Dataset Download
Please refer to [WebCode2M](https://github.com/CGCL-codes/naturalcc/tree/main/examples/webcode2m) for more details.

## Model Deployment
Please refer to the README.md file at `evaluation/DesignBench/README.md`.

### Evaluation
1. Change the test set path at line 114 of `benchmark_test/local_inference.py`.
2. Run `python benchmark_test/local_inference.py --model_name <model_name> --url <url>`
3. Run `python scripts/evaluation/evaluate.py --model_name <model_name>`