## Environment Installation
```
pip install pytest-playwright
```

## 数据源
Down load the training set of WebCode2M from [here](https://webcode2m.github.io)

## Before Running
Configure `API_URL`, `API_KEY`, `MODEL_ID`, `VLM_URL`, `VLM_KEY`, `VLM_ID` in `utils.py`.

## Data Systhesis for HTML Generation
Run `python rollout_generation.py --input_path <dir of train dataset> --output_path <save dir>`

## Data Systhesis for HTML Edit
Run `python rollout_edit.py --input_path <dir of train dataset> --output_path <save dir>`