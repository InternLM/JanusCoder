# PandasPlotBench

Original Repository: [PandasPlotBench](https://github.com/JetBrains-Research/PandasPlotBench)

## Usage

1. Install `poetry` in a python environment: `pip install poetry`
2. Navigate to `plotting_benchmark` directory: `cd plotting_benchmark`
3. Run the following command to install dependencies: `poetry install --extras "local_gpu"`
4. Edit the config file `config/config.yaml` and set up the model to be evaluated
5. Set up OpenAI key in the environment: `export OPENAI_KEY=[YOUR KEY]`
6. Start the evaluation: `poetry run python run_benchmark.py`
7. The results will be presented in `out_results/` directory
