import pandas as pd
from datasets import load_dataset


def analyze_task_length(dataset: pd.DataFrame):
    dataset["task_len"] = dataset["task__plot_description"].apply(len)
    dataset["task_len"] = dataset["task_len"] - len(
        "Plot Description: "
    )  # Substracting, heading
    dataset["style_len"] = dataset["task__plot_style"].apply(len)
    dataset["style_len"] = dataset["style_len"] - len(
        "Plot Style Description: "
    )  # Substracting, heading
    dataset["task_total_len"] = dataset["task_len"] + dataset["style_len"]

    dataset["task_short_len"] = dataset["_task__plot_description_short"].apply(len)
    dataset["task_short_len"] = dataset["task_short_len"] - len(
        "Plot Description: "
    )  # Substracting, heading

    task_mean = dataset["task_len"].mean()
    task_std = dataset["task_len"].std()
    style_mean = dataset["style_len"].mean()
    style_std = dataset["style_len"].std()
    task_tot_mean = dataset["task_total_len"].mean()
    task_tot_std = dataset["task_total_len"].std()
    task_short_mean = dataset["task_short_len"].mean()
    task_short_std = dataset["task_short_len"].std()

    print(f"Tasks main length = {task_mean:.0f} ± {task_std:.0f} symbols")
    print(f"Style length = {style_mean:.0f} ± {style_std:.0f} symbols")
    print(f"Tasks total length = {task_tot_mean:.0f} ± {task_tot_std:.0f} symbols")
    print(f"Tasks short length = {task_short_mean:.0f} ± {task_short_std:.0f} symbols")


if __name__ == "__main__":
    dataset = load_dataset("JetBrains-Research/PandasPlotBench", split="test")
    dataset_df = dataset.to_pandas()

    analyze_task_length(dataset_df)
