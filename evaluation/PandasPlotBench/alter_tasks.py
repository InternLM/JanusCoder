from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from plotting_benchmark.benchmark import PlottingBenchmark, get_model

load_dotenv()


def get_task_shanging_task(heading: str, plot_task: str) -> str:
    task = f'You will be given a task to plot a dataframe. Make it shorter, producing two sentences, keeping all useful information. Reply only the task, beginning with "{heading}". Return Nothing else more. Here is a task to plot a dataframe:\n{plot_task}.'
    return task


def get_task_changing_single_task(heading: str, plot_task: str) -> str:
    task = f'You will be given a task to plot a dataframe. Make it very short, producing single sentence, keeping all useful information. Reply only the task, beginning with "{heading}". Return Nothing else more. Here is a task to plot a dataframe:\n{plot_task}.'
    return task


def get_compressing_model():
    compressing_prompt = "You are a helpful programming assistant proficient in python and matplotlib. You are an expert in compressing tasks into very informative and short task."
    model = get_model(
        model_name="openai/gpt-4o",
        model_pars={"temperature": 0.0},
        system_prompt=compressing_prompt,
    )

    return model


def alter_tasks():
    pass


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    dataset_json_path = "dataset.json"
    config = OmegaConf.load(config_path)
    paths = config.paths

    benchmark = PlottingBenchmark(config_path="configs/config.yaml")
    dataset = benchmark.dataloader.get_dataset()
    model = get_compressing_model()

    heading = "Plot Description: "
    heading_len = len(heading)

    short_tasks = []
    for item in tqdm(dataset.itertuples(), total=len(dataset)):
        plot_task = item.task__plot_description
        plot_task = plot_task[heading_len:].strip()  # Removing heading
        task = get_task_changing_single_task(heading, plot_task)
        short_task = model.make_request(request=task)["response"]
        short_tasks.append(short_task)

    dataset["_task__plot_description_short_single"] = short_tasks
    dataset_with_short_path = dataset_json_path.with_stem(
        dataset_json_path.stem + "_with_short_single"
    )
    dataset.to_json(dataset_with_short_path)
