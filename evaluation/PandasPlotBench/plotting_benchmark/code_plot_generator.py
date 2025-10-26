import datetime
import json
from pathlib import Path
from time import time

import pandas as pd
from datasets import Dataset
from tqdm import tqdm


def dict_of_lists_to_list_of_dicts(dict_of_lists: dict[str, list]):
    keys = dict_of_lists.keys()
    list_of_dicts = [dict(zip(keys, values)) for values in zip(*dict_of_lists.values())]

    return list_of_dicts


class CodePlotGenerator:
    """
    Object that requests to write a code for the plotting the plot.
    At init pass:

    model: the model that has method:
        make_request(task: str, system_prompt: str) -> response
        task: the task for the model, used as user input
        system_prompt: system prompt
        response: dict. {"response": response text, ... any_other_meta_info}

        NOTE that we than parse response to get a code. We assume that code block is marked as following:
        ```python
        CODE
        ```

    output_file: output file to log model responses
    plotting_prompt: Additional plotting prompt, prepended to the plot task. See example in instructs/plot_gen_and_bench.json
    system_prompt: system prompt for the model
    """

    def __init__(
        self,
        model,
        output_file: str | Path,
        plotting_prompt: str = "",
        system_prompt: str = "",
    ):
        self.model = model
        self.plotting_prompt = plotting_prompt
        self.system_prompt = system_prompt
        self.output_file = Path(output_file)

    @staticmethod
    def gather_code(answer: str) -> str:
        """
        Gather all python code blocks in a response to a single code
        """

        # pattern = r"```python\n(.*?)\n```"
        # code_blocks = re.findall(pattern, answer, re.DOTALL)
        code_blocks = [
            block.split("```")[0].strip() for block in answer.split("```python\n")[1:]
        ]
        code = "\n".join(code_blocks)
        code = code.replace('df = pd.read_csv("data.csv")', "#")
        code = code.replace("df = pd.read_csv('data.csv')", "#")
        code = code.replace('df=pd.read_csv("data.csv")', "#")
        code = code.replace("df=pd.read_csv('data.csv')", "#")
        if "np." in code:
            code = "import numpy as np\n" + code

        return code

    @staticmethod
    def generate_plotting_request(datapoint: dict, plotting_prompt: str = "") -> str:
        """
        Request to ask model to write a code for plotting. Add dataframe description
        """

        task = plotting_prompt + "\n"
        task_names = [key for key in datapoint.keys() if key.startswith("task__")]
        i = 1
        for task_name in task_names:
            task_part = datapoint[task_name].lstrip()
            if len(task_part) > 0:
                task += f"{i}. {task_part}\n"
                i += 1

        task = task.replace("\n\n", "\n")

        return task

    def generate_codeplot(self, datapoint: dict) -> dict | None:
        """
        Request a model to write a plot code for given datapoint and plotting and system prompt
        Returns raw LLM response text (only message)
        """

        task = self.generate_plotting_request(datapoint, self.plotting_prompt)
        utc_timestamp = datetime.datetime.utcnow().strftime("UTC %Y-%m-%d %H:%M:%S")
        start_time = time()
        response_raw = self.model.make_request(request=task)
        time_used_gen = time() - start_time

        if response_raw is None:
            return None

        response = {
            "raw_response": response_raw,
            "id": datapoint["id"],
            "start_time": utc_timestamp,
            "time_used_gen": time_used_gen,
            "task": task,
            "code": self.gather_code(response_raw["response"]),
            "model": self.model.name,
        }

        return response

    def generate_codeplot_vllm(self, tasks: list[str], ids: list[int]) -> list[dict]:
        """
        Request a model to write a plot code for given datapoint and plotting and system prompt
        Returns raw LLM response text (only message)
        """

        num_items = len(tasks)
        utc_timestamp = datetime.datetime.utcnow().strftime("UTC %Y-%m-%d %H:%M:%S")

        start_time = time()
        response_raw = self.model.make_request(request=tasks)
        time_used_gen_per_item = (time() - start_time) / num_items

        responses = {
            "raw_response": dict_of_lists_to_list_of_dicts(response_raw),
            "id": ids,
            "start_time": num_items * [utc_timestamp],
            "time_used_gen": num_items * [time_used_gen_per_item],
            "task": tasks,
            "code": [
                self.gather_code(response) for response in response_raw["response"]
            ],
            "model": num_items * [self.model.name],
        }

        return dict_of_lists_to_list_of_dicts(responses)

    def iterate_dataset(self, dataset):
        responses = []
        for item in tqdm(dataset):
            response = self.generate_codeplot(item)

            if response is None:
                print(f'Skipping dp {item["id"]}')
                continue

            responses.append(response)

            with open(self.output_file, "a") as f:
                json.dump(response, f)
                f.write("\n")

        return responses

    def iterate_dataset_vllm(self, dataset):
        tasks = []
        ids = []
        for item in dataset:
            task = self.generate_plotting_request(item, self.plotting_prompt)
            tasks.append(task)
            ids.append(item["id"])

        responses = self.generate_codeplot_vllm(tasks, ids)

        return responses

    def generate_codeplot_datapoints(
        self, dataset: Dataset, load_intermediate: bool = False
    ) -> pd.DataFrame:
        print("Requesting the model to write a code for plots")
        if not load_intermediate:
            print(
                f"Intermediate results would be saved in temporary file {self.output_file}"
            )

            prompt_dict = {
                "system_prompt": self.system_prompt,
                "plot_prompt": self.plotting_prompt,
            }

            with open(self.output_file, "w") as f:
                json.dump(prompt_dict, f)
                f.write("\n")

            if self.model.__class__.__name__ == "VllmEngine":
                responses = self.iterate_dataset_vllm(dataset)
            else:
                responses = self.iterate_dataset(dataset)

        else:
            responses = []
            with open(self.output_file.parent / "current_results.jsonl", "r") as f:
                for line in f:
                    response = json.loads(line)
                    if "raw_response" in line:
                        responses.append(response)

        responses = pd.DataFrame(responses)
        dataset_df = dataset.to_pandas()
        if "__index_level_0__" in dataset_df.columns:
            del dataset_df["__index_level_0__"]
        common_cols = dataset_df.columns.intersection(responses.columns).drop("id")
        dataset_df = dataset_df.drop(columns=common_cols)
        dataset_df = dataset_df.merge(responses, on="id", how="inner")

        return dataset_df
