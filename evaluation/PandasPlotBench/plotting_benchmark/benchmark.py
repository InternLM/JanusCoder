import gc
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from plotting_benchmark.generation_engines.get_model import get_model_by_name

from .code_plot_generator import CodePlotGenerator
from .task_changer import TaskChanger
from .vis_generator import VisGenerator, add_index_to_filename
from .vis_judge import VisJudge

load_dotenv()


def get_config_template(config_folder: str | Path) -> None:
    config_folder = Path(config_folder)
    os.makedirs(config_folder, exist_ok=True)
    resource_folder = Path(__file__).parent.resolve() / "resources"
    config_file = resource_folder / "config_template.yaml"
    shutil.copyfile(config_file, config_folder / "config_template.yaml")


def get_instructs(instruct_folder: str | Path) -> None:
    os.makedirs(instruct_folder, exist_ok=True)
    instruct_folder = Path(instruct_folder)
    resource_folder = Path(__file__).parent.resolve() / "resources"
    config_file = resource_folder / "instructs.json"
    shutil.copyfile(config_file, instruct_folder / "instructs.json")


class PlottingBenchmark:
    def __init__(
        self,
        config_path: str | Path | None = None,
        config: DictConfig | None = None,
        task_changer: TaskChanger | None = None,
    ):
        self.resource_folder = Path(__file__).parent.resolve() / "resources"
        if task_changer is None:
            task_changer = TaskChanger()
        if config_path is not None:
            config = OmegaConf.load(config_path)
        elif config is None:
            raise ValueError("Provide either config or config path")
        self.config = config
        paths = self.config.paths
        benchmark_types = config.benchmark_types
        self.model_names = self.config.model_plot_gen.names

        out_folder = Path(paths.out_folder)
        # results filename is amended by model name
        out_folder.mkdir(exist_ok=True, parents=True)
        self.output_file = self.get_unique_filename(out_folder, "current_results.jsonl")
        self.bench_stat_file = out_folder / paths.bench_stat_filename

        if ("instructs_file" not in paths) or (paths.instructs_file is None):
            paths.instructs_file = self.resource_folder / "instructs.json"

        with open(paths.instructs_file, "r") as f:
            self.instructs = json.load(f)
        self.system_prompt = self.instructs["system_prompt"]
        self.dataset = load_dataset("JetBrains-Research/PandasPlotBench", split="test")
        self.model_judge = get_model_by_name(
            self.config.model_judge.name,
            dict(self.config.model_judge.parameters),
            self.system_prompt,
        )

        setup_instruct = self.instructs["setup_instruct"].replace(
            "[PLOTLIB]", self.config.plotting_lib
        )
        if ("matplotlib" not in self.config.plotting_lib) and (
            "seaborn" not in self.config.plotting_lib
        ):
            setup_instruct += f"DO NOT use matplotlib for plotting. Only {self.config.plotting_lib} library. "
        if "matplotlib" in self.config.plotting_lib:
            setup_instruct += "DO NOT use seaborn for plotting. Only pure matplotlib. "

        self.task_changer = task_changer
        self.task_changer.init_task_changer(
            data_instruct=self.instructs["data_instruct"],
            setup_instruct=setup_instruct,
            data_descriptor_name=self.config.data_descriptor,
        )

        dataset_folder = self.unpack_csv(paths.dataset_folder, self.dataset)

        self.plot_generator = VisGenerator(
            output_folder=out_folder,
            dataset=self.dataset,
            csv_folder=dataset_folder,
            config=self.config,
        )

        self.judge = VisJudge(
            vis_judge_model=self.model_judge,
            instructs=self.instructs,
            benchmark_types=benchmark_types,
            plot_lib=self.config.plotting_lib,
        )

        self.responses = None
        self.plot_responses = None

    def init_gen_model(self, model_name: str):
        self.model_plot = get_model_by_name(
            model_name,
            dict(self.config.model_plot_gen.parameters),
            self.system_prompt,
        )

        self.code_generator = CodePlotGenerator(
            model=self.model_plot,
            output_file=self.output_file,
            plotting_prompt=self.instructs["plot_instruct"],
            system_prompt=self.system_prompt,
        )

    @staticmethod
    def unpack_csv(csv_folder: str, dataset: Dataset) -> Path:
        csv_folder = Path(csv_folder)
        csv_folder.mkdir(parents=True, exist_ok=True)

        for item in dataset:
            data_csv_content = item["data_csv"]
            csv_path = csv_folder / f"data-{item['id']}.csv"

            if not csv_path.exists():
                with open(csv_path, "w") as file:
                    file.write(data_csv_content)
        print(f"CSV files unpacked to {csv_folder}")

        return csv_folder.resolve()

    @staticmethod
    def get_unique_filename(folder: Path, filename: str) -> Path:
        base, extension = os.path.splitext(filename)
        i = 1
        while (folder / filename).exists():
            filename = f"{base}_{i}{extension}"
            i += 1

        return folder / filename

    def kill_vllm(self):
        del self.model_plot.llm
        del self.model_plot
        gc.collect()
        # That import is here temporary to prevent import of cuda-libraries if they are not needed.
        import torch

        torch.cuda.empty_cache()
        print("Killed vLLM instance")

    def dump_results(self, dataset: pd.DataFrame) -> None:
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        dataset.to_json(self.results_file)
        print(f"results are dumped in {self.results_file}")

    def load_results(self, ids: list[int] | None = None) -> Dataset:
        dataset_df = pd.read_json(self.results_file)
        if isinstance(ids, list):
            dataset_df = dataset_df.loc[dataset_df["id"].isin(ids)]
        elif isinstance(ids, int):
            dataset_df = dataset_df.sample(n=ids, random_state=42)

        return dataset_df

    def run_benchmark_model(
        self,
        model_name: str,
        ids: list[int] | int | None = None,
        reuse_results: bool = False,
        load_intermediate: bool = False,
        only_stats: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        print(20 * "-")
        print(f"Benchmarking {model_name} model")
        print(20 * "-")
        gen_model_name = "_" + model_name.split("/")[-1]
        plot_lib = (self.config.plotting_lib).split(" ")[0]
        results_file_spostfix = (
            gen_model_name + "_" + plot_lib + "_" + self.config.data_descriptor
        )
        new_results_file, old_results_file = add_index_to_filename(
            self.config.paths.out_folder,
            self.config.paths.results_filename,
            results_file_spostfix,
        )

        if reuse_results or only_stats:
            self.results_file = old_results_file
            print(f"loading {self.results_file}")
            dataset_df = self.load_results(ids)
        else:
            self.init_gen_model(model_name)
            # Run the model
            self.results_file = new_results_file
            if isinstance(ids, list):
                self.dataset = self.dataset.select(ids)
            elif isinstance(ids, int):
                self.dataset = self.dataset.shuffle(seed=42).select(range(ids))
            dataset_df = self.dataset.to_pandas()
            dataset_df = self.task_changer.change_task(dataset_df)
            self.dataset = Dataset.from_pandas(dataset_df)
            dataset_df = self.code_generator.generate_codeplot_datapoints(
                self.dataset, load_intermediate
            )
            self.dump_results(dataset_df)
            # Kill the vllm model after running to
            if self.model_plot.__class__.__name__ == "VllmEngine":
                self.kill_vllm()

        if not only_stats:
            dataset_df = self.plot_generator.draw_plots(dataset_df)
            dataset_df = self.judge.score(dataset_df)
            self.dump_results(dataset_df)
        bench_stats = self.judge.calculate_stats(dataset_df)
        with open(self.bench_stat_file, "a") as f:
            json.dump(bench_stats, f)
            f.write("\n")
        print(f"Benchmark stats saved in {self.bench_stat_file}")

        return dataset_df, bench_stats

    def run_benchmark(
        self,
        ids: list[int] | int | None = None,
        reuse_results: bool = False,
        load_intermediate: bool = False,
        only_stats: bool = False,
    ) -> None:
        for model_name in self.model_names:
            self.run_benchmark_model(
                model_name,
                ids,
                reuse_results,
                load_intermediate,
                only_stats,
            )
