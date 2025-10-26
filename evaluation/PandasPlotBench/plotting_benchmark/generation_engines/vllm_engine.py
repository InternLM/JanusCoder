import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vllm import LLM, RequestOutput, SamplingParams


def check_files_exist(folder_path: Path | str, filenames: list[str]) -> bool:
    folder_path = Path(folder_path)
    existing_files = [(folder_path / filename).exists() for filename in filenames]
    return all(existing_files)


def get_model_name_and_path(
    model_name_or_path: str | Path, model_name: str | None = None
) -> tuple[str, str | None]:
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    # "special_tokens_map.json",
    tok_files_exist = check_files_exist(model_name_or_path, tokenizer_files)
    tok_name_or_path = None
    if not Path(model_name_or_path).exists():
        model_name = str(model_name_or_path)
    else:
        config_path = Path(model_name_or_path) / "config.json"
        if config_path.exists:
            with open(config_path, "r") as file:
                config_data = json.load(file)
            if config_data.get("_name_or_path") is not None:
                model_name = config_data.get("_name_or_path")
        if (not tok_files_exist) and (model_name is None):
            raise AttributeError(
                "You have no tokenizer files in model folder.\n"
                "Please provide model name for tokenizer either in config.json file in the model folder\n"
                "or as model.model_name parameter"
            )
        elif tok_files_exist and model_name is None:
            model_name = str(model_name_or_path)
        else:
            tok_name_or_path = model_name

    return model_name, tok_name_or_path


class VllmEngine:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are helpful assistant",
        add_args: dict = {},
        vllm_args: dict = {},
        generation_args: dict = {},
    ):
        self.name, tokenizer_name = get_model_name_and_path(
            model_name_or_path=model_name
        )
        if tokenizer_name is not None:
            vllm_args["tokenizer"] = tokenizer_name
        vllm_args.update({"max_model_len": 32000, "max_seq_len_to_capture": 32000, "trust_remote_code": True})
        if "temperature" in add_args:
            generation_args.update({"temperature": add_args["temperature"]})
        else:
            generation_args.update({"temperature": 0.0})
        generation_args.update({"max_tokens": 16000, "ignore_eos": False})
        self.llm = LLM(model=model_name, **vllm_args)
        self.sampling_params = SamplingParams(**generation_args)
        self.system_prompt = system_prompt

    def generate(
        self,
        input_texts: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        # responses = self.llm.generate(
        #     prompts=input_texts,
        #     sampling_params=self.sampling_params,
        #     # use_tqdm=False
        # )
        responses = self.llm.chat(
            input_texts,
            sampling_params=self.sampling_params
            # use_tqdm=False
        )
        outputs = [self.get_outputs(response) for response in responses]

        return self.batch_output(outputs)

    def format_input(self, message: str) -> str:
        # if "meta-llama" in self.name:
        system_mes = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
        user_mes = f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
        assist_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        model_input = system_mes + user_mes + assist_prompt
        # else:
        #     model_input = (self.system_prompt + "\n" + message).strip()

        return model_input

    def make_request(
        self,
        request: str | list[str],
    ) -> dict | None:
        if isinstance(request, str):
            requests = [request]
        else:
            requests = request
        # requests = [self.format_input(request) for request in requests]
        requests = [[{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': request}] for request in requests]
        response = self.generate(input_texts=requests)

        return {"response": response["text"]}

    @staticmethod
    def get_outputs(response: RequestOutput) -> dict[str, Any]:
        metainfo = asdict(response.outputs[0])
        del metainfo["text"], metainfo["token_ids"]
        metainfo["time_metrics"] = asdict(response.metrics) if response.metrics else {}
        output_dict = {
            "text": response.outputs[0].text,
            "tokens": list(response.outputs[0].token_ids),
            "metainfo": metainfo,
        }
        return output_dict

    @staticmethod
    def batch_output(outputs: list[dict[str, Any]]) -> dict[str, list[Any]]:
        batched_output = defaultdict(list)
        for d in outputs:
            for key, value in d.items():
                batched_output[key].append(value)
        return dict(batched_output)
