import vllm
import json
import os
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vllm inference on a dataset.")
    parser.add_argument("--process_base", type=str, default='/cpfs01/shared/XNLP_H800/chenqiaosheng/Code/ArtifactsBenchmark/dataset/', help="Base directory for dataset and output")
    parser.add_argument("--dataset_path", type=str, default='artifacts_bench.json', help="Relative path to the dataset file")
    parser.add_argument("--dump_path", type=str, default='artifacts_bench_qwen3_8b.json', help="Relative path to the output file")
    parser.add_argument("--model_path", type=str, default="/cpfs01/shared/XNLP_H800/hf_hub/Qwen3-8B", help="Path to the model directory")

    args = parser.parse_args()

    process_base = args.process_base
    dataset_path = args.dataset_path
    dump_path = args.dump_path
    model_path = args.model_path

    dataset_fullpath = os.path.join(process_base, dataset_path)
    dump_fullpath = os.path.join(process_base, dump_path)

    llm = vllm.LLM(model=model_path, tensor_parallel_size=1, max_model_len=None, max_seq_len_to_capture=None)
    sampling_params = vllm.SamplingParams(max_tokens=None, ignore_eos=False)

    source_list = []
    with open(dataset_fullpath, 'r') as f_in:
        for line in tqdm(f_in):
            line_parsed = json.loads(line)
            source_list.append(line_parsed)

    batch_size = 2048 # Can set to dataset length since vllm process data in a sample-based manner
    with open(dump_fullpath, 'w') as f_out:
        for i in range(0, len(source_list), batch_size):
            sample_split = source_list[i:i + batch_size]
            split_questions = []
            for sample in sample_split:
                question = sample['question']
                split_questions.append(question)
            outputs = llm.generate(split_questions, sampling_params)
            split_answers = []
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                split_answers.append(generated_text)
            for sample, answer in zip(sample_split, split_answers):
                dump_dict = {
                    "index": sample['index'],
                    "question": sample['question'],
                    "answer": answer
                }
                dump_str = json.dumps(dump_dict, ensure_ascii=False)
                f_out.write(dump_str + '\n')