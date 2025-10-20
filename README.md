# JanusCoder
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2505.19897-b31b1b.svg)](https://arxiv.org/abs/2505.19897)  -->
![License](https://img.shields.io/badge/License-MIT-blue)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](./JanusCoder_technical_report.pdf)
[![Discord](https://img.shields.io/discord/1222168244673314847?logo=discord&style=flat)](https://discord.com/invite/rXS2XbgfaD)
<!-- [![Generic badge](https://img.shields.io/badge/WeChat-机器之心-green.svg?logo=wechat)](https://mp.weixin.qq.com/s/naVskQ9btJFkoUyyQVr7zA) -->
<!-- [![🌐 Website](https://img.shields.io/badge/Website-🌐-informational)](https://qiushisun.github.io/ScienceBoard-Home/) -->
<!-- <a href = "https://zhuanlan.zhihu.com/p/1914038712540574158"><img src="https://img.shields.io/badge/-%E7%9F%A5%E4%B9%8E-%232f6be0" target="_blank"></a> -->

JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence

## 🗞️ Updates

- **2025-10-07**: Initial release of our [technical report](./JanusCoder_technical_report.pdf), code, data samples, and [🌐 Project Page](https://qiushisun.github.io/ScienceBoard-Home/). Check it out! 🚀


This release represents the public implementation; the full implementation and data will be made available after internal company policy requirements are met.

## 📑 Intro

JanusCoder is a suite of open models that establish a unified visual–programmatic interface for multimodal code intelligence. The models (JanusCoder and JanusCoderV) handle text-centric and vision-centric tasks in a single framework—from chart-to-code and web UI generation/editing to dynamic theorem visualizations—and show strong results across public benchmarks, approaching or even surpassing proprietary systems.

<img src="./assets/januscoder_overview.png" alt="overview" style="zoom:80%;" />



> [!NOTE]  
> Due to company policy, we need some additional time to release all datasets and checkpoints. If you require access to more data, please feel free to send qiushisun@connect.hku.hk an email.

## 📑 Data Synthesis Toolkit

We provide a versatile data synthesis toolkit that generates multimodal code data across heterogeneous domains—ranging from charts and Web UIs to visual artifacts and code-driven animations—while greatly reducing engineering efforts for large-scale corpus creation. ￼


<img src="./assets/januscoder_data_toolkit.png" alt="overview" style="zoom:80%;" />


Since the process of building JanusCode data involves a variety of synthesis pipelines, we provide a few examples here:

### Example Workflows:

Extend, refine and derive new text-centric data for chart tasks
   
```bash
python data_synthesis/viscode_extend_synthesis_pipeline.py \
    --input raw_data/viscode \
    --output processed/viscode_extended
    --output processed/mathematica_extended
```

Extend and derive new text-centric data for visual editing tasks

```bash
python data_synthesis/viscode_edit_synthesis_pipeline.py \
    --input processed/viscode_extended \
    --output processed/viscode_edited
    --output processed/mathematica_extended
```

Build data for generating dynamic animations with Manim


```bash
python data_synthesis/recontext_manim_data.py \
    --input raw_data/manim \
    --output processed/manim_recontext
    --output processed/mathematica_extended
```

Extend scientific visualizations with Mathematica

```bash
python data_synthesis/mathematica_extend_synthesis_pipeline.py \
    --input raw_data/mathematica \
    --output processed/mathematica_extended
```

More scripts will be released soon.

Data Samples:
1. We provide text-centric data samples [at this link](https://drive.google.com/file/d/1dSxNf-co4LGh93NoiUgWKdbcf8Mo_VWG/view?usp=sharing).
2. We provide vision-centric data samples [at this link](https://drive.google.com/file/d/1dSxNf-co4LGh93NoiUgWKdbcf8Mo_VWG/view?usp=sharing).


## 🧪 Training
We primarily follow the official training pipelines provided. Users can directly refer to the linked repositories for detailed instructions on SFT.

| Our Model        | Upstream Base | Training Pipelines |
|-------------------|---------------|----------------------------------|
| JanusCoder-8B     | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [Qwen3 GitHub](https://github.com/QwenLM/Qwen) |
| JanusCoder-14B    | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | [Qwen3 GitHub](https://github.com/QwenLM/Qwen) |
| JanusCoderV-7B    | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [Qwen-VL GitHub](https://github.com/QwenLM/Qwen-VL) |
| JanusCoderV-8B    | [OpenGVLab/InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | [InternVL GitHub](https://github.com/OpenGVLab/InternVL) |

We also provide some typical training configuration file for llamafactory users in [training_files](./training_files/).

All our experiments were conducted on interconnected 8× H800 GPUs.

## 📏 Evaluation


We provide several ready-to-use scripts to quickly reproduce our experimental results. You can replace them with other scripts under the evaluation directory to evaluate different tasks, for example:

```bash
bash DesignBench/scripts/designbench_vllm-januscoderv.sh
```

For evaluations on LiveCodeBench-v6, MBPP+: We directly adopt the evaluation scripts provided by OpenCompass.



## 📚License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## 📋 Citation
If you are interested in our work or find this repository / our data helpful, please consider using the following citation format when referencing our paper:

```bibtex
@article{sun2025januscoder,
  title={JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence},
  author={Sun, Qiushi and Gong, Jingyang and Liu, Yang and Chen, Qiaosheng and Li, Lei and Chen, Kai and Guo, Qipeng and Kao, Ben and Yuan, Fei},
  journal={Preprint},
  year={2025}
}
```