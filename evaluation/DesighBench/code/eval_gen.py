from evaluator.main import *
from evaluator.compile import *
models = [
    "Qwen2.5-VL-7B-Instruct"
] # Evaluated MLLMs

frame_works = ["vanilla"] # the framework used to actually implement the webpage.
implemented_frame_works = ["vanilla"] # the framework used by the MLLMs.

# collect the compile information
for frame_work in frame_works:
    if frame_work == "vanilla":
        continue
    for implemented in implemented_frameworks:
        collect_compile_information(task_name=Task.GENERATION, frame_work=frame_work, implemented_framework_or_mode=implemented)

evaluate_generation(models=models, frame_works=frame_works, implemented_frameworks=implemented_frame_works)