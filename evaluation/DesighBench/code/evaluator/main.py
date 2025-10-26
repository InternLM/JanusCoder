from tqdm import tqdm
from .metric import *


re_calculate = False

def get_begin_end(framework: Framework, task: Task) -> range:
    # (framework, task): (begin, end)
    task_ranges = {
        (Framework.VANILLA, Task.GENERATION): (1, 120),
        (Framework.REACT, Task.GENERATION): (1, 109),
        (Framework.VUE, Task.GENERATION): (1, 118),
        (Framework.ANGULAR, Task.GENERATION): (1, 83),

        (Framework.VANILLA, Task.EDIT): (1, 80),
        (Framework.REACT, Task.EDIT): (1, 108),
        (Framework.VUE, Task.EDIT): (1, 105),
        (Framework.ANGULAR, Task.EDIT): (1, 66),

        (Framework.VANILLA, Task.REPAIR): (1, 28),
        (Framework.REACT, Task.REPAIR): (1, 28),
        (Framework.VUE, Task.REPAIR): (1, 27),
        (Framework.ANGULAR, Task.REPAIR): (1, 28),

        (Framework.VANILLA, Task.COMPILE): (1, 10),
        (Framework.REACT, Task.COMPILE): (1, 10),
        (Framework.VUE, Task.COMPILE): (1, 10),
        (Framework.ANGULAR, Task.COMPILE): (1, 10),
    }

    try:
        begin, end = task_ranges[(framework, task)]
    except KeyError:
        raise ValueError(f"Invalid combination: {framework.value} {task.value}")

    return range(begin, end + 1)




def get_generation_metric(reference_code_path, generated_code_path):
    generated_img_path = generated_code_path.replace(f".html", ".png")
    reference_img_path = generated_code_path.replace(f".html", "_ref.png")

    try:
        if not os.path.exists(reference_img_path):
            screenshot(reference_code_path, reference_img_path)
        if not os.path.exists(generated_img_path):
            screenshot(generated_code_path, generated_img_path)   
    except Exception as e:
        print("Error in get_generation_metric\n", e)
        metrics = {
            "CLIP": 0,
            "MAE": 0,
            "SSIM": 0,
        }
        return metrics

    reference_img = Image.open(reference_img_path)
    generated_img = Image.open(generated_img_path)

    mae = mae_score(img1=reference_img, img2=generated_img)
    cp_similarity = clip_similarity(reference_img_path, generated_img_path)
    ssim_score = ssim_similarity(reference_img_path, generated_img_path)

    metrics = {
        "CLIP": cp_similarity,
        "MAE": mae,
        "SSIM": ssim_score,
    }

    return metrics

def evaluate_generation(iterate_range, output_dir, score_file):
    if os.path.exists(score_file):
        with open(score_file, "r") as fs:
            results = json.loads(fs.read())
    else:
        results = {}

    for web_name in tqdm(iterate_range):
        if str(web_name) in results:
            continue
        
        reference_code_path = f"/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench/data/generation/vanilla/{web_name}/{web_name}.html"
        generated_code_path = os.path.join(output_dir, f"{web_name}.html")

        metrics = get_generation_metric(reference_code_path, generated_code_path)
        results[web_name] = metrics

    with open(score_file, "w") as fs:
        fs.write(json.dumps(results, indent=4))



def get_edit_metric(web_name, generated_code_path, llm_judge_flag):
    reference_path = "/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench/data/edit/vanilla/"
    config_file = reference_path + f"{web_name}/{web_name}.json"
    with open(config_file, "r") as fs:
        config = json.loads(fs.read())

    original_code_path = reference_path + f"{web_name}/{config['src_id']}.html"
    reference_code_path = reference_path + f"{web_name}/{config['dst_id']}.html"
    
    original_img_path = generated_code_path.replace(f".html", "_ori.png")
    reference_img_path = generated_code_path.replace(f".html", "_ref.png")
    generated_img_path = generated_code_path.replace(f".html", ".png")

    try:
        if not os.path.exists(original_img_path):
            screenshot(original_code_path, original_img_path)
        if not os.path.exists(reference_img_path):
            screenshot(reference_code_path, reference_img_path)
        if not os.path.exists(generated_img_path):
            screenshot(generated_code_path, generated_img_path)
    except Exception as e:
        print("Error in get_edit_metric.py\n", e)
        metrics = {
            "CLIP": 0,
            "CMS": 0,
            "MAE": 0,
            "SSIM": 0,
        }
        return metrics

    src_code = config["src_code"]
    reference_code = config["dst_code"]
    with open(generated_code_path, "r") as f_code:
        generated_code = f_code.read()


    code_score = code_similarity(
        src_code=src_code, 
        reference_code=reference_code,
        generated_code=generated_code
    )

    reference_img = Image.open(reference_img_path)
    generated_img = Image.open(generated_img_path)
    mae = mae_score(img1=reference_img, img2=generated_img)
    cp_score = clip_similarity(reference_img_path, generated_img_path)
    ssim_score = ssim_similarity(reference_img_path, generated_img_path)

    metrics = {
        "CLIP": cp_score,
        "CMS": code_score,
        "MAE": mae,
        "SSIM": ssim_score,
    }

    if llm_judge_flag:
        llm_score = llm_edit_judge(
            original_image=original_img_path, 
            edited_image=generated_img_path,
            instruction=config["prompt"]
        )
        metrics["llm score"] = llm_score
        
    return metrics

def evaluate_edit(iterate_range, mode, output_dir, score_file, llm_judge_flag):
    if os.path.exists(score_file):
        with open(score_file, "r") as fs:
            results = json.loads(fs.read())
    else:
        results = {}

    for web_name in tqdm(iterate_range):
        generated_code_path = os.path.join(output_dir, f"{web_name}_{mode}.html")
        
        if not llm_judge_flag:
            if str(web_name) in results:
                continue
            metric = get_edit_metric(str(web_name), generated_code_path, llm_judge_flag=False)
            results[str(web_name)] = metric
        else:
            try:
                if str(web_name) in results and "llm score" in results[str(web_name)]:
                    continue
                metric = get_edit_metric(str(web_name), generated_code_path, llm_judge_flag=True)
                results[str(web_name)] = metric
            except Exception as e:
                print(f"error for {web_name}", e)

    with open(score_file, "w") as fs:
        fs.write(json.dumps(results, indent=4))  


def get_repair_metric(web_name, generated_code_path, llm_judge_flag):
    reference_path = "/cpfs01/shared/XNLP_H800/sunqiushi/MCLM/eval-pack/DesignBench/data/repair/vanilla/"
    original_code_path = reference_path + f"{web_name}/{web_name}.html"
    reference_code_path = reference_path + f"{web_name}/repaired.html"
    
    original_img_path = generated_code_path.replace(f".html", "_ori.png")
    reference_img_path = generated_code_path.replace(f".html", "_ref.png")
    generated_img_path = generated_code_path.replace(f".html", ".png")

    try:
        if not os.path.exists(original_img_path):
            screenshot(original_code_path, original_img_path)
        if not os.path.exists(reference_img_path):
            screenshot(reference_code_path, reference_img_path)
        if not os.path.exists(generated_img_path):
            screenshot(generated_code_path, generated_img_path)
    except Exception as e:
        print("Error in get_edit_metric.py\n", e)
        metrics = {
            "CLIP": 0,
            "SSIM": 0,
            "MAE": 0,
            "issue accuracy": 0,
        }
        return metrics
 
    config_file = reference_path + f"{web_name}/{web_name}.json"
    with open(config_file, "r") as fs:
        config = json.loads(fs.read())
        
    src_code = config["code"]
    with open(reference_code_path, "r") as f_code:
        reference_code = f_code.read()
    with open(generated_code_path, "r") as f_code:
        generated_code = f_code.read()

    code_score = code_similarity(
        src_code=src_code, 
        reference_code=reference_code,
        generated_code=generated_code
    )

    reference_img = Image.open(reference_img_path)
    generated_img = Image.open(generated_img_path)
    mae = mae_score(img1=reference_img, img2=generated_img)
    cp_score = clip_similarity(reference_img_path, generated_img_path)
    ssim_score = ssim_similarity(reference_img_path, generated_img_path)

    issue_flag = validate_issue(
        res_path=generated_code_path.replace(f".html", ".json"),
        config_path=config_file
    )

    metrics = {
        "CLIP": cp_score,
        "CMS": code_score,
        "MAE": mae,
        "SSIM": ssim_score,
        "issue accuracy": issue_flag
    }

    if llm_judge_flag:
        with open(config_file, "r") as f_config:
            ground_truth = json.loads(f_config.read())
            issues = ground_truth["issue"]
        llm_score = llm_repair_judge(
            original_image=original_img_path, 
            repaired_image=generated_img_path,
            reference_image=reference_img_path, 
            issues=issues
        )

        metrics["llm score"] = llm_score

    return metrics

def evaluate_repair(iterate_range, mode, output_dir, score_file, llm_judge_flag):
    if os.path.exists(score_file):
        with open(score_file, "r") as fs:
            results = json.loads(fs.read())
    else:
        results = {}

    for web_name in tqdm(iterate_range):
        generated_code_path = os.path.join(output_dir, f"{web_name}_{mode}.html")

        if not llm_judge_flag:
            if str(web_name) in results:
                continue
            metric = get_repair_metric(str(web_name), generated_code_path, llm_judge_flag=False)
            results[str(web_name)] = metric
        else:
            try:
                if str(web_name) in results and "llm score" in results[str(web_name)]:
                    continue
                metric = get_repair_metric(str(web_name), generated_code_path, llm_judge_flag=True)
                results[str(web_name)] = metric
            except Exception as e:
                print(f"error for {web_name}", e)
    
    with open(score_file, "w") as fs:
        fs.write(json.dumps(results, indent=4)) 


  





