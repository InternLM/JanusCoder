import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
sys.path.append(os.path.abspath('.'))
import torch
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import argparse
from pathlib import Path
from metrics import *
from scripts.evaluation.design2code.visual_score import visual_score_v3,pre_process
import cv2
import pandas as pd
import datetime
from scripts.train.vars import SEED
from PIL import Image
import time
from tools.processor import MultiProcessor    
import multiprocessing
import signal
import shutil
from scripts.evaluation.html2screenshot import main


def html_sim_scores(html1_path, html2_path):   
    with open(html1_path, "r") as f:
         html1 = f.read()
    with open(html2_path, "r") as f:
         html2 = f.read()
    sys.setrecursionlimit(6000)
    bleu, rouge = bleu_rouge(html1, html2)
    tree_bleu, tree_rouge_1 = dom_sim(html1, html2)
 
    return (bleu, rouge, tree_bleu, tree_rouge_1)

def image_sim_scores(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(
        img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )    

    mse_value = mse(img1, img2)
    ssim_value = ssim(img1, img2)
    clip_sim_value = clip_sim(Image.open(img1_path), Image.open(img2_path), 'cpu')

    return mse_value, ssim_value,clip_sim_value

def genertor0(args):
    input_dir, output_dir, files = args
    # for file in os.listdir(input_dir):
    for file in files:
        pred_html = output_dir / f"{file}/prediction.html"
        pred_html.parent.mkdir(exist_ok=True, parents=True)
        answer_html = input_dir / f"{file}/answer.html"
        pred_screenshot = output_dir / f"{file}/prediction.png"
        answer_screenshot = output_dir / f"{file}/answer.png"

        if os.path.exists(pred_html) and os.path.exists(pred_screenshot) and os.path.exists(answer_html) and os.path.exists(answer_screenshot):
            print(f"{file} 已有数据, 直接返回")
            # return pred_html, pred_screenshot, answer_html, answer_screenshot
            continue

        # 1. 复制预测的html文件
        pred_html_origin=input_dir / f"{file}/prediction.html"        
        shutil.copy(str(pred_html_origin), str(pred_html))
        # 做了点预处理, 比较普通的操作
        try:    
            pre_process(str(pred_html))
        except Exception as e:
            print(f"fail to prreprocess: {e}")
            continue 
        
        # 2. 答案html和预测html分别渲染图像
        main(str(answer_html), str(answer_screenshot))
        main(str(pred_html), str(pred_screenshot))
   
        # 3. 复制图像
        shutil.copy(
            str(input_dir / f"{file}/image.png"), 
            str(output_dir / f"{file}/image.png")
        )
        
        # return pred_html, pred_screenshot, answer_html, answer_screenshot

def genertor1(input_dir, output_dir:Path):
    preds_html_dir = input_dir / "preds/html"
    preds_html_dir.mkdir(exist_ok=True, parents=True)
    preds_screenshot_dir = input_dir / "preds/screenshot"
    preds_screenshot_dir.mkdir(exist_ok=True, parents=True)
    answers_screenshot_dir = input_dir / "answers/screenshot"
    answers_screenshot_dir.mkdir(exist_ok=True, parents=True)
    answers_html_dir = input_dir / "answers/html"
    answers_html_dir.mkdir(exist_ok=True, parents=True)
    print("Taking screenshot of origin htmls ...")
    os.system(f"python scripts/evaluation/html2screenshot.py --input {str(answers_html_dir)} --output {str(answers_screenshot_dir)}")
    print("Taking screenshot of predictions ...")
    os.system(f"python scripts/evaluation/html2screenshot.py --input {str(preds_html_dir)} --output {str(preds_screenshot_dir)}")
    for file in tqdm(os.listdir(preds_html_dir)):
        pred_html = preds_html_dir/ file
        pred_screenshot = preds_screenshot_dir/ f"{file.split('.')[0]}.png"
        answer_html = answers_html_dir/ file
        answer_screenshot = answers_screenshot_dir/ f"{file.split('.')[0]}.png"
        yield pred_html,pred_screenshot,answer_html,answer_screenshot

def eval_work(data):
    pred_html, pred_screenshot, answer_html, answer_screenshot = data
    if not pred_screenshot.exists() or not answer_screenshot.exists():
        print(f"Screenshot file not exits:\n {str(pred_screenshot)}.\n{str(answer_screenshot)}.")
        return None
    try:
        # print("计算tree bleu")
        bleu, rouge, tree_bleu, tree_rouge_1 = html_sim_scores(answer_html, pred_html)
        # print("计算clip")
        mse_value, ssim_value, clip_sim = image_sim_scores(str(pred_screenshot), str(answer_screenshot))
        # print("计算visual")
        _, visual, block_match, text_match, position_match, text_color_match, clip_score = \
            visual_score_v3(str(answer_html), str(pred_html), str(answer_screenshot), str(pred_screenshot), str(pred_screenshot.parent), device="cpu")
        return str(answer_html), str(pred_html), visual, clip_sim, tree_rouge_1
    except Exception as e:
        print(f"Error: {e} when eval {pred_html}")
        return None
    
    # with open(out_df, "a+") as f_csv:
    #     # f_csv.write(f"{str(pred_html)},{str(answer_html)},{bleu},{rouge},{tree_bleu}, {tree_rouge_1},{mse_value},{ssim_value},{clip_sim},{block_match}, {text_match}, {position_match}, {text_color_match}, {clip_score}\n")
    #     f_csv.write(f"{str(answer_html)},{str(pred_html)},{visual},{clip_sim},{tree_rouge_1}\n")

def eval_batch(batch):
    outputs = []
    for item in tqdm(batch):
        output = eval_work(item)
        if output is not None:
            outputs.append(output)
    return outputs

def eval(input_dir, output_dir, generator_choice):
    # generator_map={'0':genertor0,'1':genertor1}
    # generator=generator_map[generator_choice]
    device = 'cuda'
    torch.manual_seed(SEED)    
    
    # 创建输出文件夹
    input_dir = Path(input_dir)
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    # output_dir = Path(output_dir) / f"eval_{input_dir.name}_{time_string}
    output_dir = Path(output_dir) / f"{input_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 进行预处理, 复制文件, 渲染图片等
    PROCESS_NUM = 16
    files = os.listdir(input_dir)
    batches = []
    for i in range(0, len(files), len(files)//PROCESS_NUM):
        batches.append((input_dir, output_dir, files[i:i + len(files)//PROCESS_NUM]))
    with multiprocessing.Pool(PROCESS_NUM) as pool:
        pool.map(genertor0, batches)
    
    print("预处理完成, 开始计算指标")
    inputs = []
    for file in os.listdir(output_dir):
        pred_html = output_dir / f"{file}/prediction.html"
        answer_html = input_dir / f"{file}/answer.html"
        pred_screenshot = output_dir / f"{file}/prediction.png"
        answer_screenshot = output_dir / f"{file}/answer.png"

        if os.path.exists(pred_html) and os.path.exists(pred_screenshot) and os.path.exists(answer_html) and os.path.exists(answer_screenshot):
            inputs.append((pred_html, pred_screenshot, answer_html, answer_screenshot))
    batches = []
    batch_size = len(inputs)//PROCESS_NUM
    for i in range(0, len(inputs), batch_size):
        batches.append(inputs[i: min(len(inputs), i+batch_size)])
    print(f"样本数量: {len(inputs)}; batch数量: {len(batches)}; batch size: {batch_size}")
    with multiprocessing.Pool(PROCESS_NUM) as pool:
        batch_outputs = pool.map(eval_batch, batches)
    outputs = []
    for batch_output in batch_outputs:
        outputs += batch_output
    print(f"成功计算指标的样本数量: {len(outputs)}")

    print("预处理结束, 开始计算指标")
    # 创建csv文件, 存储每个样本的指标
    out_df = output_dir / "metrics_result.csv"
    with open(out_df, "w") as f_csv:
        f_csv.write("origin,pred,Visual,CLIP,TreeBLEU\n")

    with open(out_df, "a+") as f_csv:
        for answer_html, pred_html, visual, clip_sim, tree_rouge_1 in outputs:
            f_csv.write(f"{answer_html},{pred_html},{visual},{clip_sim},{tree_rouge_1}\n")
    
    df = pd.read_csv(out_df)
    avg_score = output_dir / "avg_result.txt"
    with open(avg_score, "w", encoding="utf-8") as f:
        for c in df.columns:
            if c not in ["origin","pred"]:
                f.write(f"{c}:{df[c].mean():.4f}\n")
                print(f"{c}:{df[c].mean():.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process two path strings.')
    # Define the arguments
    parser.add_argument('--input', "-i", type=str, default="")  
    parser.add_argument('--output', "-o", type=str, default="outdir") 
    parser.add_argument('--generator', "-g", type=str, choices=['0','1'], default='0') 
    # Parse the arguments
    args = parser.parse_args()
    if "results_lite" in args.input:
        args.output = "outdir_lite"
        

    def signal_handler(signal, frame):
        print(f'signal {signal} recieved, exit.')         
        for p in multiprocessing.active_children():            
            os.kill(p.pid, signal.SIGKILL)
        os._exit(1)    

    signal.signal(signal.SIGINT, signal_handler)   

    # InternVL3_5-8B-Instruct
    for model_name in ["final0_internvl3.5-4b_0820"]:
        for dataset in ["short", "mid", "long"]:
            args.input = f"/cpfs01/shared/XNLP_H800/liuyang/UI2Code_Baselines/webcode2m/results/{model_name}_{dataset}"
            if not os.path.exists(args.input):
                print(f"不存在 {args.input}")
                continue
   
            print(args)
            eval(args.input, args.output,args.generator) 