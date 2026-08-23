import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_ID = "/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"

import cv2
import json

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import argparse
import torch
from torch import nn
import torchvision.transforms.functional as TF

import numpy as np
from utils import SubRegionDivision, mkdir

from tqdm import tqdm

from baselines.IGOS_pp.utils import *
from baselines.IGOS_pp.methods_helper import *
from baselines.IGOS_pp.IGOS_pp import *

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco/val2017',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/coco_single_target_once_qwen25vl-3B.json',
                        help='Datasets.')
    parser.add_argument('--model-id',
                        type=str,
                        default=MODEL_ID,
                        help='Local Qwen checkpoint path or Hugging Face model ID.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/Qwen2.5-VL-3B-coco-object/IGOS_PP',
                        help='output directory to save results')
    args = parser.parse_args()
    return args


def output_is_complete(path):
    try:
        heatmap = np.load(path, mmap_mode="r")
        valid = heatmap.ndim == 2 and heatmap.size > 0
        del heatmap
        return valid
    except (OSError, EOFError, ValueError):
        return False


def atomic_save_npy(path, array):
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as stream:
        np.save(stream, array)
    os.replace(temporary_path, path)

def main(args):
    text_prompt = "Describe the image in one factual English sentence of no more than 20 words. Do not include information that is not clearly visible."
    
    # Load Qwen2.5-VL
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # default processor
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer

    explainer = gen_explanations_qwenvl
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
        
    save_dir = args.save_dir
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_vis_root_path = os.path.join(save_dir, "visualization")
    mkdir(save_vis_root_path)
    
    # visualization_root_path = os.path.join(save_dir, "vis")
    # mkdir(visualization_root_path)
    
    for content in tqdm(contents):
        output_path = os.path.join(
            save_npy_root_path, content["image_path"].replace(".jpg", ".npy")
        )
        if output_is_complete(output_path):
            continue
        
        image_path = os.path.join(args.Datasets, content["image_path"])
        # text_prompt = content["question"]
        
        image = Image.open(image_path)
        
        heatmap, superimposed_img = explainer(model, processor, image, text_prompt, tokenizer, positions=[content["target_generated_index"]])

        # Save npy file
        atomic_save_npy(output_path, np.array(heatmap))
        
        cv2.imwrite(os.path.join(save_vis_root_path, content["image_path"]), superimposed_img)
        
if __name__ == "__main__":
    args = parse_args()
    
    main(args)
