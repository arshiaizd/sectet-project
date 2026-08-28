#!/usr/bin/env python3
"""64-step dense faithfulness on frozen Qwen MMVP answer tokens."""

from __future__ import annotations
import argparse,json,os
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from ours.mmvp.our_method_mmvp import MAX_PIXELS,MIN_PIXELS,build_mmvp_teacher_batch,mean_probability,prepare_mmvp_prompt_inputs,target_token_probabilities
from shared.mmvp_target_utils import prompt_and_targets


def args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--Datasets",required=True); p.add_argument("--eval-list",required=True); p.add_argument("--model-id",required=True); p.add_argument("--eval-dir",required=True); p.add_argument("--division-number",type=int,default=64); return p.parse_args()


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)); os.replace(tmp,path)


def complete(path,n):
    try:
        x=json.loads(path.read_text()); return all(len(x[k])==n for k in ("insertion_score","deletion_score","insertion_word_score","deletion_word_score","region_area"))
    except Exception:return False


def perturb(image,map_,rate,insertion):
    flat=map_.ravel(); count=int(len(flat)*rate); order=np.argsort(-flat); mask=np.zeros_like(flat) if insertion else np.ones_like(flat); mask[order[:count]]=1 if insertion else 0; return (image*mask.reshape(*map_.shape,1)).astype(np.uint8)


@torch.inference_mode()
def score(processor,model,image,prompt,ids):
    batch=prepare_mmvp_prompt_inputs(processor,image,prompt,model.device); teacher=build_mmvp_teacher_batch(batch,ids,model.device); values=target_token_probabilities(model,teacher,batch["input_ids"].shape[1],ids); return mean_probability(values),values


def main():
    a=args(); processor=AutoProcessor.from_pretrained(a.model_id,use_fast=False,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS); model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model_id,torch_dtype=torch.bfloat16,device_map="auto",low_cpu_mem_usage=True).eval()
    records=json.loads(Path(a.eval_list).read_text()); root=Path(a.eval_dir); out=root/"json"; out.mkdir(parents=True,exist_ok=True)
    for r in tqdm(records,desc="Qwen MMVP dense evaluation"):
        stem=Path(r["image_path"]).stem; path=out/f"{stem}.json"
        if complete(path,a.division_number):continue
        image_path=Path(a.Datasets)/r["image_path"]; bgr=cv2.imread(str(image_path)); sal=np.asarray(np.load(root/"npy"/f"{stem}.npy"),dtype=np.float32)
        if sal.shape!=bgr.shape[:2]:sal=cv2.resize(sal,(bgr.shape[1],bgr.shape[0]))
        prompt,_,ids=prompt_and_targets(r); result={"insertion_score":[],"deletion_score":[],"insertion_word_score":[],"deletion_word_score":[],"region_area":[],"image_path":r["image_path"],"prompt":prompt,"predicted_option_text":r["predicted_option_text"],"target_token_ids":ids}
        for step in range(1,a.division_number+1):
            rate=step/a.division_number
            ins=Image.fromarray(cv2.cvtColor(perturb(bgr,sal,rate,True),cv2.COLOR_BGR2RGB)); dele=Image.fromarray(cv2.cvtColor(perturb(bgr,sal,rate,False),cv2.COLOR_BGR2RGB))
            ins_s,ins_w=score(processor,model,ins,prompt,ids); del_s,del_w=score(processor,model,dele,prompt,ids)
            result["region_area"].append(rate); result["insertion_score"].append(ins_s); result["deletion_score"].append(del_s); result["insertion_word_score"].append(ins_w); result["deletion_word_score"].append(del_w)
        atomic(path,result)


if __name__=="__main__": torch.set_grad_enabled(False); main()
