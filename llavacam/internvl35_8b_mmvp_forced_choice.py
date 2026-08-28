#!/usr/bin/env python3
"""InternVL3.5 LLaVA-CAM on all predicted MMVP option tokens."""

from __future__ import annotations

import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from tqdm import tqdm
from baselines.llavacam import LLaVACAM
from shared.internvl35_utils import atomic_save_npy,load_internvl,npy_is_complete,prepare_inputs
from shared.mmvp_target_utils import prompt_and_targets


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--Datasets",required=True); p.add_argument("--eval-list",required=True); p.add_argument("--model-id",required=True); p.add_argument("--save-dir",required=True); return p.parse_args()


def main(args):
    model,processor,_,dtype=load_internvl(args.model_id,gradients=True)
    explainer=LLaVACAM(model,processor,model.model.language_model.layers[32].post_attention_layernorm,mode="internvl")
    records=json.loads(Path(args.eval_list).read_text(encoding="utf-8")); out=Path(args.save_dir); nd,vd=out/"npy",out/"visualization"; nd.mkdir(parents=True,exist_ok=True); vd.mkdir(parents=True,exist_ok=True)
    for record in tqdm(records,desc="InternVL MMVP LLaVA-CAM"):
        stem=Path(record["image_path"]).stem; npth=nd/f"{stem}.npy"
        if npy_is_complete(npth): continue
        image_path=Path(args.Datasets)/record["image_path"]; prompt,positions,ids=prompt_and_targets(record)
        inputs=prepare_inputs(processor,str(image_path),model.device,dtype,prompt); prompt_ids=inputs["input_ids"][0].tolist()
        explainer.generated_ids=torch.tensor([prompt_ids+ids],dtype=torch.long,device=model.device)
        explainer.target_token_position=np.asarray(positions)+len(prompt_ids); explainer.selected_interpretation_token_word_id=ids
        heatmap=np.nan_to_num(np.asarray(explainer.generate_smooth_cam(str(image_path),prompt),dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0); atomic_save_npy(npth,heatmap)
        original=cv2.imread(str(image_path)); colored=cv2.applyColorMap(np.uint8(255*cv2.resize(heatmap,(original.shape[1],original.shape[0]))),cv2.COLORMAP_JET)
        cv2.imwrite(str(vd/record["image_path"]),np.clip(.4*colored+original,0,255).astype(np.uint8))


if __name__=="__main__": main(parse_args())
