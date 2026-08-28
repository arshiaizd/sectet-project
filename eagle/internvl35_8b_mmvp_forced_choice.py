#!/usr/bin/env python3
"""InternVL3.5 EAGLE on all tokens of the predicted MMVP option."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from internvl35_8b_coco_object import InternVLAdaptor, outputs_are_complete
from shared.internvl35_utils import atomic_save_npy, atomic_write_json, load_internvl, prepare_inputs
from shared.mmvp_target_utils import prompt_and_targets
from submodular_vision import MLLMSubModularExplanationVision
from utils import SubRegionDivision


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Datasets",required=True); p.add_argument("--eval-list",required=True)
    p.add_argument("--model-id",required=True); p.add_argument("--save-dir",required=True)
    p.add_argument("--division-number",type=int,default=64); p.add_argument("--lambda1",type=float,default=1.0); p.add_argument("--lambda2",type=float,default=1.0)
    return p.parse_args()


def main(args):
    model,processor,tokenizer,dtype=load_internvl(args.model_id)
    adaptor=InternVLAdaptor(model,processor,dtype)
    explainer=MLLMSubModularExplanationVision(adaptor,lambda1=args.lambda1,lambda2=args.lambda2)
    records=json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    root=Path(args.save_dir)/f"slico-{args.lambda1}-{args.lambda2}-division-number-{args.division_number}"
    jd,nd=root/"json",root/"npy"; jd.mkdir(parents=True,exist_ok=True); nd.mkdir(parents=True,exist_ok=True)
    for record in tqdm(records,desc="InternVL MMVP EAGLE"):
        stem=Path(record["image_path"]).stem; jp,npth=jd/f"{stem}.json",nd/f"{stem}.npy"
        if outputs_are_complete(jp,npth): continue
        image_path=Path(args.Datasets)/record["image_path"]; prompt,positions,ids=prompt_and_targets(record)
        inputs=prepare_inputs(processor,str(image_path),model.device,dtype,prompt); prompt_ids=inputs["input_ids"][0].tolist()
        adaptor.text_prompt=prompt
        adaptor.generated_ids=torch.tensor([prompt_ids+ids],dtype=torch.long,device=model.device)
        adaptor.target_token_position=np.asarray(positions)+len(prompt_ids)
        adaptor.selected_interpretation_token_word_id=ids
        image=cv2.imread(str(image_path)); region_size=int((image.shape[0]*image.shape[1]/args.division_number)**0.5)
        regions=SubRegionDivision(image,mode="slico",region_size=region_size)
        selected,saved=explainer(image,regions)
        saved.update({"image_path":record["image_path"],"prompt":prompt,
            "predicted_option_text":record["predicted_option_text"],"prediction_correct":record["prediction_correct"],
            "selected_interpretation_token_id":positions,"selected_interpretation_token_word_id":ids,
            "output_word_id":ids,"output_words":tokenizer.convert_ids_to_tokens(ids),"target_protocol":record["target_protocol"]})
        atomic_save_npy(npth,np.asarray(selected)); atomic_write_json(jp,saved)


if __name__=="__main__": main(parse_args())
