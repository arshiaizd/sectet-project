#!/usr/bin/env python3
"""InternVL3.5 TAM; average one map per predicted MMVP answer token."""

from __future__ import annotations

import argparse,json,traceback
from pathlib import Path
import numpy as np,torch
from PIL import Image
from tqdm import tqdm
from transformers import LogitsProcessor,LogitsProcessorList
from baselines.tam_for_internvl import TAM
from shared.internvl35_utils import atomic_save_npy,load_internvl,npy_is_complete,prepare_inputs
from shared.mmvp_target_utils import prompt_and_targets


class ForceTokens(LogitsProcessor):
    def __init__(self,prompt_length,ids): self.prompt_length=int(prompt_length); self.ids=list(map(int,ids))
    def __call__(self,input_ids,scores):
        step=input_ids.shape[1]-self.prompt_length
        if 0<=step<len(self.ids):
            forced=torch.full_like(scores,-float("inf")); forced[:,self.ids[step]]=0; return forced
        return scores


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--Datasets",required=True); p.add_argument("--eval-list",required=True); p.add_argument("--model-id",required=True); p.add_argument("--save-dir",required=True); return p.parse_args()


def maps_for_answer(model,processor,dtype,image_path,prompt,positions,ids,vis_dir):
    inputs=prepare_inputs(processor,str(image_path),model.device,dtype,prompt)
    outputs=model.generate(**inputs,do_sample=False,num_beams=1,min_new_tokens=len(ids),max_new_tokens=len(ids),output_hidden_states=True,return_dict_in_generate=True,logits_processor=LogitsProcessorList([ForceTokens(inputs["input_ids"].shape[1],ids)]))
    logits=[model.lm_head(features[-1]) for features in outputs.hidden_states]
    if len(logits)!=len(ids): raise RuntimeError(f"TAM generated {len(logits)} steps for {len(ids)} answer tokens")
    generated=outputs.sequences[0].cpu().tolist(); image_token_id=int(getattr(model.config,"image_token_id",None) or 151671)
    special={"img_id":[image_token_id],"prompt_id":[151653,[151645,198,151644,77091]],"answer_id":[[198,151644,77091,198],-1]}
    image=Image.open(image_path).convert("RGB")
    return [np.asarray(TAM(generated,(16,16),logits,special,image,processor,str(vis_dir/f"{Path(image_path).stem}_token{position}.jpg"),position,[],True),dtype=np.float32) for position in positions]


def main(args):
    model,processor,_,dtype=load_internvl(args.model_id); records=json.loads(Path(args.eval_list).read_text(encoding="utf-8")); out=Path(args.save_dir); nd,vd=out/"npy",out/"vis"; nd.mkdir(parents=True,exist_ok=True); vd.mkdir(parents=True,exist_ok=True)
    for record in tqdm(records,desc="InternVL MMVP TAM"):
        path=nd/f"{Path(record['image_path']).stem}.npy"
        if npy_is_complete(path): continue
        image_path=Path(args.Datasets)/record["image_path"]; prompt,positions,ids=prompt_and_targets(record)
        try: atomic_save_npy(path,np.nan_to_num(np.mean(np.stack(maps_for_answer(model,processor,dtype,image_path,prompt,positions,ids,vd)),axis=0),nan=0.0,posinf=0.0,neginf=0.0))
        except Exception: print(f"Error in InternVL MMVP TAM for {image_path}",flush=True); traceback.print_exc()


if __name__=="__main__": main(parse_args())
