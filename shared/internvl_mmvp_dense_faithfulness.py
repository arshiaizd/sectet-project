#!/usr/bin/env python3
"""64-step dense faithfulness on frozen InternVL MMVP answer tokens."""

from __future__ import annotations
import argparse,json,os
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from tqdm import tqdm
from shared.internvl35_utils import load_internvl,prepare_inputs
from shared.mmvp_target_utils import prompt_and_targets


def args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--Datasets",required=True); p.add_argument("--eval-list",required=True); p.add_argument("--model-id",required=True); p.add_argument("--eval-dir",required=True); p.add_argument("--division-number",type=int,default=64); return p.parse_args()


def perturb(image,map_,rate,insertion):
    flat=map_.ravel(); count=int(len(flat)*rate); order=np.argsort(-flat); mask=np.zeros_like(flat) if insertion else np.ones_like(flat); mask[order[:count]]=1 if insertion else 0; return (image*mask.reshape(*map_.shape,1)).astype(np.uint8)


@torch.inference_mode()
def score(model,processor,dtype,image,prompt,ids):
    batch=prepare_inputs(processor,image,model.device,dtype,prompt); plen=batch["input_ids"].shape[1]
    prefix=torch.tensor([ids[:-1]],dtype=batch["input_ids"].dtype,device=model.device)
    if prefix.numel(): batch["input_ids"]=torch.cat([batch["input_ids"],prefix],1)
    batch["attention_mask"]=torch.ones_like(batch["input_ids"])
    for k in ("position_ids","cache_position","rope_deltas"):batch.pop(k,None)
    logits=model(**batch,use_cache=False,return_dict=True).logits[0].float(); selected=logits[plen-1:plen-1+len(ids)].softmax(-1); index=torch.tensor(ids,device=selected.device); values=selected.gather(1,index[:,None]).squeeze(1).tolist(); return float(sum(values)/len(values)),[float(v) for v in values]


def complete(path,n):
    try:
        x=json.loads(path.read_text());return all(len(x[k])==n for k in ("insertion_score","deletion_score","insertion_word_score","deletion_word_score","region_area"))
    except Exception:return False


def main():
    a=args(); model,processor,_,dtype=load_internvl(a.model_id); records=json.loads(Path(a.eval_list).read_text()); root=Path(a.eval_dir); out=root/"json"; out.mkdir(parents=True,exist_ok=True)
    for r in tqdm(records,desc="InternVL MMVP dense evaluation"):
        stem=Path(r["image_path"]).stem; path=out/f"{stem}.json"
        if complete(path,a.division_number):continue
        bgr=cv2.imread(str(Path(a.Datasets)/r["image_path"])); sal=np.asarray(np.load(root/"npy"/f"{stem}.npy"),dtype=np.float32)
        if sal.shape!=bgr.shape[:2]:sal=cv2.resize(sal,(bgr.shape[1],bgr.shape[0]))
        prompt,_,ids=prompt_and_targets(r); result={"insertion_score":[],"deletion_score":[],"insertion_word_score":[],"deletion_word_score":[],"region_area":[],"image_path":r["image_path"],"prompt":prompt,"predicted_option_text":r["predicted_option_text"],"target_token_ids":ids}
        for step in range(1,a.division_number+1):
            rate=step/a.division_number; ins=Image.fromarray(cv2.cvtColor(perturb(bgr,sal,rate,True),cv2.COLOR_BGR2RGB)); dele=Image.fromarray(cv2.cvtColor(perturb(bgr,sal,rate,False),cv2.COLOR_BGR2RGB))
            ins_s,ins_w=score(model,processor,dtype,ins,prompt,ids);del_s,del_w=score(model,processor,dtype,dele,prompt,ids)
            result["region_area"].append(rate);result["insertion_score"].append(ins_s);result["deletion_score"].append(del_s);result["insertion_word_score"].append(ins_w);result["deletion_word_score"].append(del_w)
        tmp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp");tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));os.replace(tmp,path)


if __name__=="__main__":torch.set_grad_enabled(False);main()
