import os
# Local server model path is used directly below; no HF download needed.
# (Kept minimal on purpose so nothing tries to reach the internet.)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Local paths on the server (override the argparse defaults so the script can be
# run with no extra flags). These match your other experiment scripts.
MODEL_ID = "/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"

import cv2
import json
import importlib.util
import time

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import argparse
import torch
from torch import nn
import torchvision.transforms.functional as TF

import numpy as np
from submodular_vision import MLLMSubModularExplanationVision
from utils import SubRegionDivision, mkdir

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='/home/mmd/asal/EAGLE/datasets/coco/val2017',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='/home/mmd/asal/EAGLE/datasets/coco_single_target_once_qwen25vl-7B-subset100.json',
                        help='Datasets.')
    parser.add_argument('--model-id',
                        type=str,
                        default=MODEL_ID,
                        help='Local Qwen checkpoint path or Hugging Face model ID.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--lambda1', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--division-number', 
                        type=int, default=64,
                        help='')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./interpretation_results/Qwen2.5-VL-7B-coco-object/',
                        help='output directory to save results')
    parser.add_argument('--attention-implementation',
                        type=str,
                        default='auto',
                        choices=['auto', 'eager', 'flash_attention_2'],
                        help='Attention backend. Auto uses FlashAttention-2 on supported GPUs.')
    args = parser.parse_args()
    return args

def outputs_are_complete(json_path, npy_path):
    """Return True only when both per-image outputs are readable and complete."""
    if not (os.path.isfile(json_path) and os.path.isfile(npy_path)):
        return False

    required_json_keys = {
        "insertion_score",
        "deletion_score",
        "smdl_score",
        "region_area",
        "sub-region_number",
        "selected_interpretation_token_id",
        "selected_interpretation_token_word_id",
    }

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            saved_json = json.load(f)

        saved_regions = np.load(npy_path, mmap_mode="r")
        valid = (
            required_json_keys.issubset(saved_json)
            and saved_regions.ndim == 4
            and saved_regions.shape[0] == saved_json["sub-region_number"]
        )
        del saved_regions
        return valid
    except (OSError, EOFError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False

def atomic_save_npy(path, array):
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as f:
        np.save(f, array)
    os.replace(temporary_path, path)

def atomic_save_json(path, data):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(",", ":"))
    os.replace(temporary_path, path)


def select_rank_contents(contents, begin, end, rank, world_size):
    selected = contents[begin:end]
    return selected[rank::world_size]


def configure_worker(attention_implementation):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    capability_major, _ = torch.cuda.get_device_capability(device)
    flash_available = importlib.util.find_spec("flash_attn") is not None
    flash_supported = capability_major >= 8 and flash_available

    if attention_implementation == "auto":
        attention_implementation = (
            "flash_attention_2" if flash_supported else "eager"
        )
    elif attention_implementation == "flash_attention_2" and not flash_supported:
        raise RuntimeError(
            "FlashAttention-2 requires an Ampere-or-newer GPU and flash-attn."
        )

    model_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    return rank, local_rank, world_size, device, model_dtype, attention_implementation


class QwenVLAdaptor(torch.nn.Module):
    def __init__(self, 
                 model,
                 processor,
                 device = "cuda"):
        super().__init__()
        self.model = model
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        
        self.processor = processor
        self.image_only_text = None
        
        self.generated_ids = None
        
        # The position of the token that needs to be explained in the newly generated content (include all tokens)
        self.target_token_position = None
        self.selected_interpretation_token_word_id = None
    
    def forward(self, image):
        """_summary_

        Args:
            image: PIL format
        """
        if isinstance(image, torch.Tensor):
            if image.shape[-1] == 3:
                image_tensor = image[..., [2, 1, 0]]  # BGR → RGB
                image_tensor = image_tensor.permute(2, 0, 1)
                image_tensor = image_tensor.clamp(0, 255).byte()
                image = TF.to_pil_image(image_tensor)

        info = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                ],},
            ]
        
        # Preparation for inference
        if self.image_only_text is None:
            self.image_only_text = self.processor.apply_chat_template(
                info, tokenize=False, add_generation_prompt=True
            )
        text = self.image_only_text
        image_inputs, video_inputs = process_vision_info(info)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        self.generated_ids = self.generated_ids[:max(self.target_token_position)]   #bug
        inputs['input_ids'] = self.generated_ids
        inputs['attention_mask'] = torch.ones_like(self.generated_ids)

        # === ONLY behavioural deviation from the original EAGLE script ===
        # The original was written for an older transformers where replacing
        # input_ids after the processor pass was harmless. On transformers
        # >=4.49 the processor/model carry stale position_ids / cache_position /
        # rope_deltas sized for the ORIGINAL input, so after we shrink input_ids
        # to the teacher-forced caption, get_rope_index mismatches and raises.
        # Deleting them forces RoPE to be recomputed for the new length. This
        # does not change the scoring math; it only lets the identical forward
        # run on a modern transformers. Remove these 3 lines to get byte-for-byte
        # original behaviour on the transformers version EAGLE shipped with.
        for stale_key in ("position_ids", "cache_position", "rope_deltas"):
            if stale_key in inputs:
                del inputs[stale_key]

        inputs = inputs.to(self.model.device)    # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
        
        # Forward calculation to get all logits (including the logits of the input part)
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                return_dict=True,
                use_cache=False,
            )
            all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        if self.generated_ids != None:
            returned_logits = all_logits[:, self.target_token_position - 1] # The reason for the minus 1 is that the generated content is in the previous position
            returned_logits = self.softmax(returned_logits)
            
            if self.selected_interpretation_token_word_id != None:
                self.selected_interpretation_token_word_id = torch.tensor(self.selected_interpretation_token_word_id).to(self.model.device)
                indices = self.selected_interpretation_token_word_id.unsqueeze(0).unsqueeze(-1) # [1, N, 1]
                
                returned_logits = returned_logits.gather(dim=2, index=indices) # [1, N, 1]
                
                returned_logits = returned_logits.squeeze(-1)  # [1, N]
        return returned_logits[0]   # size [N]

def main(args):
    text_prompt = "Describe the image in one factual English sentence of no more than 20 words. Do not include information that is not clearly visible."

    rank, local_rank, world_size, device, model_dtype, attention_implementation = (
        configure_worker(args.attention_implementation)
    )
    print(
        "rank={} local_rank={} world_size={} gpu={} dtype={} attention={}".format(
            rank,
            local_rank,
            world_size,
            torch.cuda.get_device_name(device),
            model_dtype,
            attention_implementation,
        ),
        flush=True,
    )
    
    # Load Qwen2.5-VL from the local server path
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        device_map={"": device},
        attn_implementation=attention_implementation,
    )
    model.eval()
    
    # default processor
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    
    # Encapsulation Qwen
    Qwen = QwenVLAdaptor(
        model=model,
        processor=processor,
        device=device,
    )

    # Submodular
    smdl = MLLMSubModularExplanationVision(
        Qwen,
        lambda1=args.lambda1,
        lambda2=args.lambda2
    )
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
        
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    # visualization_root_path = os.path.join(save_dir, "vis")
    # mkdir(visualization_root_path)
    
    end = args.end
    if end == -1:
        end = None
    select_contents = select_rank_contents(
        contents, args.begin, end, rank, world_size
    )
    print(
        "rank={} assigned_images={}".format(rank, len(select_contents)),
        flush=True,
    )
    
    for content in tqdm(select_contents, desc="rank {} images".format(rank)):
        json_output_path = os.path.join(
            save_json_root_path, content["image_path"].replace(".jpg", ".json")
        )
        npy_output_path = os.path.join(
            save_npy_root_path, content["image_path"].replace(".jpg", ".npy")
        )

        if outputs_are_complete(json_output_path, npy_output_path):
            tqdm.write("Skipping completed image: {}".format(content["image_path"]))
            continue

        image_start_time = time.perf_counter()
        
        image_path = os.path.join(args.Datasets, content["image_path"])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Data proccessing
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)    # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])

        selected_interpretation_token_id = [content["target_generated_index"]]
        selected_interpretation_token_word_id = [content["target_generated_id"]]

        # The original EAGLE manifests store a previously generated full token
        # sequence. The audited COCO-caption benchmark instead stores the exact
        # caption and its caption-relative target index. Build the equivalent
        # teacher-forced sequence from the current image prompt so visual-token
        # counts and positions match this processor run.
        if "selected_coco_caption" in content:
            output_word_id = tokenizer(
                content["selected_coco_caption"],
                add_special_tokens=False,
            )["input_ids"]
            target_index = int(content["target_generated_index"])
            if not 0 <= target_index < len(output_word_id):
                raise IndexError(
                    "target_generated_index={} outside caption token range {}"
                    .format(target_index, len(output_word_id))
                )
            if int(output_word_id[target_index]) != int(content["target_generated_id"]):
                raise ValueError(
                    "Stored target token id does not match selected COCO caption"
                )
            prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
            generated_ids = prompt_ids + [int(value) for value in output_word_id]
        else:
            generated_ids = content["generated_ids"]
            output_word_id = content["output_word_id"]

        Qwen.generated_ids = torch.tensor([generated_ids], dtype=torch.long).to(model.device).detach()
        Qwen.target_token_position = np.array(selected_interpretation_token_id) + len(inputs['input_ids'][0])
        Qwen.selected_interpretation_token_word_id = selected_interpretation_token_word_id
    
        image = cv2.imread(image_path)

        # Sub-region division
        region_size = int((image.shape[0] * image.shape[1] / args.division_number) ** 0.5)
        V_set = SubRegionDivision(image, mode=args.superpixel_algorithm, region_size = region_size)
        
        S_set, saved_json_file = smdl(image, V_set)
        saved_json_file["selected_interpretation_token_id"] = selected_interpretation_token_id
        saved_json_file["selected_interpretation_token_word_id"] = selected_interpretation_token_word_id
        saved_json_file["select_category"] = content["select_category"]
        saved_json_file["words"] = content["target_generated_token"]
         
        saved_json_file["location"] = content["location"]
        saved_json_file["segmentation"] = content["segmentation"]
        
        saved_json_file["output_word_id"] = output_word_id
        saved_json_file["output_words"] = processor.batch_decode(
            output_word_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Save npy file atomically
        atomic_save_npy(npy_output_path, np.array(S_set))
        saved_json_file["runtime_seconds"] = time.perf_counter() - image_start_time
        saved_json_file["worker_rank"] = rank
        saved_json_file["attention_implementation"] = attention_implementation

        # Save json file atomically; this is the completion marker.
        atomic_save_json(json_output_path, saved_json_file)
        tqdm.write(
            "rank {} completed {} in {:.2f} seconds".format(
                rank,
                content["image_path"],
                saved_json_file["runtime_seconds"],
            )
        )
    
if __name__ == "__main__":
    args = parse_args()
    
    main(args)
