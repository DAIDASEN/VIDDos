#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Sponge Attack Script for LanguageBind/Video-LLaVA-7B-hf
Optimized for 5D visual tensor sequences (batch, num_frames, channels, height, width).
"""

import argparse
import os
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import av
from tqdm import tqdm
from transformers import AutoProcessor, VideoLlavaForConditionalGeneration

# ==============================================================================
# Utils
# ==============================================================================
def expanduser(p: str) -> str:
    return os.path.expanduser(p)

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def now() -> float:
    return time.perf_counter()

def load_qa_dataset(dataset_dir: str) -> List[Dict[str, str]]:
    json_path = os.path.join(dataset_dir, "QA.json")
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data

def decode_text(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def split_train_eval_items(all_items: List[Dict[str, Any]], n_train: int, n_eval: int, seed: int):
    rng = random.Random(seed)
    total = min(n_train + n_eval, len(all_items))
    picked = rng.sample(all_items, k=total)
    train = picked[:min(n_train, len(picked))]
    eval_list = picked[len(train):len(train) + n_eval]
    return train, eval_list

def extract_assistant(txt: str) -> str:
    if "ASSISTANT:" in txt: return txt.split("ASSISTANT:", 1)[1].strip()
    return txt.strip()

def load_video(video_path: str, num_frames: int = 8) -> Optional[np.ndarray]:
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        if total_frames is None or total_frames <= 0:
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i >= 2000: break
                frames.append(frame.to_ndarray(format="rgb24"))
            container.close()
            if len(frames) < num_frames: return None
            idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            return np.stack([frames[i] for i in idx]).astype(np.uint8)

        base = np.linspace(0, total_frames - 1, num_frames)
        indices = np.clip(np.round(base), 0, total_frames - 1).astype(int)
        idx_set = set(indices.tolist())
        start_i, end_i = int(indices.min()), int(indices.max())

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_i: break
            if i >= start_i and i in idx_set:
                frames.append(frame.to_ndarray(format="rgb24"))
        container.close()

        if len(frames) != num_frames: return None
        return np.stack(frames).astype(np.uint8)
    except Exception:
        return None

# ==============================================================================
# Timing Definition
# ==============================================================================
@dataclass
class Timing:
    preprocess_s: float = 0.0
    gen_before_s: float = 0.0
    apply_attack_s: float = 0.0
    gen_after_s: float = 0.0
    gen_before_new_tokens: int = 0
    gen_after_new_tokens: int = 0
    @property
    def total_before_s(self) -> float: return self.preprocess_s + self.gen_before_s
    @property
    def total_after_s(self) -> float: return self.preprocess_s + self.apply_attack_s + self.gen_after_s
    @property
    def overhead_s(self) -> float: return self.total_after_s - self.total_before_s

def stats(xs: List[float]) -> Dict[str, float]:
    if not xs: return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    arr = np.array(xs, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}

# ==============================================================================
# Video-LLaVA Specific Trainer
# ==============================================================================
class VideoLlavaUniversalSpongeTrainer:
    def __init__(self, model, processor, device, args):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device
        self.args = args

        # Video-LLaVA 使用 CLIP 视觉编码器，对应标准的 CLIP Normalization
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
        
        self.norm_eps = args.eps / 255.0
        self.norm_alpha = args.alpha / 255.0
        
        self.min_val = torch.tensor((0.0 - mean) / std, device=device, dtype=model.dtype)
        self.max_val = torch.tensor((1.0 - mean) / std, device=device, dtype=model.dtype)
        self.eps_tensor = torch.tensor(self.norm_eps / std, device=device, dtype=model.dtype)

    def _build_prompts_for_item(self, item: Dict[str, Any]) -> Tuple[str, str]:
        question = item["question"]
        base_sponge = item.get("SPONGE_TARGET", "Yes. Furthermore, the video shows an object ")
        target = base_sponge * self.args.sponge_multiplier

        # Video-LLaVA 原生的 Prompt 格式
        prompt_user = f"USER: <video>\n{question} ASSISTANT:"
        prompt_full = f"{prompt_user} {target}"
        return prompt_full, prompt_user

    def _process_inputs(self, prompt: str, frames_uint8: np.ndarray):
        # 传入视频列表 (HuggingFace 处理器会自动提取特征)
        batch = self.processor(text=prompt, videos=list(frames_uint8), return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def _init_params(self, frames_uint8: np.ndarray, dummy_prompt_user: str) -> Dict[str, torch.Tensor]:
        batch0 = self._process_inputs(dummy_prompt_user, frames_uint8)
        # Video-LLaVA 专用的视频像素键名
        pv_key = "pixel_values_videos" if "pixel_values_videos" in batch0 else "pixel_values"
        pv0 = batch0[pv_key].to(self.model.dtype)
        
        # 形状预期为 5D: (batch_size, num_frames, channels, height, width)
        B, T, C, H, W = pv0.shape
        ph, pw = self.args.patch_h, self.args.patch_w
        
        params: Dict[str, torch.Tensor] = {}
        mode = self.args.attack_mode

        if mode == "uap_delta":
            delta = torch.zeros((1, 1, C, H, W), device=self.device, dtype=pv0.dtype)
            delta.uniform_(-self.norm_eps, self.norm_eps)
            delta.requires_grad_(True)
            params["delta_u"] = delta
        elif mode in ["patch_delta", "patch_replace"]:
            patch = torch.zeros((C, ph, pw), device=self.device, dtype=pv0.dtype)
            if self.args.patch_init == "random" and mode == "patch_replace":
                for c in range(3):
                    patch[c].uniform_(float(self.min_val[c]), float(self.max_val[c]))
            else:
                patch.uniform_(-self.norm_eps, self.norm_eps)
            patch.requires_grad_(True)
            params["patch"] = patch

        return params

    def _apply_attack(self, pixel_clean: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        adv = pixel_clean.clone()
        B, T, C, H, W = adv.shape
        mode = self.args.attack_mode

        if mode == "uap_delta":
            adv = adv + params["delta_u"]
        elif mode in ["patch_delta", "patch_replace"]:
            patch = params["patch"]
            ph, pw = patch.shape[1], patch.shape[2]
            top, left = H - ph, W - pw  # 放置在右下角
            
            if mode == "patch_delta":
                # 广播到所有的 batch 和 frames
                adv[:, :, :, top:top+ph, left:left+pw] += patch.unsqueeze(0).unsqueeze(0)
            elif mode == "patch_replace":
                adv[:, :, :, top:top+ph, left:left+pw] = patch.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
                
        return adv

    def _project_params(self, params: Dict[str, torch.Tensor]):
        mode = self.args.attack_mode
        with torch.no_grad():
            if mode == "uap_delta":
                delta = params["delta_u"]
                for c in range(3):
                    delta[:, :, c, :, :] = torch.clamp(delta[:, :, c, :, :], -float(self.eps_tensor[c]), float(self.eps_tensor[c]))
            elif mode == "patch_delta":
                patch = params["patch"]
                for c in range(3):
                    patch[c, :, :] = torch.clamp(patch[c, :, :], -float(self.eps_tensor[c]), float(self.eps_tensor[c]))
            elif mode == "patch_replace":
                patch = params["patch"]
                for c in range(3):
                    patch[c, :, :] = torch.clamp(patch[c, :, :], float(self.min_val[c]), float(self.max_val[c]))

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, prompt_len: int) -> torch.Tensor:
        logits_shift = logits[:, :-1, :].contiguous()
        labels_shift = labels[:, 1:].contiguous()
        V = logits_shift.size(-1)

        per_tok = F.cross_entropy(logits_shift.view(-1, V), labels_shift.view(-1), reduction="none").view_as(labels_shift)
        mask = (labels_shift != -100).float()
        weights = mask.clone()

        # Sponge Forcing Loss
        start = max(0, prompt_len - 1)
        end = min(start + int(self.args.prefix_k), weights.shape[1])
        if end > start:
            weights[:, start:end] = weights[:, start:end] * float(self.args.prefix_weight)
            
        loss_ce = (per_tok * weights).sum() / (weights.sum() + 1e-6)
        
        # EOS Suppression Loss
        loss_eos = torch.tensor(0.0, device=self.device)
        if self.args.eos_lambda > 0.0 and hasattr(self.model.config, "eos_token_id"):
            eos_id = self.model.config.eos_token_id
            if eos_id is not None:
                eos_id_list = [eos_id] if isinstance(eos_id, int) else eos_id
                seq_len = logits_shift.shape[1]
                end_eos = min(start + int(self.args.prefix_k), seq_len)
                if end_eos > start:
                    target_logits = logits_shift[:, start:end_eos, :]
                    probs = F.softmax(target_logits, dim=-1)
                    eos_probs = probs[:, :, eos_id_list].sum(dim=-1)
                    loss_eos = eos_probs.mean() * float(self.args.eos_lambda)

        return loss_ce + loss_eos

    def train(self, train_items: List[Dict[str, Any]], dataset_dir: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        first_frames, dummy_prompt_user = None, None
        for item in train_items:
            vp = os.path.join(dataset_dir, item["video"])
            fr = load_video(vp, num_frames=self.args.num_frames)
            if fr is not None:
                first_frames = fr
                _, dummy_prompt_user = self._build_prompts_for_item(item)
                break
        
        params = self._init_params(first_frames, dummy_prompt_user)
        meta = {"attack_mode": self.args.attack_mode, "patch_h": self.args.patch_h, "patch_w": self.args.patch_w}

        self.model.train()
        t0 = now()
        rng = random.Random(self.args.seed + 999)

        for ep in range(self.args.uap_epochs):
            items_copy = list(train_items)
            rng.shuffle(items_copy)
            pbar = tqdm(items_copy, desc=f"Video-LLaVA Train epoch {ep+1}/{self.args.uap_epochs}")

            for item in pbar:
                vp = os.path.join(dataset_dir, item["video"])
                frames = load_video(vp, num_frames=self.args.num_frames)
                if frames is None: continue

                prompt_full, prompt_user = self._build_prompts_for_item(item)
                
                user_inputs = self._process_inputs(prompt_user, frames)
                prompt_len = int(user_inputs["input_ids"].shape[1])

                batch = self._process_inputs(prompt_full, frames)
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                pv_key = "pixel_values_videos" if "pixel_values_videos" in batch else "pixel_values"
                pixel_clean = batch[pv_key].to(self.model.dtype)

                model_kwargs = {k: v for k, v in batch.items() if k not in ["input_ids", "attention_mask", "labels", pv_key]}

                for _ in range(self.args.uap_iters_per_video):
                    for k in params:
                        if params[k].grad is not None: params[k].grad.zero_()

                    pixel_adv = self._apply_attack(pixel_clean, params)
                    model_kwargs[pv_key] = pixel_adv 

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        **model_kwargs
                    )
                    logits = outputs.logits.float()

                    loss = self._compute_loss(logits, labels, prompt_len)
                    loss.backward()

                    with torch.no_grad():
                        for k in params:
                            params[k].data = params[k].data - self.norm_alpha * params[k].grad.sign()
                        self._project_params(params)

                    pbar.set_postfix({"loss": f"{float(loss.detach().cpu().item()):.3f}"})
            
            ckpt_path = os.path.join(self.args.output_dir, "universal_params_latest.pt")
            with torch.no_grad():
                save_obj = {"attack_mode": self.args.attack_mode, "epoch": ep + 1}
                for k, v in params.items(): save_obj[k] = v.detach().cpu()
            torch.save(save_obj, ckpt_path)

        cuda_sync()
        meta["train_time_s"] = float(now() - t0)
        self.model.eval()
        return params, meta

    @torch.no_grad()
    def generate_with_new_tokens(self, item: Dict[str, Any], frames_uint8: np.ndarray, pixel_override: Optional[torch.Tensor] = None) -> Tuple[str, int]:
        _, prompt_user = self._build_prompts_for_item(item)
        inputs = self._process_inputs(prompt_user, frames_uint8)
        
        pv_key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
        if pixel_override is not None:
            inputs[pv_key] = pixel_override
                
        prompt_len = int(inputs["input_ids"].shape[1])
        out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        new_tokens = int(max(0, int(out.shape[1]) - prompt_len))
        return decode_text(self.tokenizer, out[0]), new_tokens

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="train_eval", choices=["train_eval", "eval_only"])
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--load-params", type=str, default=None)

    # 默认路径修改为 Video-LLaVA-7B
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B-hf")
    parser.add_argument("--num-train-videos", type=int, default=20)
    parser.add_argument("--num-eval-videos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=8) # Video-LLaVA default is often 8 frames
    parser.add_argument("--sponge-multiplier", type=int, default=15)
    
    parser.add_argument("--uap-epochs", type=int, default=10)
    parser.add_argument("--uap-iters-per-video", type=int, default=15)
    parser.add_argument("--eps", type=float, default=16.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--prefix-k", type=int, default=96)
    parser.add_argument("--prefix-weight", type=float, default=12.0)
    parser.add_argument("--eos-lambda", type=float, default=30.0)

    parser.add_argument("--attack-mode", type=str, default="patch_replace", choices=["uap_delta", "patch_delta", "patch_replace"])
    parser.add_argument("--patch-h", type=int, default=96)
    parser.add_argument("--patch-w", type=int, default=96)
    parser.add_argument("--patch-init", type=str, default="random", choices=["random", "zero"])
    
    args = parser.parse_args()
    seed_all(args.seed)

    dataset_dir = expanduser(args.dataset_dir)
    output_dir = expanduser(args.output_dir)
    safe_makedirs(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16

    print(f"[Init] Loading Video-LLaVA from: {args.model_path}")
    
    # Hugging Face 原生完美支持，直接使用 device_map="auto" 即可
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype=torch_dtype, 
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.eval()

    all_items = load_qa_dataset(dataset_dir)
    
    if args.stage == "eval_only":
        save_obj = torch.load(args.load_params, map_location="cpu")
        args.attack_mode = save_obj.get("attack_mode", args.attack_mode)
        params = {k: save_obj[k] for k in ["delta_u", "patch_delta", "patch"] if k in save_obj}
        _, eval_list = split_train_eval_items(all_items, 0, args.num_eval_videos, args.seed + 1337)
        train_list = []
    else:
        train_list, eval_list = split_train_eval_items(all_items, args.num_train_videos, args.num_eval_videos, args.seed)
        if len(eval_list) == 0: eval_list = train_list

    trainer = VideoLlavaUniversalSpongeTrainer(model, processor, device, args)

    if args.stage != "eval_only":
        print(f"[Train] mode={args.attack_mode}, train_videos={len(train_list)}")
        params, _ = trainer.train(train_list, dataset_dir)
        save_obj = {"attack_mode": args.attack_mode}
        for k, v in params.items(): save_obj[k] = v.detach().cpu()
        torch.save(save_obj, os.path.join(output_dir, "universal_params.pt"))

    summary: Dict[str, Any] = {}
    timings, before_new_tokens_list, after_new_tokens_list = [], [], []

    pbar = tqdm(eval_list, desc="Video-LLaVA Eval")
    for item in pbar:
        vid = os.path.splitext(os.path.basename(item["video"]))[0]
        vdir = os.path.join(output_dir, "eval", vid)
        safe_makedirs(vdir)
        
        frames = load_video(os.path.join(dataset_dir, item["video"]), num_frames=args.num_frames)
        if frames is None: continue

        tinfo = Timing()
        
        cuda_sync(); t0 = now()
        raw_before, b_toks = trainer.generate_with_new_tokens(item, frames, None)
        cuda_sync(); tinfo.gen_before_s, tinfo.gen_before_new_tokens = float(now() - t0), b_toks

        batch = trainer._process_inputs(item["question"], frames)
        pv_key = "pixel_values_videos" if "pixel_values_videos" in batch else "pixel_values"
        pixel_clean = batch[pv_key].to(model.dtype)
        pixel_adv = trainer._apply_attack(pixel_clean, {k: v.to(device) for k, v in params.items()})

        cuda_sync(); t4 = now()
        raw_after, a_toks = trainer.generate_with_new_tokens(item, frames, pixel_adv)
        cuda_sync(); tinfo.gen_after_s, tinfo.gen_after_new_tokens = float(now() - t4), a_toks
        
        rec = {
            "video_path": item["video"], "before_new_tokens": b_toks, "after_new_tokens": a_toks,
            "gen_before_s": tinfo.gen_before_s, "gen_after_s": tinfo.gen_after_s,
        }
        summary[vid] = rec
        timings.append(tinfo)
        before_new_tokens_list.append(b_toks)
        after_new_tokens_list.append(a_toks)
        
        with open(os.path.join(vdir, "timing.json"), "w", encoding="utf-8") as f: json.dump(asdict(tinfo), f, indent=2)

    with open(os.path.join(output_dir, "final_summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump([summary[k] for k in sorted(summary.keys())], f, indent=2)

    timing_summary = {
        "stats_before_new_tokens": stats(before_new_tokens_list),
        "stats_after_new_tokens": stats(after_new_tokens_list),
        "stats_overhead_s": stats([t.overhead_s for t in timings]),
    }
    with open(os.path.join(output_dir, "timing_summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"\nDone! Video-LLaVA Eval processed: {len(timings)}. Outputs in {output_dir}")

if __name__ == "__main__":
    main()
