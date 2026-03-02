#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Sponge Attack Script for Qwen3-VL
Optimized for Flattened 2D visual tensor sequences.
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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}")
    # 使用 utf-8-sig 自动处理可能存在的 BOM 隐形字符
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data

def apply_chat_template(processor, conv: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(conv, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass
    
    # Qwen fallback if template fails
    lines: List[str] = []
    for msg in conv:
        role = str(msg.get("role", "")).upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for it in content:
                if isinstance(it, dict) and it.get("type") == "video":
                    parts.append("<|video_pad|>") # Qwen specific video token placeholder
                elif isinstance(it, dict) and it.get("type") == "text":
                    parts.append(str(it.get("text", "")))
            text = "\n".join([p for p in parts if p])
        else:
            text = str(content)
        lines.append(f"{role}: {text}".strip())
    if add_generation_prompt:
        lines.append("ASSISTANT:")
    return "\n".join(lines).strip()

def decode_text(processor, token_ids: torch.Tensor) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(token_ids, skip_special_tokens=True)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "decode"):
        return tok.decode(token_ids, skip_special_tokens=True)
    return str(token_ids)

def get_pixel_values_key(batch: Dict[str, Any]) -> str:
    if "pixel_values_videos" in batch:
        return "pixel_values_videos"
    if "pixel_values" in batch:
        return "pixel_values"
    if "image_embeds" in batch:
        return "image_embeds"
    raise KeyError(f"Cannot find pixel values in batch keys: {list(batch.keys())}")

def split_train_eval_items(all_items: List[Dict[str, Any]], n_train: int, n_eval: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    total = min(n_train + n_eval, len(all_items))
    picked = rng.sample(all_items, k=total)
    train = picked[:min(n_train, len(picked))]
    eval_list = picked[len(train):len(train) + n_eval]
    return train, eval_list

def extract_assistant(txt: str) -> str:
    if "ASSISTANT:" in txt:
        return txt.split("ASSISTANT:", 1)[1].strip()
    return txt.strip()

def load_video(video_path: str, num_frames: int = 16, random_jitter: bool = False, jitter: int = 3) -> Optional[np.ndarray]:
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames

        if total_frames is None or total_frames <= 0:
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i >= 4000: break
                frames.append(frame.to_ndarray(format="rgb24"))
            container.close()
            if len(frames) < num_frames: return None
            idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            return np.stack([frames[i] for i in idx]).astype(np.uint8)

        base = np.linspace(0, total_frames - 1, num_frames)
        if random_jitter and jitter > 0:
            base = base + np.random.randint(-jitter, jitter + 1, size=num_frames)
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

def stats(xs: List[float]) -> Dict[str, float]:
    if not xs: return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    arr = np.array(xs, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95))}

def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if den != 0 else default

def count_text_tokens(tokenizer, text: str) -> int:
    try:
        return int(len(tokenizer(text, add_special_tokens=False).input_ids))
    except Exception:
        return 0

# ==============================================================================
# Timing
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

# ==============================================================================
# Qwen3-VL Specific Trainer
# ==============================================================================
class QwenUniversalSpongeTrainer:
    def __init__(self, model, processor, device, args):
        self.model = model
        self.processor = processor
        self.device = device
        self.args = args

        ip = getattr(processor, "image_processor", None)
        mean = getattr(ip, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        std = getattr(ip, "image_std", [0.26862954, 0.26130258, 0.27577711])
        
        self.norm_eps = args.eps / 255.0
        self.norm_alpha = args.alpha / 255.0
        self.min_val = (0.0 - np.mean(mean)) / np.mean(std)
        self.max_val = (1.0 - np.mean(mean)) / np.mean(std)

        # ----------------------------------------------------
        # 🟢 新增这两行：确保在 eval_only 模式下也能知道 Patch 的空间维度
        self.p_h = max(1, self.args.patch_h // 14)
        self.p_w = max(1, self.args.patch_w // 14)
        # ----------------------------------------------------

        self.banned_ids = []
        eos_id = getattr(model.config, "eos_token_id", None)
        if eos_id is not None: self.banned_ids.append(int(eos_id))
        tok = getattr(processor, "tokenizer", None)
        if tok:
            for c in ["No", "No.", " No", "no", "Yes", "Yes.", " Yes", "yes"]:
                ids = tok(c, add_special_tokens=False).input_ids
                if ids: self.banned_ids.extend(ids)
        self.banned_ids = sorted(list(set(self.banned_ids)))

    def _build_prompts_for_item(self, item: Dict[str, Any]) -> Tuple[str, str]:
        question = item["question"]
        base_sponge = item.get("SPONGE_TARGET", "Yes. Furthermore, the video shows an object ")
        target = base_sponge * self.args.sponge_multiplier

        conv_target = [
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]
        prompt_full = apply_chat_template(self.processor, conv_target, add_generation_prompt=False)

        conv_user = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        prompt_user = apply_chat_template(self.processor, conv_user, add_generation_prompt=True)
        return prompt_full, prompt_user

    def _get_prompt_len(self, prompt_user: str, frames_uint8: np.ndarray) -> int:
        batch = self.processor(text=prompt_user, videos=[list(frames_uint8)], return_tensors="pt").to(self.device)
        return int(batch["input_ids"].shape[1])

    def _init_params(self, frames_uint8: np.ndarray, dummy_prompt_user: str) -> Dict[str, torch.Tensor]:
        batch0 = self.processor(text=dummy_prompt_user, videos=[list(frames_uint8)], return_tensors="pt").to(self.device)
        pv_key = get_pixel_values_key(batch0)
        pv0 = batch0[pv_key].to(self.model.dtype)
        
        # Qwen3-VL processes video into a flattened 2D tensor: (Total_Patches, Patch_Dim)
        if pv0.dim() != 2:
            raise ValueError(f"Expected Qwen3-VL to output 2D pixel_values, but got {pv0.dim()}D tensor.")
            
        patch_dim = pv0.shape[-1]
        
        # 计算替换的 Patch 数量 (近似模拟右下角 96x96 的空间补丁)
        # 假设 Qwen spatial patch size 默认为 14
        p_h = max(1, self.args.patch_h // 14)
        p_w = max(1, self.args.patch_w // 14)
        num_patches = p_h * p_w

        params: Dict[str, torch.Tensor] = {}
        mode = self.args.attack_mode

        if mode == "uap_delta":
            delta = torch.zeros((1, patch_dim), device=self.device, dtype=pv0.dtype)
            delta.uniform_(-self.norm_eps, self.norm_eps)
            delta.requires_grad_(True)
            params["delta_u"] = delta
        elif mode in ["patch_delta", "patch_replace"]:
            patch = torch.zeros((num_patches, patch_dim), device=self.device, dtype=pv0.dtype)
            if self.args.patch_init == "random" and mode == "patch_replace":
                patch.uniform_(self.min_val, self.max_val)
            else:
                patch.uniform_(-self.norm_eps, self.norm_eps)
            patch.requires_grad_(True)
            params["patch"] = patch

        return params

    def _apply_attack(self, pixel_clean: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        adv = pixel_clean.clone()
        mode = self.args.attack_mode

        if mode == "uap_delta":
            adv = adv + params["delta_u"]
        elif mode == "patch_delta":
            p_len = params["patch"].shape[0]
            adv[-p_len:] = adv[-p_len:] + params["patch"] # 加在特征序列最后（代表最后几帧的角落）
        elif mode == "patch_replace":
            p_len = params["patch"].shape[0]
            adv[-p_len:] = params["patch"] # 替换特征序列最后
            
        return adv

    def _project_params(self, params: Dict[str, torch.Tensor]):
        mode = self.args.attack_mode
        if mode == "uap_delta":
            params["delta_u"].data = torch.clamp(params["delta_u"].data, -self.norm_eps, self.norm_eps)
        elif mode == "patch_delta":
            params["patch"].data = torch.clamp(params["patch"].data, -self.norm_eps, self.norm_eps)
        elif mode == "patch_replace":
            params["patch"].data = torch.clamp(params["patch"].data, self.min_val, self.max_val)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, prompt_len: int) -> torch.Tensor:
        logits_shift = logits[:, :-1, :].contiguous()
        labels_shift = labels[:, 1:].contiguous()
        V = logits_shift.size(-1)

        per_tok = F.cross_entropy(logits_shift.view(-1, V), labels_shift.view(-1), reduction="none").view_as(labels_shift)
        mask = (labels_shift != -100).float()
        weights = mask.clone()

        start = max(0, prompt_len - 1)
        end = min(start + int(self.args.prefix_k), weights.shape[1])
        if end > start:
            weights[:, start:end] = weights[:, start:end] * float(self.args.prefix_weight)

        loss_ce = (per_tok * weights).sum() / (weights.sum() + 1e-6)

        eos_id = getattr(self.model.config, "eos_token_id", None)
        loss_eos = 0.0
        if eos_id is not None and end > start and float(self.args.eos_lambda) > 0:
            probs = F.softmax(torch.clamp(logits_shift[0, start:end, :], -1000, 1000), dim=-1)
            p_eos = probs[:, int(eos_id)].clamp(1e-9, 1.0 - 1e-9)
            loss_eos = (-torch.log(1.0 - p_eos)).mean()

        loss_ban = 0.0
        if float(self.args.ban_first_token_lambda) > 0 and len(self.banned_ids) > 0:
            p0 = F.softmax(torch.clamp(logits_shift[0, start, :], -1000, 1000), dim=-1)
            banned_mass = sum(p0[int(bid)] for bid in self.banned_ids)
            loss_ban = banned_mass

        return loss_ce + float(self.args.eos_lambda) * loss_eos + float(self.args.ban_first_token_lambda) * loss_ban

    def train(self, train_items: List[Dict[str, Any]], dataset_dir: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        first_frames, dummy_prompt_user = None, None
        for item in train_items:
            vp = os.path.join(dataset_dir, item["video"])
            fr = load_video(vp, num_frames=self.args.num_frames)
            if fr is not None:
                first_frames = fr
                _, dummy_prompt_user = self._build_prompts_for_item(item)
                break
        if first_frames is None: raise RuntimeError("All training videos failed to decode.")

        params = self._init_params(first_frames, dummy_prompt_user)

        # 断点续跑恢复权重
        if self.args.resume_from and os.path.exists(self.args.resume_from):
            print(f"[*] Resuming Qwen3 from checkpoint: {self.args.resume_from}")
            ckpt = torch.load(self.args.resume_from, map_location="cpu")
            for k in params.keys():
                if k in ckpt: params[k].data = ckpt[k].to(self.device).data

        meta = {"attack_mode": self.args.attack_mode, "patch_h": self.args.patch_h, "patch_w": self.args.patch_w}

        self.model.train()
        t0 = now()
        rng = random.Random(self.args.seed + 999)

        for ep in range(self.args.uap_epochs):
            items_copy = list(train_items)
            rng.shuffle(items_copy)
            pbar = tqdm(items_copy, desc=f"Train {self.args.attack_mode} epoch {ep+1}/{self.args.uap_epochs}")

            for item in pbar:
                vp = os.path.join(dataset_dir, item["video"])
                frames = load_video(vp, num_frames=self.args.num_frames, random_jitter=self.args.train_random_jitter, jitter=self.args.jitter)
                if frames is None: continue

                prompt_full, prompt_user = self._build_prompts_for_item(item)
                prompt_len = self._get_prompt_len(prompt_user, frames)

                batch = self.processor(text=prompt_full, videos=[list(frames)], return_tensors="pt").to(self.device)
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                
                # Retrieve Qwen specific vision kwargs
                vision_kwargs = {k: v for k, v in batch.items() if k not in ["input_ids", "attention_mask", "labels"]}

                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                pv_key = get_pixel_values_key(batch)
                pixel_clean = batch[pv_key].to(self.model.dtype)

                for _ in range(self.args.uap_iters_per_video):
                    for k in params:
                        if params[k].grad is not None: params[k].grad.zero_()

                    pixel_adv = self._apply_attack(pixel_clean, params)
                    vision_kwargs[pv_key] = pixel_adv # Replace with adversarial tensor

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        **vision_kwargs
                    )
                    logits = outputs.logits.float()

                    loss = self._compute_loss(logits, labels, prompt_len)
                    loss.backward()

                    with torch.no_grad():
                        for k in params:
                            params[k].data = params[k].data - self.norm_alpha * params[k].grad.sign()
                        self._project_params(params)

                    pbar.set_postfix({"loss": float(loss.detach().cpu().item())})
            
            # 实时保存
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
        inputs = self.processor(text=prompt_user, videos=[list(frames_uint8)], return_tensors="pt").to(self.device)
        
        if pixel_override is not None:
            pv_key = get_pixel_values_key(inputs)
            inputs[pv_key] = pixel_override
                
        prompt_len = int(inputs["input_ids"].shape[1])
        out = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False)
        new_tokens = int(max(0, int(out.shape[1]) - prompt_len))
        return decode_text(self.processor, out[0]), new_tokens

    @torch.no_grad()
    def build_adv_pixels_for_eval(self, item: Dict[str, Any], frames_uint8: np.ndarray, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, prompt_user = self._build_prompts_for_item(item)
        batch = self.processor(text=prompt_user, videos=[list(frames_uint8)], return_tensors="pt").to(self.device)
        pv_key = get_pixel_values_key(batch)
        pixel_clean = batch[pv_key].to(self.model.dtype)
        pixel_adv = self._apply_attack(pixel_clean, params)
        pixel_adv = torch.clamp(pixel_adv, self.min_val, self.max_val)
        return pixel_adv

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="train_eval", choices=["train_eval", "eval_only"])
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--load-params", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--num-train-videos", type=int, default=20)
    parser.add_argument("--num-eval-videos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sponge-multiplier", type=int, default=15)
    
    parser.add_argument("--uap-epochs", type=int, default=10)
    parser.add_argument("--uap-iters-per-video", type=int, default=15)
    parser.add_argument("--eps", type=float, default=16.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--prefix-k", type=int, default=96)
    parser.add_argument("--prefix-weight", type=float, default=12.0)
    parser.add_argument("--eos-lambda", type=float, default=30.0)
    parser.add_argument("--ban-first-token-lambda", dest="ban_first_token_lambda", type=float, default=20.0)

    parser.add_argument("--attack-mode", type=str, default="patch_replace", choices=["uap_delta", "patch_delta", "patch_replace"])
    parser.add_argument("--patch-h", type=int, default=96)
    parser.add_argument("--patch-w", type=int, default=96)
    parser.add_argument("--patch-init", type=str, default="random", choices=["random", "zero"])
    
    parser.add_argument("--train-random-jitter", action="store_true")
    parser.add_argument("--jitter", type=int, default=3)
    
    args = parser.parse_args()
    seed_all(args.seed)

    dataset_dir = expanduser(args.dataset_dir)
    output_dir = expanduser(args.output_dir)
    safe_makedirs(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[Init] device={device}, dtype={args.dtype}")
    print(f"[Init] Loading Qwen3-VL from: {args.model_path}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
    if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    model.eval()

    all_items = load_qa_dataset(dataset_dir)
    print(f"[Dataset] Loaded {len(all_items)} items from {os.path.join(dataset_dir, 'QA.json')}")

    params: Dict[str, torch.Tensor]
    train_list: List[Dict[str, Any]]
    eval_list: List[Dict[str, Any]]

    if args.stage == "eval_only":
        if not args.load_params: raise ValueError("--load-params required for eval_only")
        save_obj = torch.load(args.load_params, map_location="cpu")
        args.attack_mode = save_obj.get("attack_mode", args.attack_mode)
        params = {k: save_obj[k] for k in ["delta_u", "patch_delta", "patch"] if k in save_obj}
        _, eval_list = split_train_eval_items(all_items, 0, args.num_eval_videos, args.seed + 1337)
        train_list = []
    else:
        train_list, eval_list = split_train_eval_items(all_items, args.num_train_videos, args.num_eval_videos, args.seed)

    with open(os.path.join(output_dir, "train_items.json"), "w", encoding="utf-8") as f: json.dump(train_list, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "eval_items.json"), "w", encoding="utf-8") as f: json.dump(eval_list, f, indent=2, ensure_ascii=False)

    trainer = QwenUniversalSpongeTrainer(model, processor, device, args)
    params_path = args.load_params

    if args.stage != "eval_only":
        print(f"[Train] mode={args.attack_mode}, train_videos={len(train_list)}")
        params, _ = trainer.train(train_list, dataset_dir)
        save_obj = {"attack_mode": args.attack_mode}
        for k, v in params.items(): save_obj[k] = v.detach().cpu()
        params_path = os.path.join(output_dir, "universal_params.pt")
        torch.save(save_obj, params_path)

    summary: Dict[str, Any] = {}
    timings, corrupt, before_new_tokens_list, after_new_tokens_list = [], [], [], []

    pbar = tqdm(eval_list, desc="Eval (before/after)")
    for item in pbar:
        vid = os.path.splitext(os.path.basename(item["video"]))[0]
        vdir = os.path.join(output_dir, "eval", vid)
        safe_makedirs(vdir)
        
        vpath = os.path.join(dataset_dir, item["video"])
        t_pre0 = now()
        frames = load_video(vpath, num_frames=args.num_frames)
        t_pre1 = now()
        
        if frames is None:
            corrupt.append(item["video"])
            continue

        tinfo = Timing(preprocess_s=float(t_pre1 - t_pre0))

        cuda_sync(); t0 = now()
        raw_before, b_toks = trainer.generate_with_new_tokens(item, frames, None)
        cuda_sync(); t1 = now()
        tinfo.gen_before_s, tinfo.gen_before_new_tokens = float(t1 - t0), b_toks

        cuda_sync(); t2 = now()
        pixel_adv = trainer.build_adv_pixels_for_eval(item, frames, {k: v.to(device) for k, v in params.items()})
        cuda_sync(); t3 = now()
        tinfo.apply_attack_s = float(t3 - t2)

        cuda_sync(); t4 = now()
        raw_after, a_toks = trainer.generate_with_new_tokens(item, frames, pixel_adv)
        cuda_sync(); t5 = now()
        tinfo.gen_after_s, tinfo.gen_after_new_tokens = float(t5 - t4), a_toks

        before_ans, after_ans = extract_assistant(raw_before), extract_assistant(raw_after)
        
        rec = {
            "video_path": item["video"], "before_len": len(before_ans), "after_len": len(after_ans),
            "before_new_tokens": b_toks, "after_new_tokens": a_toks,
            "gen_before_s": tinfo.gen_before_s, "gen_after_s": tinfo.gen_after_s,
            "response_before": before_ans, "response_after": after_ans,
        }
        summary[vid] = rec
        timings.append(tinfo)
        before_new_tokens_list.append(b_toks)
        after_new_tokens_list.append(a_toks)
        
        with open(os.path.join(vdir, "log.json"), "w", encoding="utf-8") as f: json.dump(rec, f, indent=2, ensure_ascii=False)
        with open(os.path.join(vdir, "timing.json"), "w", encoding="utf-8") as f: json.dump(asdict(tinfo), f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "final_summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump([summary[k] for k in sorted(summary.keys())], f, indent=2, ensure_ascii=False)

    timing_summary = {
        "n_eval_processed": len(timings), "n_eval_corrupt": len(corrupt),
        "stats_before_new_tokens": stats([float(x) for x in before_new_tokens_list]),
        "stats_after_new_tokens": stats([float(x) for x in after_new_tokens_list]),
        "stats_overhead_s": stats([t.overhead_s for t in timings]),
    }
    with open(os.path.join(output_dir, "timing_summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Qwen3 Eval processed: {len(timings)}. Outputs in {output_dir}")

if __name__ == "__main__":
    main()