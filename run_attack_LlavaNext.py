#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ==============================================================================
# Constants
# ==============================================================================
# CLIP normalize used by LLaVA-NeXT (You might need to adjust for Qwen/InternVL if strictly needed)
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


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
    """Loads QA.json from the dataset directory."""
    json_path = os.path.join(dataset_dir, "QA.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}")
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data

def apply_chat_template(processor, conv: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(conv, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass

    lines: List[str] = []
    for msg in conv:
        role = str(msg.get("role", "")).upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for it in content:
                if isinstance(it, dict) and it.get("type") == "video":
                    parts.append("<video>")
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
        try:
            return processor.decode(token_ids, skip_special_tokens=True)
        except Exception:
            pass
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "decode"):
        return tok.decode(token_ids, skip_special_tokens=True)
    return str(token_ids)

def get_image_mean_std(processor) -> Tuple[List[float], List[float]]:
    ip = getattr(processor, "image_processor", None)
    mean = getattr(ip, "image_mean", None)
    std = getattr(ip, "image_std", None)
    if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)) and len(mean) == 3 and len(std) == 3:
        return list(mean), list(std)
    return OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

def get_pixel_values_key(batch: Dict[str, Any]) -> str:
    if "pixel_values_videos" in batch:
        return "pixel_values_videos"
    if "pixel_values" in batch:
        return "pixel_values"
    if "image_embeds" in batch:
        return "image_embeds"
    raise KeyError(f"Cannot find pixel values in batch keys: {list(batch.keys())}")

def load_video_llm_backend(model_path: str, model_family: str, torch_dtype):
    if model_family == "llava_next_video":
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        processor = LlavaNextVideoProcessor.from_pretrained(model_path, use_fast=True)
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto"
        )
        return model, processor

    if model_family == "video_llava":
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
        processor = VideoLlavaProcessor.from_pretrained(model_path, use_fast=True)
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto"
        )
        return model, processor

    if model_family == "qwen_vl":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        return model, processor

    if model_family == "intern_vl":
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # InternVL typically uses AutoModel rather than AutoModelForCausalLM
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        return model, processor

    raise ValueError(f"Unknown --model-family: {model_family}")

def split_train_eval_items(all_items: List[Dict[str, Any]],
                           n_train: int,
                           n_eval: int,
                           seed: int,
                           eval_from_train: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    if eval_from_train:
        train = rng.sample(all_items, k=min(n_train, len(all_items)))
        rng2 = random.Random(seed + 1337)
        eval_list = rng2.sample(train, k=min(n_eval, len(train)))
        return train, eval_list
    total = min(n_train + n_eval, len(all_items))
    picked = rng.sample(all_items, k=total)
    train = picked[:min(n_train, len(picked))]
    eval_list = picked[len(train):len(train) + n_eval]
    return train, eval_list

def extract_assistant(txt: str) -> str:
    if "ASSISTANT:" in txt:
        return txt.split("ASSISTANT:", 1)[1].strip()
    return txt.strip()

def load_video(video_path: str,
               num_frames: int = 16,
               random_jitter: bool = False,
               jitter: int = 3,
               max_decode_frames_fallback: int = 4000) -> Optional[np.ndarray]:
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames

        if total_frames is None or total_frames <= 0:
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_decode_frames_fallback:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
            container.close()
            if len(frames) < num_frames:
                return None
            idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            picked = [frames[i] for i in idx]
            return np.stack(picked).astype(np.uint8)

        base = np.linspace(0, total_frames - 1, num_frames)
        if random_jitter and jitter > 0:
            base = base + np.random.randint(-jitter, jitter + 1, size=num_frames)
        indices = np.clip(np.round(base), 0, total_frames - 1).astype(int)
        idx_set = set(indices.tolist())
        start_i, end_i = int(indices.min()), int(indices.max())

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_i:
                break
            if i >= start_i and i in idx_set:
                frames.append(frame.to_ndarray(format="rgb24"))
        container.close()

        if len(frames) != num_frames:
            return None
        return np.stack(frames).astype(np.uint8)
    except Exception:
        return None

def save_video(frames_uint8: np.ndarray, output_path: str, fps: int = 10):
    h, w = int(frames_uint8.shape[1]), int(frames_uint8.shape[2])
    if h % 2 != 0: h -= 1
    if w % 2 != 0: w -= 1
    frames_uint8 = frames_uint8[:, :h, :w, :]

    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h

    for fr in frames_uint8:
        av_fr = av.VideoFrame.from_ndarray(fr, format="rgb24")
        for packet in stream.encode(av_fr):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

def denorm_video_to_uint8(pixel_values_videos_norm: torch.Tensor,
                          mean: torch.Tensor,
                          std: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        x = pixel_values_videos_norm
        if x.dim() == 5:
            x = x.squeeze(0)
        x = x.float()
        x = x * std + mean
        x = x.clamp(0.0, 1.0)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = (x * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return x

def stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    arr = np.array(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }

def safe_div(num: float, den: float, default: float = 0.0) -> float:
    try:
        if den == 0:
            return default
        return float(num) / float(den)
    except Exception:
        return default

def count_text_tokens(tokenizer, text: str) -> int:
    try:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        return int(len(ids))
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
    def total_before_s(self) -> float:
        return self.preprocess_s + self.gen_before_s

    @property
    def total_after_s(self) -> float:
        return self.preprocess_s + self.apply_attack_s + self.gen_after_s

    @property
    def overhead_s(self) -> float:
        return self.total_after_s - self.total_before_s


# ==============================================================================
# Core: Universal Trainer (UAP / Patch)
# ==============================================================================
class UniversalSpongeTrainer:
    def __init__(self, model, processor, device, args):
        self.model = model
        self.processor = processor
        self.device = device
        self.args = args

        image_mean, image_std = get_image_mean_std(processor)
        self.image_mean = image_mean
        self.image_std = image_std

        self.mean = torch.tensor(image_mean, device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor(image_std,  device=device).view(1, 3, 1, 1)
        self.std_bc = self.std.view(1, 1, 3, 1, 1)

        self.norm_eps   = (args.eps / 255.0) / self.std_bc
        self.norm_alpha = (args.alpha / 255.0) / self.std_bc

        self.norm_min = ((0.0 - self.mean) / self.std).view(1, 1, 3, 1, 1)
        self.norm_max = ((1.0 - self.mean) / self.std).view(1, 1, 3, 1, 1)

        self.banned_ids = []
        eos_id = getattr(model.config, "eos_token_id", None)
        if eos_id is not None:
            self.banned_ids.append(int(eos_id))
        for c in ["No", "No.", " No", "no", "Yes", "Yes.", " Yes", "yes"]:
            ids = processor.tokenizer(c, add_special_tokens=False).input_ids
            if ids:
                self.banned_ids.extend(ids)
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
        
        # Adaptation for InternVL / Qwen returning 4D (T, C, H, W) instead of 5D (1, T, C, H, W)
        if pv0.dim() == 4:
            pv0 = pv0.unsqueeze(0)
        elif pv0.dim() == 2:
            print("[Warning] Flattened 2D pixel_values detected. Spatial patching might require specific mapping for this model.")
            # Create a dummy 5D tensor just to initialize parameters (actual application will need mapping)
            _, T, _, H, W = 1, self.args.num_frames, 3, 224, 224
        else:
            _, T, _, H, W = pv0.shape

        params: Dict[str, torch.Tensor] = {}
        mode = self.args.attack_mode

        if mode == "uap_delta":
            if self.args.delta_mode == "shared_time":
                delta = torch.zeros((1, 1, 3, H, W), device=self.device, dtype=pv0.dtype)
            else:
                delta = torch.zeros((1, T, 3, H, W), device=self.device, dtype=pv0.dtype)
            delta.uniform_(-1.0, 1.0)
            delta = torch.max(torch.min(delta, self.norm_eps), -self.norm_eps)
            delta.requires_grad_(True)
            params["delta_u"] = delta

        elif mode == "patch_delta":
            ph, pw = self.args.patch_h, self.args.patch_w
            pdelta = torch.zeros((1, 1, 3, ph, pw), device=self.device, dtype=pv0.dtype)
            pdelta.uniform_(-1.0, 1.0)
            pdelta = torch.max(torch.min(pdelta, self.norm_eps[..., :ph, :pw]), -self.norm_eps[..., :ph, :pw])
            pdelta.requires_grad_(True)
            params["patch_delta"] = pdelta

        elif mode == "patch_replace":
            ph, pw = self.args.patch_h, self.args.patch_w
            patch = torch.zeros((1, 1, 3, ph, pw), device=self.device, dtype=pv0.dtype)
            if self.args.patch_init == "random":
                patch.uniform_(-1.0, 1.0)
            else:
                patch.zero_()
            patch = torch.max(torch.min(patch, self.norm_max[..., :ph, :pw]), self.norm_min[..., :ph, :pw])
            patch.requires_grad_(True)
            params["patch"] = patch

        return params

    def _make_patch_coords(self, H: int, W: int) -> Tuple[int, int]:
        ph, pw = self.args.patch_h, self.args.patch_w
        if self.args.patch_random_loc:
            top = random.randint(0, max(0, H - ph))
            left = random.randint(0, max(0, W - pw))
            return top, left

        loc = self.args.patch_loc
        if loc == "top_left":
            return 0, 0
        if loc == "top_right":
            return 0, max(0, W - pw)
        if loc == "bottom_left":
            return max(0, H - ph), 0
        return max(0, H - ph), max(0, W - pw)

    def _apply_attack(self, pixel_clean: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        mode = self.args.attack_mode
        
        # Expand back if 4D
        is_4d = False
        if pixel_clean.dim() == 4:
            pixel_clean = pixel_clean.unsqueeze(0)
            is_4d = True
            
        _, T, _, H, W = pixel_clean.shape

        if mode == "uap_delta":
            delta_u = params["delta_u"]
            if delta_u.shape[1] == 1:
                delta = delta_u.repeat(1, T, 1, 1, 1)
            else:
                delta = delta_u
            adv = pixel_clean + delta
            return adv.squeeze(0) if is_4d else adv

        ph, pw = self.args.patch_h, self.args.patch_w
        ph_eff, pw_eff = min(int(ph), int(H)), min(int(pw), int(W))
        top, left = self._make_patch_coords(H, W) if (ph_eff == ph and pw_eff == pw) else (0, 0)

        if mode == "patch_delta":
            pdelta = params["patch_delta"]
            pdelta = pdelta[..., :ph_eff, :pw_eff]
            pdeltaT = pdelta.repeat(1, T, 1, 1, 1)
            adv = pixel_clean.clone()
            adv[:, :, :, top:top+ph_eff, left:left+pw_eff] = adv[:, :, :, top:top+ph_eff, left:left+pw_eff] + pdeltaT
            return adv.squeeze(0) if is_4d else adv

        if mode == "patch_replace":
            patch = params["patch"]
            patch = patch[..., :ph_eff, :pw_eff]
            patchT = patch.repeat(1, T, 1, 1, 1)
            adv = pixel_clean.clone()
            adv[:, :, :, top:top+ph_eff, left:left+pw_eff] = patchT
            return adv.squeeze(0) if is_4d else adv

    def _project_params(self, params: Dict[str, torch.Tensor]):
        mode = self.args.attack_mode
        if mode == "uap_delta":
            delta = params["delta_u"]
            delta.data = torch.max(torch.min(delta.data, self.norm_eps), -self.norm_eps)
            return
        if mode == "patch_delta":
            pdelta = params["patch_delta"]
            ph, pw = pdelta.shape[-2], pdelta.shape[-1]
            eps = self.norm_eps[..., :ph, :pw]
            pdelta.data = torch.max(torch.min(pdelta.data, eps), -eps)
            return
        if mode == "patch_replace":
            patch = params["patch"]
            ph, pw = patch.shape[-2], patch.shape[-1]
            pmin = self.norm_min[..., :ph, :pw]
            pmax = self.norm_max[..., :ph, :pw]
            patch.data = torch.max(torch.min(patch.data, pmax), pmin)
            return

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, prompt_len: int) -> torch.Tensor:
        logits_shift = logits[:, :-1, :].contiguous()
        labels_shift = labels[:, 1:].contiguous()
        V = logits_shift.size(-1)

        per_tok = F.cross_entropy(
            logits_shift.view(-1, V),
            labels_shift.view(-1),
            reduction="none"
        ).view_as(labels_shift)

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
            banned_mass = 0.0
            for bid in self.banned_ids:
                banned_mass = banned_mass + p0[int(bid)]
            loss_ban = banned_mass

        return loss_ce + float(self.args.eos_lambda) * loss_eos + float(self.args.ban_first_token_lambda) * loss_ban

    def train(self, train_items: List[Dict[str, Any]], dataset_dir: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        first_frames = None
        dummy_prompt_user = None
        for item in train_items:
            vp = os.path.join(dataset_dir, item["video"])
            fr = load_video(vp, num_frames=self.args.num_frames, random_jitter=False)
            if fr is not None:
                first_frames = fr
                _, dummy_prompt_user = self._build_prompts_for_item(item)
                break
        if first_frames is None:
            raise RuntimeError("No valid videos to initialize (all decode failed).")

        params = self._init_params(first_frames, dummy_prompt_user)

        # ==================== Resume Logic ====================
        if self.args.resume_from and os.path.exists(self.args.resume_from):
            print(f"[*] Resuming from checkpoint: {self.args.resume_from}")
            ckpt = torch.load(self.args.resume_from, map_location="cpu")
            for k in params.keys():
                if k in ckpt:
                    params[k].data = ckpt[k].to(self.device).data
        # ======================================================

        meta = {
            "attack_mode": self.args.attack_mode,
            "delta_mode": self.args.delta_mode,
            "patch_h": self.args.patch_h,
            "patch_w": self.args.patch_w,
            "patch_loc": self.args.patch_loc,
            "eps": self.args.eps,
            "alpha": self.args.alpha,
            "uap_epochs": self.args.uap_epochs,
            "sponge_multiplier": self.args.sponge_multiplier,
        }

        self.model.train()
        t0 = now()
        rng = random.Random(self.args.seed + 999)

        for ep in range(self.args.uap_epochs):
            items_copy = list(train_items)
            rng.shuffle(items_copy)
            pbar = tqdm(items_copy, desc=f"Train {self.args.attack_mode} epoch {ep+1}/{self.args.uap_epochs}")

            for item in pbar:
                vp = os.path.join(dataset_dir, item["video"])
                frames = load_video(
                    vp,
                    num_frames=self.args.num_frames,
                    random_jitter=self.args.train_random_jitter,
                    jitter=self.args.jitter
                )
                if frames is None:
                    continue

                prompt_full, prompt_user = self._build_prompts_for_item(item)
                prompt_len = self._get_prompt_len(prompt_user, frames)

                batch = self.processor(text=prompt_full, videos=[list(frames)], return_tensors="pt").to(self.device)
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                image_sizes = batch.get("image_sizes")

                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                pv_key = get_pixel_values_key(batch)
                pixel_clean = batch[pv_key].to(self.model.dtype)

                for _ in range(self.args.uap_iters_per_video):
                    for k in params:
                        if params[k].grad is not None:
                            params[k].grad.zero_()

                    pixel_adv = self._apply_attack(pixel_clean, params)

                    try:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values_videos=pixel_adv,
                            image_sizes=image_sizes,
                            use_cache=False,
                        )
                    except TypeError:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_adv,
                            image_sizes=image_sizes,
                            use_cache=False,
                        )
                    logits = outputs.logits.float()

                    loss = self._compute_loss(logits, labels, prompt_len)
                    loss.backward()

                    with torch.no_grad():
                        if self.args.attack_mode == "uap_delta":
                            params["delta_u"].data = params["delta_u"].data - self.norm_alpha * params["delta_u"].grad.sign()
                        elif self.args.attack_mode == "patch_delta":
                            ph, pw = params["patch_delta"].shape[-2], params["patch_delta"].shape[-1]
                            alpha = self.norm_alpha[..., :ph, :pw]
                            params["patch_delta"].data = params["patch_delta"].data - alpha * params["patch_delta"].grad.sign()
                        elif self.args.attack_mode == "patch_replace":
                            ph, pw = params["patch"].shape[-2], params["patch"].shape[-1]
                            alpha = self.norm_alpha[..., :ph, :pw]
                            params["patch"].data = params["patch"].data - alpha * params["patch"].grad.sign()

                        self._project_params(params)

                    pbar.set_postfix({"loss": float(loss.detach().cpu().item())})
            
            # ==================== Real-time Checkpointing ====================
            ckpt_path = os.path.join(self.args.output_dir, "universal_params_latest.pt")
            with torch.no_grad():
                save_obj = {
                    "attack_mode": self.args.attack_mode,
                    "delta_mode": self.args.delta_mode,
                    "epoch": ep + 1,
                    "args": vars(self.args),
                    "mean": self.image_mean,
                    "std": self.image_std,
                }
                for k, v in params.items():
                    save_obj[k] = v.detach().cpu()
            torch.save(save_obj, ckpt_path)
            # =================================================================

        cuda_sync()
        t1 = now()
        meta["train_time_s"] = float(t1 - t0)
        self.model.eval()
        return params, meta

    @torch.no_grad()
    def generate_with_new_tokens(self, item: Dict[str, Any], frames_uint8: np.ndarray, pixel_override: Optional[torch.Tensor] = None) -> Tuple[str, int]:
        _, prompt_user = self._build_prompts_for_item(item)
        inputs = self.processor(text=prompt_user, videos=[list(frames_uint8)], return_tensors="pt").to(self.device)
        
        if pixel_override is not None:
            pv_key = get_pixel_values_key(inputs)
            inputs[pv_key] = pixel_override
            if pv_key != "pixel_values_videos" and "pixel_values_videos" in inputs:
                del inputs["pixel_values_videos"]
            if pv_key != "pixel_values" and "pixel_values" in inputs:
                del inputs["pixel_values"]
                
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
        
        # Clamp bounds
        if pixel_adv.dim() == 4:
            pixel_adv = pixel_adv.unsqueeze(0)
            pixel_adv = torch.max(torch.min(pixel_adv, self.norm_max), self.norm_min)
            return pixel_adv.squeeze(0)
        
        pixel_adv = torch.max(torch.min(pixel_adv, self.norm_max), self.norm_min)
        return pixel_adv


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage", type=str, default="train_eval", choices=["train_eval", "eval_only"])
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the dataset containing QA.json and videos folder")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--load-params", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None, help="Path to universal_params_latest.pt to resume training")

    parser.add_argument("--model-family", type=str, default="llava_next_video", 
                        choices=["llava_next_video", "video_llava", "qwen_vl", "intern_vl"])
    parser.add_argument("--model-path", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # CHANGED DEFAULTS TO 20 AS REQUESTED
    parser.add_argument("--num-train-videos", type=int, default=20)
    parser.add_argument("--num-eval-videos", type=int, default=20)
    parser.add_argument("--eval-from-train", action="store_true")
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
    parser.add_argument("--delta-mode", type=str, default="full_time", choices=["shared_time", "full_time"])
    parser.add_argument("--patch-h", type=int, default=96)
    parser.add_argument("--patch-w", type=int, default=96)
    parser.add_argument("--patch-loc", type=str, default="bottom_right", choices=["top_left", "top_right", "bottom_left", "bottom_right"])
    parser.add_argument("--patch-random-loc", action="store_true")
    parser.add_argument("--patch-init", type=str, default="random", choices=["random", "zero"])

    parser.add_argument("--train-random-jitter", action="store_true")
    parser.add_argument("--jitter", type=int, default=3)
    parser.add_argument("--save-adv-videos", action="store_true")
    parser.add_argument("--fps", type=int, default=10)

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)

    args = parser.parse_args()
    seed_all(args.seed)

    dataset_dir = expanduser(args.dataset_dir)
    output_dir = expanduser(args.output_dir)
    safe_makedirs(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[Init] device={device}, dtype={args.dtype}")
    print(f"[Init] Loading backend={args.model_family} from: {args.model_path}")

    model, processor = load_video_llm_backend(args.model_path, args.model_family, torch_dtype=torch_dtype)
    if args.adapter_path:
        from peft import PeftModel
        print(f"[Init] Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.eval()

    # Load QA Dataset JSON
    all_items = load_qa_dataset(dataset_dir)
    print(f"[Dataset] Loaded {len(all_items)} items from {os.path.join(dataset_dir, 'QA.json')}")

    params: Dict[str, torch.Tensor]
    train_list: List[Dict[str, Any]]
    eval_list: List[Dict[str, Any]]
    train_meta: Dict[str, Any] = {}

    if args.stage == "eval_only":
        if not args.load_params:
            raise ValueError("--load-params is required when --stage=eval_only")

        ckpt_path = expanduser(args.load_params)
        print(f"[EvalOnly] Loading universal params from: {ckpt_path}")
        save_obj = torch.load(ckpt_path, map_location="cpu")

        if "attack_mode" in save_obj:
            args.attack_mode = save_obj["attack_mode"]
        if "delta_mode" in save_obj:
            args.delta_mode = save_obj["delta_mode"]

        params = {k: save_obj[k] for k in ("delta_u", "patch_delta", "patch") if k in save_obj}
        if not params:
            raise ValueError(f"No params found in checkpoint: {ckpt_path}")

        _, eval_list = split_train_eval_items(all_items, n_train=0, n_eval=args.num_eval_videos, seed=args.seed + 1337, eval_from_train=False)
        train_list = []

    else:
        train_list, eval_list = split_train_eval_items(
            all_items,
            n_train=args.num_train_videos,
            n_eval=args.num_eval_videos,
            seed=args.seed,
            eval_from_train=args.eval_from_train
        )

    # Save split records
    with open(os.path.join(output_dir, "train_items.json"), "w", encoding="utf-8") as f:
        json.dump(train_list, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "eval_items.json"), "w", encoding="utf-8") as f:
        json.dump(eval_list, f, indent=2, ensure_ascii=False)

    trainer = UniversalSpongeTrainer(model, processor, device, args)

    params_path: Optional[str] = None
    if args.stage != "eval_only":
        print(f"[Train] mode={args.attack_mode}, train_videos={len(train_list)}")
        cuda_sync()
        params, train_meta = trainer.train(train_list, dataset_dir)

        save_obj = {
            "attack_mode": args.attack_mode,
            "delta_mode": args.delta_mode,
            "args": vars(args),
            "train_meta": train_meta,
            "mean": trainer.image_mean,
            "std": trainer.image_std,
        }
        for k, v in params.items():
            save_obj[k] = v.detach().cpu()

        params_path = os.path.join(output_dir, "universal_params.pt")
        torch.save(save_obj, params_path)
        print(f"[Train] Final params saved to: {params_path}")
    else:
        params_path = expanduser(args.load_params) if args.load_params else None

    # Eval
    mean_t = torch.tensor(trainer.image_mean, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(trainer.image_std,  device=device).view(1, 3, 1, 1)

    summary: Dict[str, Any] = {}
    corrupt: List[str] = []
    timings: List[Timing] = []
    success_sponge = 0
    success_longer = 0
    before_new_tokens_list: List[int] = []
    after_new_tokens_list: List[int] = []
    token_ratio_list: List[float] = []

    pbar = tqdm(eval_list, desc="Eval (before/after)")
    for item in pbar:
        vid = os.path.splitext(os.path.basename(item["video"]))[0]
        vdir = os.path.join(output_dir, "eval", vid)
        safe_makedirs(vdir)
        log_path = os.path.join(vdir, "log.json")
        timing_path = os.path.join(vdir, "timing.json")

        if args.skip_existing and os.path.isfile(log_path) and os.path.isfile(timing_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    summary[vid] = json.load(f)
                continue
            except Exception:
                pass

        vpath = os.path.join(dataset_dir, item["video"])

        t_pre0 = now()
        frames = load_video(vpath, num_frames=args.num_frames, random_jitter=False)
        cuda_sync()
        t_pre1 = now()
        if frames is None:
            corrupt.append(item["video"])
            continue

        tinfo = Timing(preprocess_s=float(t_pre1 - t_pre0))

        cuda_sync(); t0 = now()
        raw_before, before_new_tokens = trainer.generate_with_new_tokens(item, frames, pixel_override=None)
        cuda_sync(); t1 = now()
        tinfo.gen_before_s = float(t1 - t0)
        tinfo.gen_before_new_tokens = int(before_new_tokens)

        cuda_sync(); t2 = now()
        pixel_adv = trainer.build_adv_pixels_for_eval(item, frames, {k: v.to(device) for k, v in params.items()})
        cuda_sync(); t3 = now()
        tinfo.apply_attack_s = float(t3 - t2)

        if args.save_adv_videos:
            adv_uint8 = denorm_video_to_uint8(pixel_adv, mean=mean_t, std=std_t)
            save_video(adv_uint8, os.path.join(vdir, f"adv_{vid}.mp4"), fps=args.fps)

        cuda_sync(); t4 = now()
        raw_after, after_new_tokens = trainer.generate_with_new_tokens(item, frames, pixel_override=pixel_adv)
        cuda_sync(); t5 = now()
        tinfo.gen_after_s = float(t5 - t4)
        tinfo.gen_after_new_tokens = int(after_new_tokens)

        before_ans = extract_assistant(raw_before)
        after_ans = extract_assistant(raw_after)

        base_sponge_check = item.get("SPONGE_TARGET", "However")[:20]
        if base_sponge_check in after_ans:
            success_sponge += 1
        if len(after_ans) > len(before_ans) + 30:
            success_longer += 1

        token_ratio = safe_div(after_new_tokens, max(1, before_new_tokens), default=0.0)

        rec = {
            "video_path": item["video"],
            "question": item["question"],
            "before_len": len(before_ans),
            "after_len": len(after_ans),
            "ratio": round(len(after_ans) / max(1, len(before_ans)), 3),
            "before_new_tokens": int(before_new_tokens),
            "after_new_tokens": int(after_new_tokens),
            "token_ratio": round(token_ratio, 6),
            "gen_before_s": float(tinfo.gen_before_s),
            "gen_after_s": float(tinfo.gen_after_s),
            "response_before": before_ans,
            "response_after": after_ans,
        }
        summary[vid] = rec
        timings.append(tinfo)
        before_new_tokens_list.append(int(before_new_tokens))
        after_new_tokens_list.append(int(after_new_tokens))
        token_ratio_list.append(float(token_ratio))

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

        with open(timing_path, "w", encoding="utf-8") as f:
            d = asdict(tinfo)
            d.update({
                "total_before_s": tinfo.total_before_s,
                "total_after_s": tinfo.total_after_s,
                "overhead_s": tinfo.overhead_s,
            })
            json.dump(d, f, indent=2, ensure_ascii=False)

    eval_list_out = [summary[k] for k in sorted(summary.keys())]
    with open(os.path.join(output_dir, "final_summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_list_out, f, indent=2, ensure_ascii=False)

    timing_summary = {
        "n_eval_requested": len(eval_list),
        "n_eval_processed": len(timings),
        "n_eval_corrupt_or_failed": len(corrupt),
        "success_sponge_triggered": success_sponge,
        "success_longer_than_before_plus_30chars": success_longer,
        "stats_before_new_tokens": stats([float(x) for x in before_new_tokens_list]),
        "stats_after_new_tokens": stats([float(x) for x in after_new_tokens_list]),
        "stats_token_ratio": stats([float(x) for x in token_ratio_list]),
        "stats_total_before_s": stats([t.total_before_s for t in timings]),
        "stats_total_after_s": stats([t.total_after_s for t in timings]),
        "stats_overhead_s": stats([t.overhead_s for t in timings]),
    }
    with open(os.path.join(output_dir, "timing_summary_eval.json"), "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("Done!")
    print(f"Dataset Dir: {dataset_dir}")
    print(f"Mode: {args.attack_mode}")
    print(f"Train videos: {len(train_list)}")
    print(f"Eval videos:  {len(eval_list)} (evaluated {len(timings)}, corrupt {len(corrupt)})")
    print(f"Saved universal params: {params_path}")
    print("==============================\n")

if __name__ == "__main__":
    main()