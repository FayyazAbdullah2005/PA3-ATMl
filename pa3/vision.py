"""Cached SigLIP features and a causal Qwen visual interface."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .config import *
from .io import *
from .text import *
from .models import install_lora, LoRALinear


def cache_features(root, commit=lambda: None):
    from PIL import Image
    from transformers import SiglipVisionModel, AutoImageProcessor
    root = Path(root)
    source = root/"data"/"task2"
    store = Store(root, commit)
    if not store.done(source):
        raise RuntimeError("Run task2-prepare first")
    index = read_json(source/"image_index.json")
    pins = read_json(root/"data"/"pins.json")
    signature = digest({"data": read_json(source/"DONE.json")["signature"],
        "siglip": pins["models"][SIGLIP], "encoder_contract":"siglip-last-hidden-state-fp16-196-v1"})
    d = source/"features"
    if store.done(d, signature):
        return
    ids = sorted(map(int, index))
    output_index = {}
    model, processor = None, None
    for offset in range(0, len(ids), 256):
        selected = ids[offset:offset+256]
        name = f"shard-{offset//256:04d}"
        unit = d/name
        usig = digest([signature, selected])
        if not store.done(unit, usig):
            if model is None:
                model = SiglipVisionModel.from_pretrained(SIGLIP, revision=pins["models"][SIGLIP],
                    torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
                processor = AutoImageProcessor.from_pretrained(SIGLIP, revision=pins["models"][SIGLIP])
            begin = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            outputs = []
            with torch.inference_mode():
                for start in range(0, len(selected), 32):
                    images = []
                    for image_id in selected[start:start+32]:
                        with Image.open(source/"images"/index[str(image_id)]["file"]) as image:
                            images.append(image.convert("RGB"))
                    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
                    features = model(pixel_values=pixels).last_hidden_state
                    # last_hidden_state already includes SigLIP's final layer norm.
                    if features.shape[1:] != (196,768) or not torch.isfinite(features).all():
                        raise ValueError(f"Invalid SigLIP features: {features.shape}")
                    outputs.append(features.cpu().numpy())
            data = np.concatenate(outputs).astype(np.float16)
            with atomic_path(unit/"features.npy") as tmp, tmp.open("wb") as f:
                np.save(f, data, allow_pickle=False)
            torch.cuda.synchronize()
            write_json(unit/"resources.json", {"images":len(selected),"seconds":time.perf_counter()-begin,
                "peak_gpu_bytes":torch.cuda.max_memory_allocated()})
            store.finish(unit, usig, [unit/"features.npy", unit/"resources.json"])
        for row, image_id in enumerate(selected):
            output_index[str(image_id)] = {"shard":name,"row":row}
    write_json(d/"index.json", output_index)
    write_json(d/"provenance.json", {"signature": signature, "shape":[196,768],
        "dtype":"float16","normalization":"SigLIP last_hidden_state; no second normalization"})
    store.finish(d, signature, [d/"index.json", d/"provenance.json"])


class Features:
    def __init__(self, directory, tokens=196):
        self.directory, self.tokens = Path(directory), tokens
        self.index = read_json(self.directory/"index.json")
        self._shards={}
        self._verified=set()
        if tokens not in (196,49):
            raise ValueError("Visual tokens must be 196 or 49")

    def shard(self, name):
        if name not in self._shards:
            if (self.directory/name/"DONE.json").exists() and name not in self._verified:
                Store(self.directory).done(self.directory/name)
                self._verified.add(name)
            if len(self._shards)>=64:
                oldest=next(iter(self._shards))
                self._shards.pop(oldest)._mmap.close()
            self._shards[name]=np.load(self.directory/name/"features.npy",mmap_mode="r",allow_pickle=False)
        return self._shards[name]

    def close(self):
        for value in self._shards.values():
            value._mmap.close()
        self._shards.clear()

    def __del__(self):
        if hasattr(self,"_shards"):
            self.close()

    def get(self, image_id):
        row = self.index[str(image_id)]
        x = np.array(self.shard(row["shard"])[row["row"]], copy=True)
        if self.tokens == 49:
            x = x.reshape(7,2,7,2,768).astype(np.float32).mean(axis=(1,3)).reshape(49,768).astype(np.float16)
        return torch.from_numpy(x)


def user_header_end(tokenizer, ids):
    header = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    for i in range(len(ids)-len(header)+1):
        if ids[i:i+len(header)] == header:
            return i+len(header)
    raise ValueError("Could not locate the Qwen user-message header")


def question_text(example):
    return example["question"] if example["type"] == "caption" else "Answer the question with a short answer.\n"+example["question"]


def vision_batch(examples, tokenizer, features, image_ids=None, generation=False, no_image=False):
    ids_list, labels_list, vmasks, tensors = [], [], [], []
    for i, example in enumerate(examples):
        text = question_text(example)
        if generation:
            ids = chat_ids(tokenizer, [{"role":"user","content":text}], True)
            labels = [-100]*len(ids)
        else:
            e = supervised_example(tokenizer, text, example["answer"])
            ids, labels = e["input_ids"], e["labels"]
        position = user_header_end(tokenizer, ids)
        count = 0 if no_image else features.tokens
        image = example["image_id"] if image_ids is None else image_ids[i]
        if count:
            tensors.append(features.get(image))
        ids_list.append(ids[:position]+[tokenizer.pad_token_id]*count+ids[position:])
        labels_list.append(labels[:position]+[-100]*count+labels[position:])
        vmasks.append([False]*position+[True]*count+[False]*(len(ids)-position))
    width = max(map(len,ids_list))
    ids = torch.full((len(examples),width),tokenizer.pad_token_id,dtype=torch.long)
    labels = torch.full_like(ids,-100)
    mask = torch.zeros_like(ids)
    visual = torch.zeros_like(ids,dtype=torch.bool)
    for i,(x,y,v) in enumerate(zip(ids_list,labels_list,vmasks)):
        start = width-len(x) if generation else 0
        ids[i,start:start+len(x)] = torch.tensor(x)
        labels[i,start:start+len(x)] = torch.tensor(y)
        mask[i,start:start+len(x)] = 1
        visual[i,start:start+len(x)] = torch.tensor(v)
    return {"input_ids":ids,"labels":labels,"attention_mask":mask,"visual_mask":visual,
            "features":torch.stack(tensors) if tensors else None}


class VLM(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.lm.requires_grad_(False)
        device = next(lm.parameters()).device
        self.connector = nn.Sequential(nn.Linear(768,1024),nn.GELU(),nn.Linear(1024,1024)).to(device,torch.float32)
        self.register_buffer("visual_scale",torch.tensor(1.0,device=device,dtype=torch.float32))
        install_lora(self.lm,VISION_TARGETS,16,32)
        self.set_stage(0)

    def set_stage(self, stage):
        for module in self.lm.modules():
            if isinstance(module,LoRALinear):
                module.enabled = stage>0
                module.A.requires_grad_(stage>0)
                module.B.requires_grad_(stage>0)

    def embedding_inputs(self,batch):
        device = next(self.parameters()).device
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        visual_mask = batch["visual_mask"].to(device)
        embeddings = self.lm.get_input_embeddings()(ids)
        if batch["features"] is not None:
            projected = self.connector(batch["features"].to(device,torch.float32))*self.visual_scale
            projected = projected.to(embeddings.dtype)
            if int(visual_mask.sum()) != projected.shape[0]*projected.shape[1]:
                raise ValueError("Visual mask/feature count mismatch")
            embeddings = embeddings.masked_scatter(visual_mask.unsqueeze(-1).expand_as(embeddings),projected.reshape(-1))
        positions = (mask.cumsum(dim=1)-1).clamp(min=0)
        return embeddings,mask,positions

    def forward(self,batch):
        embeddings,mask,positions = self.embedding_inputs(batch)
        return self.lm(inputs_embeds=embeddings,attention_mask=mask,position_ids=positions,use_cache=False)

    @torch.no_grad()
    def calibrate(self,examples,tokenizer,features):
        # A fixed training-only subset; no development/test features influence scale.
        batch = vision_batch(examples,tokenizer,features)
        device = next(self.parameters()).device
        values = batch["features"].to(device,torch.float32)
        projected = self.connector(values)
        valid = (batch["attention_mask"].bool() & ~batch["visual_mask"]).to(device)
        text = self.lm.get_input_embeddings()(batch["input_ids"].to(device))[valid].float()
        visual_rms = projected.square().mean().sqrt()
        text_rms = text.square().mean().sqrt()
        scale = text_rms/visual_rms.clamp(min=1e-8)
        if not torch.isfinite(scale) or scale<=0:
            raise ValueError("Invalid connector calibration")
        self.visual_scale.copy_(scale)
        return {"rule":"training-only text RMS / initial projected-visual RMS", "scale":float(scale),
                "text_rms":float(text_rms),"visual_rms":float(visual_rms),
                "image_ids":[e["image_id"] for e in examples]}

