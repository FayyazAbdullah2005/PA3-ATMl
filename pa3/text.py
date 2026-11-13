"""Chat boundaries, pinned model loading, and resumable individual generations."""
from __future__ import annotations
from pathlib import Path
import time
import torch
from .config import QWEN, SIGLIP, LIVE_MODEL, LIVE_REVISION
from .io import Store, digest, read_json, write_json, environment


def ensure_pins(root: Path, commit=lambda: None):
    from huggingface_hub import HfApi
    root = Path(root)
    target = root / "data" / "pins.json"
    if target.exists():
        return read_json(target)
    api = HfApi()
    pins = {"models": {name: api.model_info(name).sha for name in (QWEN, SIGLIP)},
            "datasets": {name: api.dataset_info(name).sha for name in
                         ("meta-math/MetaMathQA", "openai/gsm8k", "Rowan/hellaswag","aps/super_glue","allenai/ai2_arc")}}
    pins["models"][LIVE_MODEL] = LIVE_REVISION
    write_json(target, pins)
    commit()
    return pins


def cache_model(root: Path, name: str, commit=lambda: None):
    from huggingface_hub import snapshot_download
    pins = ensure_pins(root, commit)
    return snapshot_download(name, revision=pins["models"][name],
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])


def tokenizer_for(root: Path, name=QWEN):
    from transformers import AutoTokenizer
    pins = read_json(Path(root) / "data" / "pins.json")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=pins["models"][name])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_qwen(root: Path, full=False, name=QWEN, training=False, device="cuda"):
    from transformers import AutoModelForCausalLM
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for assignment GPU jobs. CPU tests use tiny synthetic models.")
    pins = read_json(Path(root) / "data" / "pins.json")
    model = AutoModelForCausalLM.from_pretrained(name, revision=pins["models"][name],
        torch_dtype=torch.float32 if full else torch.float16, attn_implementation="sdpa",
        low_cpu_mem_usage=True).to(device)
    model.config.use_cache = not training
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def chat_ids(tokenizer, messages, generation=False, thinking=False):
    kwargs = {"tokenize": True, "add_generation_prompt": generation}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def supervised_example(tokenizer, question, answer, thinking=False):
    prefix = chat_ids(tokenizer, [{"role": "user", "content": question}], True, thinking)
    complete = chat_ids(tokenizer, [{"role": "user", "content": question},
                                   {"role": "assistant", "content": answer}], False, thinking)
    # Some Qwen templates put the non-thinking empty <think> block only in the
    # generation prefix. Render the supplied assistant content with that block
    # when needed so training and evaluation share exactly the same prefix.
    if complete[:len(prefix)] != prefix:
        empty = "<think>\n\n</think>\n\n"
        complete = chat_ids(tokenizer, [{"role": "user", "content": question},
            {"role": "assistant", "content": empty+answer}], False, thinking)
    if complete[:len(prefix)] != prefix:
        raise ValueError("Chat template prefix does not match the supplied assistant sequence; refusing incorrect loss masking")
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Templates can add a trailing newline after end-of-turn; do not supervise it.
    endings = [i for i in range(len(prefix), len(complete)) if complete[i] == eot]
    if not endings:
        raise ValueError("Assistant end-of-turn token missing")
    complete = complete[:endings[-1]+1]
    return {"input_ids": complete, "labels": [-100]*len(prefix)+complete[len(prefix):],
            "prefix_length": len(prefix)}


@torch.inference_mode()
def generate_text(model, tokenizer, messages, max_new_tokens, *, thinking=False,
                  seed=None, sampling=False):
    from .train import seed_all
    if seed is not None:
        seed_all(seed)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    ids = chat_ids(tokenizer, messages, True, thinking)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": sampling,
              "use_cache": True, "pad_token_id": tokenizer.pad_token_id,
              "eos_token_id": tokenizer.eos_token_id}
    if sampling:
        kwargs.update(temperature=.7, top_p=.8, top_k=20, repetition_penalty=1.0)
    output = model.generate(input_ids=x, attention_mask=torch.ones_like(x), **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    generated = output[0, len(ids):].tolist()
    decoded=tokenizer.decode(generated,skip_special_tokens=True)
    seconds=time.perf_counter()-begin
    eos = model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    stopped = bool(generated and generated[-1] in stops)
    return {"messages": messages, "input_ids": ids, "generated_token_ids": generated,
            "generated_text":decoded,
            "input_tokens": len(ids), "output_tokens": len(generated),
            "hit_max_new_tokens": len(generated) == max_new_tokens and not stopped,
            "seconds": seconds, "seed": seed,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}

