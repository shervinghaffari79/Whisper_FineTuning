#!/usr/bin/env python3
"""
Local chat / transcript-analysis LLM for the AI Analysis panel.

Backends:
  - "mlx"          Qwen3-4B-Instruct (MLX 4-bit) on the Apple-Silicon GPU.
                   Only importable on macOS + Apple Silicon.
  - "transformers" The same Qwen3-4B-Instruct model via HF `transformers`,
                   run on an NVIDIA GPU (float16) if available, else CPU.
                   Cross-platform: this is the path used on Windows/Linux.

Selection is automatic ("auto"): tries mlx first, falls back to transformers.
Force one explicitly with the CHAT_BACKEND env var ("mlx" | "transformers")
if the auto-detection ever guesses wrong for your machine.

Both backends expose the same public functions used by server.py:
stream_chat(messages, transcript) and make_title(transcript).
"""
import glob
import os
import threading

MLX_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
HF_MODEL = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen3.5-4B")
_SNAP_GLOB = "models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/chat_template.jinja"

_active = None  # "mlx" | "transformers"
_MODEL = None
_TOK = None
_TMPL = None


def _try_mlx() -> bool:
    try:
        import mlx_lm  # noqa: F401 -- availability probe only
        return True
    except Exception:
        return False


def _select():
    global _active
    if _active is not None:
        return
    requested = os.environ.get("CHAT_BACKEND", "auto").lower()
    if requested in ("mlx", "auto") and _try_mlx():
        _active = "mlx"
    else:
        _active = "transformers"


def active_model_name() -> str:
    _select()
    return MLX_MODEL if _active == "mlx" else HF_MODEL


def _ensure():
    """Lazily load the active backend's model/tokenizer (and MLX chat
    template, if applicable)."""
    global _MODEL, _TOK, _TMPL
    _select()
    if _MODEL is not None:
        return _MODEL, _TOK, _TMPL
    if _active == "mlx":
        from mlx_lm import load
        _MODEL, _TOK = load(MLX_MODEL)
        hits = glob.glob(os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), _SNAP_GLOB))
        _TMPL = open(hits[0]).read() if hits else None
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _TOK = AutoTokenizer.from_pretrained(HF_MODEL)
        _MODEL = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=dtype).to(device)
        _MODEL.eval()
    return _MODEL, _TOK, _TMPL


def _system_prompt(transcript: str) -> str:
    if transcript:
        return (
            "You are a speech analytics AI assistant. You have access to the following "
            "transcript from a recorded conversation.\n\nTRANSCRIPT:\n" + transcript +
            "\n\nGuidelines:\n"
            "- Always respond in Persian (Farsi) regardless of the question's language\n"
            "- Keep answers brief and to the point\n"
            "- Reference specific speakers (S1, S2, S3, …) when relevant\n"
            "- Include timestamps only when directly useful"
        )
    return ("You are a helpful AI assistant specialized in speech transcription and audio "
            "analysis. Always respond in Persian (Farsi) briefly and to the point.")


def _build_messages(messages, transcript):
    msgs = [{"role": "system", "content": _system_prompt(transcript)}]
    for m in messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def stream_chat(messages, transcript="", max_tokens=1024, temperature=0.7):
    """Yield generated Persian text token-by-token, on whichever backend is active."""
    model, tok, tmpl = _ensure()
    msgs = _build_messages(messages, transcript)

    if _active == "mlx":
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, chat_template=tmpl)
        sampler = make_sampler(temp=temperature)
        for resp in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
            if resp.text:
                yield resp.text
        return

    # transformers backend: generate in a background thread, stream via TextIteratorStreamer
    import torch
    from transformers import TextIteratorStreamer
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer,
                      do_sample=temperature > 0, temperature=max(temperature, 0.01))
    thread = threading.Thread(target=lambda: model.generate(**gen_kwargs), daemon=True)
    thread.start()
    for text in streamer:
        if text:
            yield text
    thread.join()


def make_title(transcript: str) -> str:
    model, tok, tmpl = _ensure()
    msgs = [{"role": "user", "content":
             "Generate a concise 4-6 word Persian title for this transcript "
             "(no quotes, no trailing punctuation):\n\n" + transcript[:500]}]

    if _active == "mlx":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, chat_template=tmpl)
        out = generate(model, tok, prompt=prompt, max_tokens=24,
                       sampler=make_sampler(temp=0.5), verbose=False).strip()
    else:
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
        out_ids = model.generate(**inputs, max_new_tokens=24, do_sample=True, temperature=0.5)
        out = tok.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    return out.splitlines()[0].strip('"“”') if out else "تحلیل جدید"
