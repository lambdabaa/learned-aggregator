"""Minimal OpenAI-compatible chat completions server using transformers + MPS.

Replaces mlx_lm.server for trajectory generation.  mlx_lm compiles its
sampling function with a baked-in random state, making temperature sampling
completely deterministic across requests.  This server uses PyTorch MPS, whose
random state advances correctly between calls.

Usage:
    python scripts/lm_server.py --model Qwen/Qwen2.5-1.5B-Instruct --port 8100
"""

from __future__ import annotations

import argparse
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="lm_server", description="Minimal OAI-compatible server (transformers+MPS)")

_tokenizer = None
_model = None
_model_id = None


def _load_model(model_id: str, device: str) -> None:
    global _tokenizer, _model, _model_id
    print(f"Loading {model_id} on {device} …")
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device).eval()
    _model_id = model_id
    print("Model ready.")


class _Message(BaseModel):
    role: str
    content: str


class _ChatRequest(BaseModel):
    model: str
    messages: list[_Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    seed: int | None = None


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": _model_id, "object": "model", "created": int(time.time())}]}


@app.post("/v1/chat/completions")
def chat_completions(req: _ChatRequest):
    if req.seed is not None:
        torch.manual_seed(req.seed)

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    text = _tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature if req.temperature > 0 else 1.0,
            do_sample=req.temperature > 0,
            top_p=req.top_p,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][prompt_len:]
    content = _tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": len(new_ids), "total_tokens": prompt_len + len(new_ids)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    _load_model(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
