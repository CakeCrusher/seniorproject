import asyncio
import json
from contextlib import asynccontextmanager

import tiktoken
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from omegaconf import OmegaConf
from pydantic import BaseModel

from src_model.inference import (
    EOT,
    build_context_tokens,
    build_model,
    load_checkpoint,
)

# Load config manually (no Hydra) — CWD must be the repo root when uvicorn is launched.
_cfg = OmegaConf.load("src_model/config/config_inference.yaml")

# Global state populated at startup
_state: dict = {}

STOP_SEQUENCES = ("\nUser:", "\nAssistant:")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model(_cfg, device)
    load_checkpoint(model, _cfg.inference.checkpoint, device)

    _state.update({
        "model": model,
        "device": device,
        "enc": tiktoken.get_encoding("gpt2"),
        "cfg": _cfg,
    })
    print(f"GatorLM ready on {device}.")
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    turns: list[list[str]]  # [[user, assistant], ...] — may be empty
    message: str


def _stream_tokens(model, idx, max_new_tokens, topk, temperature, repetition_penalty, enc, queue):
    """
    Runs the generation loop token-by-token in a background thread.
    Puts decoded text pieces into the queue; puts None when done.
    Replicates GPT.generate() logic but yields each token immediately.
    """
    idx = idx.unsqueeze(0)  # add batch dim (num_sequences=1)
    accumulated = ""

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, _ = model.forward(idx)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for token_id in idx[0].tolist():
                    logits[0, token_id] /= repetition_penalty

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=topk)
            idx_next = torch.multinomial(topk_probs, num_samples=1)
            idx_next = torch.gather(topk_indices, -1, idx_next)

            token_id = idx_next[0, 0].item()

            # Stop at EOT
            if token_id == EOT:
                break

            idx = torch.cat([idx, idx_next], dim=-1)

            # Clamp tokens beyond tiktoken's range
            safe_id = token_id if token_id < 50257 else EOT
            piece = enc.decode([safe_id])
            accumulated += piece

            # Check for stop sequences — stop generating but send text up to the marker
            stop_found = False
            for stop in STOP_SEQUENCES:
                if stop in accumulated:
                    piece = accumulated[: accumulated.index(stop)]
                    # Send any remaining clean text before the stop marker
                    already_sent = accumulated[: accumulated.index(stop)]
                    unsent = already_sent[len(accumulated) - len(piece):]
                    if unsent:
                        queue.put_nowait(unsent)
                    stop_found = True
                    break

            if stop_found:
                break

            if piece:
                queue.put_nowait(piece)

    queue.put_nowait(None)  # signal completion


@app.post("/generate/stream")
async def generate_stream_endpoint(req: GenerateRequest):
    if not _state.get("model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    turns: list[tuple[str, str]] = [
        (pair[0], pair[1]) for pair in req.turns if len(pair) == 2
    ]

    enc = _state["enc"]
    device = _state["device"]
    cfg = _state["cfg"]
    model = _state["model"]

    tokens = build_context_tokens(turns, req.message, enc)
    idx = torch.tensor(tokens, dtype=torch.long, device=device)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Run blocking generation in a thread pool; results flow back via queue
    loop.run_in_executor(
        None,
        _stream_tokens,
        model, idx,
        cfg.inference.max_new_tokens,
        cfg.inference.topk,
        cfg.inference.temperature,
        cfg.inference.repetition_penalty,
        enc, queue,
    )

    async def event_stream():
        while True:
            piece = await queue.get()
            if piece is None:
                break
            yield f"data: {json.dumps({'content': piece})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
