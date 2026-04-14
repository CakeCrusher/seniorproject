import tiktoken
import torch
from omegaconf import DictConfig

from model.gpt import GPT

EOT = 50256

SYSTEM_PROMPT = "You are GatorLM, a helpful AI assistant developed at the University of Florida."


def build_model(cfg: DictConfig, device: torch.device):
    model = GPT(
        n_embd=cfg.model.n_embd,
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_heads=cfg.model.n_heads,
        head_size=cfg.model.head_size,
        rope_head_size=cfg.model.rope_head_size,
        n_layers=cfg.model.n_layers,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded: {checkpoint_path}")
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")


def build_context_tokens(turns: list[tuple[str, str]], current_user_input: str, enc) -> list[int]:
    tokens: list[int] = []
    if turns:
        first_user, first_reply = turns[0]
        tokens += [EOT] + enc.encode_ordinary(
            f"System: {SYSTEM_PROMPT}\nUser: {first_user}\nAssistant: {first_reply}"
        )
        for user_msg, reply in turns[1:]:
            tokens += [EOT] + enc.encode_ordinary(f"User: {user_msg}\nAssistant: {reply}")
        tokens += [EOT] + enc.encode_ordinary(f"User: {current_user_input}\nAssistant: ")
    else:
        tokens += [EOT] + enc.encode_ordinary(
            f"System: {SYSTEM_PROMPT}\nUser: {current_user_input}\nAssistant: "
        )
    return tokens


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, topk: int, temperature: float,
             repetition_penalty: float) -> torch.Tensor:
    return model.generate(
        idx,
        num_sequences=1,
        max_tokens=max_new_tokens,
        topk=topk,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        chat_mode=True,
        eos_token=EOT,
    )


def clean_reply(raw: str) -> str:
    for stop in ("\nUser:", "\nAssistant:"):
        if stop in raw:
            raw = raw[: raw.index(stop)]
    return raw.strip()
