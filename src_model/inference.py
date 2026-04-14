import hydra
import tiktoken
import torch
from omegaconf import DictConfig, OmegaConf

from src_model.model.gpt import GPT

EOT = 50256  # <|endoftext|> — only special token in GPT-2 vocab; used as BOS and EOS

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
    """
    Build the token sequence for generation, matching the training format exactly:

      <EOT> System: ... \n User: turn1 \n Assistant: reply1 <EOT> User: turn2 \n Assistant:

    The system prompt appears once at the very start. Subsequent turns use
    EOT as a separator with no repeated system prefix.
    """
    tokens: list[int] = []
    if turns:
        # First historical turn carries the system prompt
        first_user, first_reply = turns[0]
        tokens += [EOT] + enc.encode_ordinary(
            f"System: {SYSTEM_PROMPT}\nUser: {first_user}\nAssistant: {first_reply}"
        )
        for user_msg, reply in turns[1:]:
            tokens += [EOT] + enc.encode_ordinary(f"User: {user_msg}\nAssistant: {reply}")
        tokens += [EOT] + enc.encode_ordinary(f"User: {current_user_input}\nAssistant: ")
    else:
        # First turn of a fresh conversation
        tokens += [EOT] + enc.encode_ordinary(
            f"System: {SYSTEM_PROMPT}\nUser: {current_user_input}\nAssistant: "
        )
    return tokens


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, topk: int, temperature: float, repetition_penalty: float) -> torch.Tensor:
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
    """
    Strip trailing whitespace and cut off anything the model wrote after its turn.
    The model was trained on single-turn data, so it sometimes writes a fake
    'User:' continuation before generating EOT.
    """
    # Cut at the first occurrence of a new user turn
    for stop in ("\nUser:", "\nAssistant:"):
        if stop in raw:
            raw = raw[: raw.index(stop)]
    return raw.strip()


@hydra.main(version_base=None, config_name="config_inference", config_path="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.inference.device if torch.cuda.is_available() else "cpu")

    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    enc = tiktoken.get_encoding("gpt2")
    model = build_model(cfg, device)
    load_checkpoint(model, cfg.inference.checkpoint, device)

    print("\nType your message and press Enter. Type 'exit' or Ctrl-C to quit, 'reset' to clear history.\n")

    # List of (user_msg, assistant_reply) prior turns
    turns: list[tuple[str, str]] = []

    while True:
        try:
            print("> User: ", end="", flush=True)
            user_input = input()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        cmd = user_input.strip().lower()
        if cmd in ("exit", "quit"):
            break
        if cmd == "reset":
            turns.clear()
            print("(conversation reset)\n")
            continue
        if not user_input.strip():
            continue

        tokens = build_context_tokens(turns, user_input, enc)
        idx = torch.tensor(tokens, dtype=torch.long, device=device)

        print("> Assistant: ", end="", flush=True)
        with torch.inference_mode():
            out = generate(model, idx, cfg.inference.max_new_tokens, cfg.inference.topk, cfg.inference.temperature, cfg.inference.repetition_penalty)

        input_len = idx.size(0)
        gen = out[0, input_len:].tolist()
        # Model vocab padded to 50304; tokenizer only knows 50257 tokens.
        gen = [t if t < 50257 else EOT for t in gen]

        raw_reply = enc.decode(gen)
        reply = clean_reply(raw_reply)

        print(f"{reply}\n")
        turns.append((user_input, reply))


if __name__ == "__main__":
    main()
