import os
import torch
import tiktoken
from omegaconf import OmegaConf
from inference import build_model, load_checkpoint, build_context_tokens, generate, clean_reply, EOT


class EndpointHandler:
    def __init__(self, path=""):
        cfg_path = os.path.join(path, "config_inference.yaml")
        self.cfg = OmegaConf.load(cfg_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.cfg, self.device)

        ckpt_path = os.path.join(path, "best_val.pt")
        load_checkpoint(self.model, ckpt_path, self.device)

        self.enc = tiktoken.get_encoding("gpt2")
        print(f"GatorLM ready on {self.device}.")

    def __call__(self, data: dict) -> dict:
        # HF wraps the request body under an "inputs" key
        payload = data.get("inputs", data)
        turns = [(p[0], p[1]) for p in payload.get("turns", []) if len(p) == 2]
        message = payload.get("message", "")

        tokens = build_context_tokens(turns, message, self.enc)
        idx = torch.tensor(tokens, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            out = generate(
                self.model, idx,
                self.cfg.inference.max_new_tokens,
                self.cfg.inference.topk,
                self.cfg.inference.temperature,
                self.cfg.inference.repetition_penalty,
            )

        gen = [t if t < 50257 else EOT for t in out[0, idx.size(0):].tolist()]
        return {"reply": clean_reply(self.enc.decode(gen))}
