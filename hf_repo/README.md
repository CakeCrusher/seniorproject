---
language:
- en
pipeline_tag: text-generation
tags:
- gpt
- custom
- university-of-florida
- gatorlm
inference: true
widget:
- messages:
  - role: user
    content: "Who are you?"
- messages:
  - role: user
    content: "What can you help me with?"
- messages:
  - role: user
    content: "Tell me about the University of Florida."
---

# GatorLM

GatorLM is a custom GPT-style language model developed at the University of Florida.

## Model Details

- **Architecture:** Custom GPT with RoPE, GQA, RMSNorm, and SwiGLU MLP
- **Parameters:** ~2B
- **Context length:** 2048 tokens
- **Tokenizer:** GPT-2 (tiktoken)
- **Dtype:** bfloat16

## Usage

This model uses a custom inference handler. Send requests in the following format:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="Krill11/GatorLM1")
response = client.post(json={"turns": [], "message": "Hello, who are you?"})
```

### Input format

```json
{
  "turns": [["previous user message", "previous assistant reply"]],
  "message": "current user message"
}
```

- `turns`: list of completed `[user, assistant]` exchange pairs (empty list for a fresh conversation)
- `message`: the new user message

### Output format

```json
{
  "reply": "GatorLM's response"
}
```

## Training

Fine-tuned via supervised fine-tuning (SFT) on conversational data using the Muon optimizer.
