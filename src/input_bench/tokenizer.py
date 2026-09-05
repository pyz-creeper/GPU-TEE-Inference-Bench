from __future__ import annotations

import hashlib
import random
from typing import Any, Protocol


class Tokenizer(Protocol):
    name_or_path: str
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...
    def apply_chat_template(self, messages: list[dict[str, Any]], *, tokenize: bool,
                            add_generation_prompt: bool) -> Any: ...


class WhitespaceTokenizer:
    """Small deterministic tokenizer intended for tests and offline smoke runs."""
    name_or_path = "whitespace"
    vocab_size = 65536
    chat_template = "<role> content"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) % self.vocab_size
                for x in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(f"tok{i}" for i in ids)

    def apply_chat_template(self, messages: list[dict[str, Any]], *, tokenize: bool,
                            add_generation_prompt: bool) -> Any:
        text = " ".join(f"<{m['role']}> {m.get('content', '')}" for m in messages)
        if add_generation_prompt:
            text += " <assistant>"
        return self.encode(text) if tokenize else text


def load_tokenizer(identifier: str, revision: str | None = None) -> Tokenizer:
    if identifier == "whitespace":
        return WhitespaceTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for a model tokenizer; install input-bench") from exc
    return AutoTokenizer.from_pretrained(identifier, revision=revision, trust_remote_code=False)


def count_input(tokenizer: Tokenizer, *, prompt: str | None,
                messages: list[dict[str, Any]] | None) -> int:
    if messages is not None:
        try:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"tokenizer cannot apply its chat template: {exc}") from exc
        return len(ids)
    assert prompt is not None
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def tokenizer_metadata(tokenizer: Tokenizer, identifier: str, revision: str | None) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None) or ""
    return {"id": identifier, "revision": revision,
            "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest()}


def exact_synthetic_text(tokenizer: Tokenizer, length: int, seed: int,
                         hash_ids: list[int] | None = None) -> str:
    """Build deterministic text whose re-tokenized length is exactly *length*."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return ""
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer.get_vocab()))
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    # Materialize token IDs from the model's actual vocabulary. A Mooncake hash ID
    # seeds each 512-token block, so equal prefix hashes produce equal synthetic blocks.
    for attempt in range(16):
        ids: list[int] = []
        block_count = (length + 511) // 512
        for block in range(block_count):
            block_key = hash_ids[block] if hash_ids and block < len(hash_ids) else block
            digest = hashlib.sha256(f"{seed}:{attempt}:{block_key}".encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            for _ in range(min(512, length - len(ids))):
                token_id = rng.randrange(vocab_size)
                while token_id in special: token_id = rng.randrange(vocab_size)
                ids.append(token_id)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if len(tokenizer.encode(text, add_special_tokens=False)) == length:
            return text
    # Some tokenizers do not round-trip arbitrary token streams. Find a decoded
    # vocabulary unit whose repetition is stable, still deriving text from valid IDs.
    start = seed % vocab_size
    for offset in range(min(vocab_size, 10000)):
        token_id = (start + offset) % vocab_size
        if token_id in special: continue
        unit = tokenizer.decode([token_id], skip_special_tokens=False)
        trial = unit * length
        if unit and len(tokenizer.encode(trial, add_special_tokens=False)) == length:
            return trial
    raise ValueError(f"cannot synthesize exactly {length} tokens for tokenizer {tokenizer.name_or_path}")
