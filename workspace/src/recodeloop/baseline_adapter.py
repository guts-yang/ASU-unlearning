"""External adapter over the ASU baseline. No baseline file is modified."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from .checkpoints import checkpoint_refs, save_model_with_hash, verify_file_hash
from .config import FrozenConfig
from .errors import ContractViolation
from .types import CheckpointRef
from .utils.hashing import sha256_state_dict
from .utils.logging import JsonlLogger

ALLOWED_COMPONENTS = ("self_attn.o_proj", "mlp.down_proj")


class TinyTokenizer:
    """Deterministic in-memory tokenizer for CPU smoke tests; zero downloads."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 2
        self.padding_side = "left"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = []
        for token in str(text).split():
            value = sum(ord(ch) for ch in token) % (self.vocab_size - 2)
            tokens.append(value)
        return tokens or [self.eos_token_id]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"<{int(i)}>" for i in ids)

    def __call__(self, text: str | Sequence[str], return_tensors: str = "pt",
                 padding: bool = False, max_length: int | None = None,
                 truncation: bool = False) -> dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self.encode(t) for t in texts]
        if truncation and max_length:
            encoded = [e[:max_length] for e in encoded]
        width = max(len(e) for e in encoded)
        if max_length:
            width = min(width, max_length)
        input_ids, attention_mask = [], []
        for e in encoded:
            pad = width - len(e)
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad + e)
                attention_mask.append([0] * pad + [1] * len(e))
            else:
                input_ids.append(e + [self.pad_token_id] * pad)
                attention_mask.append([1] * len(e) + [0] * pad)
        ids = torch.tensor(input_ids, dtype=torch.long)
        mask = torch.tensor(attention_mask, dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": ids.tolist(), "attention_mask": mask.tolist()}

    def save_pretrained(self, path: Path | str) -> None:  # pragma: no cover - test stub
        Path(path).mkdir(parents=True, exist_ok=True)


class BaselineAdapter:
    """Wraps one model instance; provides forward/generate/component/freeze helpers."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any | None = None,
                 device: str = "cpu", dtype: torch.dtype = torch.float32,
                 logger: JsonlLogger | None = None) -> None:
        self.model = model.to(device=device, dtype=dtype)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.logger = logger
        vocab = getattr(self.model.config, "vocab_size", None)
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None and vocab is not None:
            if gen_cfg.pad_token_id is None or gen_cfg.pad_token_id >= vocab:
                fallback = getattr(self.model.config, "pad_token_id", None)
                if fallback is None or fallback >= vocab:
                    fallback = getattr(self.tokenizer, "eos_token_id", 0)
                gen_cfg.pad_token_id = int(fallback)
            gen_cfg.temperature = 1.0
            gen_cfg.top_p = 1.0
        if logger:
            logger.log("adapter.init", device=device, dtype=str(dtype),
                       model_hash=sha256_state_dict(self.model),
                       cuda_peak_bytes=torch.cuda.max_memory_allocated()
                       if torch.cuda.is_available() else 0)

    @classmethod
    def from_checkpoint(cls, ref: CheckpointRef, cfg: FrozenConfig,
                        logger: JsonlLogger | None = None,
                        device: str | None = None,
                        dtype: torch.dtype | None = None) -> "BaselineAdapter":
        """Load exactly one model from a G0-frozen checkpoint. No deep copies.

        device/dtype are explicit runtime overrides; they must be recorded by
        the caller and never written back to the frozen scientific config.
        """
        from .checkpoints import require_g0
        require_g0(cfg)
        if not verify_file_hash(ref.path, ref.sha256 or ""):
            # HF ids are not local files; hash verification is delegated to G0 manifest.
            if Path(ref.path).exists():
                raise ContractViolation(f"sha256 mismatch for {ref.path}")
        runtime = cfg.section("m0_p08_m2")["runtime"]
        device = str(device if device is not None else runtime["device"])
        dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(ref.path, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(ref.tokenizer_path)
        return cls(model, tokenizer, device=device, dtype=dtype, logger=logger)

    @classmethod
    def tiny(cls, vocab_size: int = 128, hidden_size: int = 32,
             intermediate_size: int = 64, layers: int = 2,
             heads: int = 4, device: str = "cpu") -> "BaselineAdapter":
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=max(1, heads // 2),
            max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
        return cls(model, TinyTokenizer(vocab_size), device=device, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, **kwargs: Any) -> Any:
        return self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            labels=labels.to(self.device) if labels is not None else None,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8,
                 do_sample: bool = False, seed: int | None = None,
                 **kwargs: Any) -> torch.Tensor:
        # Baseline run_generation() hardcodes manual_seed(0); here seeding is explicit.
        if seed is not None:
            torch.manual_seed(seed)
        return self.model.generate(
            input_ids=input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            **kwargs,
        )

    def find_all_linear_names(self) -> tuple[str, ...]:
        """Full dotted-path variant of forget.py::find_all_linear_names (leaf names)."""
        names = {
            name for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name
        }
        return tuple(sorted(names))

    def component_paths(self, kinds: Sequence[str] = ALLOWED_COMPONENTS) -> tuple[str, ...]:
        allowed = tuple(kinds)
        return tuple(sorted(
            name for name in self.find_all_linear_names()
            if name.endswith(allowed)
        ))

    @contextmanager
    def frozen(self) -> Iterator[Mapping[str, bool]]:
        snapshot = {name: p.requires_grad for name, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        try:
            yield snapshot
        finally:
            for name, p in self.model.named_parameters():
                if name in snapshot:
                    p.requires_grad_(snapshot[name])

    def restore_grad(self, snapshot: Mapping[str, bool]) -> None:
        for name, p in self.model.named_parameters():
            if name in snapshot:
                p.requires_grad_(snapshot[name])

    def state_hash(self) -> str:
        return sha256_state_dict(self.model)

    def save(self, output_dir: Path | str) -> str:
        return save_model_with_hash(self.model, output_dir)

    def retain_loss_tensor(self, input_ids: torch.Tensor, labels: torch.Tensor,
                           attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Differentiable baseline retain NLL (losses.py::gd_loss, -100 ignore_index)."""
        out = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
        if out.loss is None:
            raise ContractViolation("retain loss requires labels producing a loss")
        return out.loss

    def retain_nll(self, input_ids: torch.Tensor, labels: torch.Tensor,
                   attention_mask: torch.Tensor | None = None) -> float:
        return float(self.retain_loss_tensor(input_ids, labels, attention_mask).detach().item())
