"""RIU plug-in trainer for OpenUnlearning.

Adds recovery-invariant objectives on top of a base unlearning algorithm:

    L = L_base(GA | NPO | ASU) + λ1 L_probe(GRL) + λ2 L_cons + λ3 L_geom

The linear probe is attached to the model *before* the HF Trainer constructor
so that its parameters enter the optimizer.  All RIU hyper-parameters are
passed through Hydra `trainer.method_args`.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_dpo_loss, compute_kl_divergence

from open_unlearning_plugins.asu.attention import attention_temperature
from open_unlearning_plugins.asu.trainer import ASU
from open_unlearning_plugins.riu.losses import (
    RECOVERY_OPS,
    LinearProbe,
    hsic_linear_loss,
    masked_mean,
    riu_consistency_loss,
    riu_fact_probe_loss,
    riu_geometry_loss,
    riu_probe_loss,
    update_fact_direction,
)

logger = logging.getLogger("riu")


class RIU(GradDiff):
    def __init__(
        self,
        base: str = "ASU",
        attention_temp: float = 2.3,
        layers_id: Any = None,
        beta: float = 0.1,
        probe_layer: int = 7,
        probe_hidden: int = 64,
        riu_objective: str = "grl",
        probe_target: str = "both",
        num_classes: int = 10,
        fact_id_stride: int = 20,
        probe_weight: float = 0.1,
        info_weight: float = 0.1,
        grl_lambda: float = 0.1,
        cons_weight: float = 0.1,
        geom_weight: float = 0.1,
        geom_mode: str = "corrected",
        geom_momentum: float = 0.9,
        recovery_ops: Any = "identity,noise,round,amplify",
        recovery_noise: float = 0.05,
        recovery_amplify: float = 0.2,
        recovery_dropout: float = 0.1,
        riu_log_every: int = 0,
        riu_debug_grad: int = 0,
        riu_grad_dump: str = "",
        *args,
        **kwargs,
    ):
        model = kwargs.get("model", None)
        probe = None
        fact_probe = None
        if model is not None:
            base_model = getattr(model, "module", model)
            config = getattr(base_model, "config", None)
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("RIU needs model.config.hidden_size to build the probe")
            # Register on the model so HF Trainer's optimizer includes probe params.
            probe = LinearProbe(int(hidden_size), int(probe_hidden))
            model.riu_probe = probe
            if str(probe_target) in {"fact", "both"}:
                fact_probe = LinearProbe(
                    int(hidden_size), int(probe_hidden), out_dim=int(num_classes)
                )
                model.riu_fact_probe = fact_probe

        super().__init__(*args, **kwargs)

        self.base = str(base).upper()
        if self.base not in {"GA", "NPO", "ASU"}:
            raise ValueError(f"Unsupported RIU base: {base!r} (use GA/NPO/ASU)")
        self.attention_temp = float(attention_temp)
        if layers_id is None or layers_id == [] or layers_id == "null":
            self.layers_id = None
        else:
            self.layers_id = [int(x) for x in layers_id]
        self.beta = float(beta)
        self.probe_layer = int(probe_layer)
        self.riu_objective = str(riu_objective)
        if self.riu_objective not in {"grl", "info"}:
            raise ValueError(f"Unknown riu_objective: {riu_objective!r}")
        self.probe_target = str(probe_target)
        if self.probe_target not in {"membership", "fact", "both"}:
            raise ValueError(f"Unknown probe_target: {probe_target!r}")
        self.num_classes = int(num_classes)
        self.fact_id_stride = max(1, int(fact_id_stride))
        self.probe_weight = float(probe_weight)
        self.info_weight = float(info_weight)
        self.grl_lambda = float(grl_lambda)
        self.cons_weight = float(cons_weight)
        self.geom_weight = float(geom_weight)
        self.geom_mode = str(geom_mode)
        self.geom_momentum = float(geom_momentum)
        self.recovery_noise = float(recovery_noise)
        self.recovery_amplify = float(recovery_amplify)
        self.recovery_dropout = float(recovery_dropout)
        self.riu_log_every = int(riu_log_every)
        self.riu_debug_grad = int(riu_debug_grad)
        self.riu_grad_dump = str(riu_grad_dump or "")
        if isinstance(recovery_ops, str):
            ops = tuple(op.strip() for op in recovery_ops.split(",") if op.strip())
        else:
            ops = tuple(str(op) for op in recovery_ops)
        self.recovery_ops = ops or ("identity",)
        for op in self.recovery_ops:
            if op not in RECOVERY_OPS:
                raise ValueError(f"Unknown recovery op {op!r}; choose from {RECOVERY_OPS}")

        self.riu_probe = probe if probe is not None else getattr(self.model, "riu_probe")
        self.riu_probe.to(self.accelerator.device)
        self.riu_fact_probe = None
        if self.probe_target in {"fact", "both"}:
            self.riu_fact_probe = (
                fact_probe
                if fact_probe is not None
                else getattr(self.model, "riu_fact_probe", None)
            )
            if self.riu_fact_probe is not None:
                self.riu_fact_probe.to(self.accelerator.device)
        if self.base in {"ASU", "NPO"} and getattr(self, "ref_model", None) is None:
            self.ref_model = self._prepare_ref_model(self.model)
        self.riu_metrics: dict[str, float] = {}

    @staticmethod
    def _clean(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

    def _pool(self, outputs, labels, tag: str) -> torch.Tensor:
        hidden = getattr(outputs, "hidden_states", None)
        if hidden is None:
            raise RuntimeError(
                "RIU requires output_hidden_states=True; the loaded model may "
                "not expose hidden states."
            )
        idx = self.probe_layer + 1  # hidden_states[0] is the embedding layer
        if idx >= len(hidden):
            raise ValueError(
                f"probe_layer={self.probe_layer} exceeds model layers ({len(hidden) - 1})"
            )
        return masked_mean(hidden[idx], labels)

    def _dump_dec_gradients(self, model, dec_loss: torch.Tensor) -> None:
        """Attribute gradients of L_dec to backbone layers (diagnostic only)."""
        names = [name for name, param in model.named_parameters() if param.requires_grad]
        params = [param for _, param in model.named_parameters() if param.requires_grad]
        # torch.utils.checkpoint (reentrant) does not support autograd.grad;
        # use a real backward with saved/restored grads to avoid disturbing
        # gradient accumulation performed by the HF Trainer.
        saved = [param.grad.detach().clone() if param.grad is not None else None for param in params]
        try:
            dec_loss.backward(retain_graph=True)
            by_layer: dict[str, float] = {}
            total_sq = 0.0
            for name, param in zip(names, params):
                if "riu_" in name or param.grad is None:
                    continue
                sq = float(param.grad.detach().float().norm()) ** 2
                match = re.search(r"layers\.(\d+)\.", name)
                key = f"layer_{int(match.group(1))}" if match else "other"
                by_layer[key] = by_layer.get(key, 0.0) + sq
                total_sq += sq
        finally:
            for param, grad in zip(params, saved):
                param.grad = grad
        payload = {
            "step": self.state.global_step,
            "dec_grad_total_l2": total_sq**0.5,
            "by_layer_l2": {k: v**0.5 for k, v in sorted(by_layer.items())},
        }
        logger.info("RIU dec gradient diagnostic %s", payload)
        if self.riu_grad_dump:
            path = Path(self.riu_grad_dump)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    def _base_and_hidden(
        self, model, forget_inputs, retain_inputs
    ):
        forget_outputs = model(**forget_inputs, output_hidden_states=True)

        if self.base == "GA":
            forget_loss = -forget_outputs.loss
        elif self.base == "ASU":
            with torch.no_grad():
                with attention_temperature(
                    self.ref_model, self.attention_temp, self.layers_id
                ):
                    ref_forget = self.ref_model(**forget_inputs)
            forget_loss = ASU._token_kl(
                self, forget_outputs.logits, ref_forget.logits, forget_inputs["labels"]
            )
        else:  # NPO
            forget_loss, _ = compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=self.beta,
            )

        retain_outputs = model(**retain_inputs, output_hidden_states=True)
        if self.retain_loss_type == "NLL":
            retain_loss = retain_outputs.loss
        elif self.retain_loss_type == "KL":
            retain_loss, _ = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
        else:
            raise NotImplementedError(
                f"retain_loss_type={self.retain_loss_type!r} not supported by RIU"
            )

        base_loss = self.gamma * forget_loss + self.alpha * retain_loss
        return base_loss, forget_outputs, retain_outputs

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        forget_inputs = self._clean(inputs["forget"])
        retain_inputs = self._clean(inputs["retain"])

        base_loss, forget_outputs, retain_outputs = self._base_and_hidden(
            model, forget_inputs, retain_inputs
        )
        h_f = self._pool(forget_outputs, forget_inputs["labels"], "forget")
        h_r = self._pool(retain_outputs, retain_inputs["labels"], "retain")
        # Raw hidden states have norm ~50 and saturate the probe logits; RIU
        # works with direction, not magnitude, so normalize for its own terms.
        h_f = F.normalize(h_f, dim=-1)
        h_r = F.normalize(h_r, dim=-1)

        self.riu_probe.fact_dir = update_fact_direction(
            self.riu_probe.fact_dir,
            h_f,
            h_r,
            self.geom_momentum,
        )

        probe_loss = h_f.new_zeros(())
        fact_probe_loss = h_f.new_zeros(())
        info_loss = h_f.new_zeros(())
        indices = inputs["forget"].get("index", None)
        fact_labels = None
        if indices is not None:
            fact_labels = indices.to(h_f.device).long() // self.fact_id_stride

        if self.riu_objective == "grl":
            if self.probe_target in {"membership", "both"}:
                probe_loss = riu_probe_loss(
                    h_f, h_r, self.riu_probe, self.recovery_ops, self.grl_lambda,
                    self.recovery_noise, self.recovery_amplify, self.recovery_dropout,
                )
            if (
                self.probe_target in {"fact", "both"}
                and self.riu_fact_probe is not None
                and fact_labels is not None
            ):
                fact_probe_loss = riu_fact_probe_loss(
                    h_f, fact_labels, self.riu_fact_probe, self.num_classes,
                    self.recovery_ops, self.grl_lambda,
                    self.recovery_noise, self.recovery_amplify, self.recovery_dropout,
                )
        else:
            # Probe-free information-theoretic objective: suppress linear
            # dependence between representations and fact / membership labels.
            if fact_labels is not None:
                info_loss = info_loss + hsic_linear_loss(h_f, fact_labels)
            joint = torch.cat([h_f, h_r], dim=0)
            membership = torch.cat(
                [h_f.new_ones(h_f.shape[0]), h_r.new_zeros(h_r.shape[0])], dim=0
            )
            info_loss = info_loss + hsic_linear_loss(joint, membership)
        if self.cons_weight > 0:
            op_a, op_b = random.sample(list(RECOVERY_OPS), 2)
            cons_loss = riu_consistency_loss(
                h_f,
                op_a,
                op_b,
                self.recovery_noise,
                self.recovery_amplify,
                self.recovery_dropout,
            )
        else:
            cons_loss = h_f.new_zeros(())
        geom_loss = (
            riu_geometry_loss(h_f, self.riu_probe.fact_dir, self.geom_mode)
            if self.geom_weight > 0
            else h_f.new_zeros(())
        )

        dec_terms = (
            self.probe_weight * (probe_loss + fact_probe_loss)
            if self.riu_objective == "grl"
            else self.info_weight * info_loss
        )
        loss = base_loss + dec_terms + self.cons_weight * cons_loss + self.geom_weight * geom_loss

        if self.riu_debug_grad:
            if self.state.global_step == 0 or self.state.global_step % self.riu_debug_grad == 0:
                self._dump_dec_gradients(model, dec_terms)

        self.riu_metrics = {
            "base": float(base_loss.detach()),
            "probe": float(probe_loss.detach()),
            "fact_probe": float(fact_probe_loss.detach()),
            "info": float(info_loss.detach()),
            "cons": float(cons_loss.detach()),
            "geom": float(geom_loss.detach()),
        }
        if self.riu_log_every and self.state.global_step % self.riu_log_every == 0:
            logger.info("RIU step %s metrics=%s", self.state.global_step, self.riu_metrics)
            try:
                self.log({f"riu/{key}": value for key, value in self.riu_metrics.items()})
            except Exception as exc:  # logging must never break training
                logger.debug("RIU log skipped: %s", exc)

        return (loss, forget_outputs) if return_outputs else loss


class RIU_GA(RIU):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("base", "GA")
        super().__init__(*args, **kwargs)


class RIU_NPO(RIU):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("base", "NPO")
        super().__init__(*args, **kwargs)


class RIU_ASU(RIU):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("base", "ASU")
        super().__init__(*args, **kwargs)
