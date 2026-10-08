import copy
import sys
from pathlib import Path

import deepspeed
from transformers import Trainer

from .losses import get_loss

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "workspace") not in sys.path:
    sys.path.insert(0, str(_REPO / "workspace"))

from da_asu.anchor import derivation_anchor_loss, derivation_logit_kl
from da_asu.protocol import normalize_layers_id


class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('loss_type')
        self.ref_model = kwargs.pop('ref_model')

        # the coefficient of each part in the loss function. This is used in ablation study.
        self.forget_coeff = kwargs.pop('forget_coeff')
        self.regularization_coeff = kwargs.pop('regularization_coeff')
        # beta for NPO/DPO/RS
        self.beta = kwargs.pop('beta')

        # layers_id for ASU. Empty list would disable temperature; treat it as every layer.
        self.layers_id = normalize_layers_id(kwargs.pop('layers_id', None))
        # attention temperature for ASU
        self.attention_temp = kwargs.pop('attention_temp')
        self.da_asu_mode = kwargs.pop('da_asu_mode', None)
        self.da_asu_mu = kwargs.pop('da_asu_mu', 1.0)

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        self.ref_model = self.e_prepare_deepspeed(self.ref_model)

    def compute_loss(self, model, inputs, return_outputs=False):

        forget_loss, regularization_loss = get_loss(model, self.ref_model, inputs, self.loss_type, self.beta, self.attention_temp, self.layers_id)
        loss = self.forget_coeff * forget_loss + self.regularization_coeff * regularization_loss
        if self.da_asu_mode:
            if len(inputs) < 6:
                raise RuntimeError("da_asu is enabled but the batch has no D_reason tensors")
            input_ids, labels, attention_mask, query_mask = inputs[5]
            if self.da_asu_mode == "anchor":
                extra, self.last_anchor_per_layer = derivation_anchor_loss(
                    model, self.ref_model, input_ids, attention_mask, query_mask
                )
            elif self.da_asu_mode == "logit_control":
                extra = derivation_logit_kl(
                    model, self.ref_model, input_ids, labels, attention_mask, query_mask
                )
            else:
                raise ValueError(f"unknown da_asu mode {self.da_asu_mode}")
            loss = loss + self.da_asu_mu * extra
            step = getattr(self.state, "global_step", 0)
            if step % 10 == 0:
                message = f"da_asu {self.da_asu_mode} loss {float(extra.detach()):.4f}"
                if self.da_asu_mode == "anchor":
                    per_layer = [round(float(value), 4) for value in self.last_anchor_per_layer]
                    message += f" per_layer {per_layer}"
                print(message)

        return (loss, None) if return_outputs else loss

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        # set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False

        return model
