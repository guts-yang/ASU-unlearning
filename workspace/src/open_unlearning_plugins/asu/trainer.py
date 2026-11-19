"""ASU (Attention Smoothing Unlearning) on OpenUnlearning GradDiff stack.

Forget: KL(p_τ || p) with frozen teacher under attention temperature τ.
Retain: NLL (GD) or KL to frozen ref — default NLL to match ASU+GD.
loss = alpha * L_forget + L_retain  (paper λ / forget_coeff → alpha).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff

from open_unlearning_plugins.asu.attention import attention_temperature


class ASU(GradDiff):
    def __init__(
        self,
        attention_temp: float = 2.3,
        layers_id=None,
        alpha: float = 0.1,
        gamma: float = 1.0,
        retain_loss_type: str = "NLL",
        *args,
        **kwargs,
    ):
        # Always need a frozen teacher; force ref even when retain is NLL.
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            retain_loss_type=retain_loss_type,
            *args,
            **kwargs,
        )
        self.attention_temp = float(attention_temp)
        if layers_id is None or layers_id == [] or layers_id == "null":
            self.layers_id = None
        else:
            self.layers_id = [int(x) for x in layers_id]
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def _token_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        log_p = F.log_softmax(student_logits[..., :-1, :], dim=-1)
        log_q = F.log_softmax(teacher_logits[..., :-1, :], dim=-1)
        mask = labels[..., 1:] != -100
        # KL(teacher || student): F.kl_div(input=student, target=teacher, log_target=True)
        token_kl = F.kl_div(log_p, log_q, reduction="none", log_target=True).sum(-1)
        denom = mask.sum(-1).clamp_min(1)
        return ((token_kl * mask).sum(-1) / denom).mean()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        with torch.no_grad():
            with attention_temperature(
                self.ref_model, self.attention_temp, self.layers_id
            ):
                ref_forget = self.ref_model(**forget_inputs)
        forget_loss = self._token_kl(
            forget_outputs.logits, ref_forget.logits, forget_inputs["labels"]
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # alpha = λ (forget weight); retain unweighted (matches Rtbf forget_coeff * L_f + L_r)
        loss = self.alpha * forget_loss + retain_loss
        return (loss, forget_outputs) if return_outputs else loss
