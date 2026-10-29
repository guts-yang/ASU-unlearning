"""In-memory SwiGLU gate temperature for a loaded ASU Llama teacher.

τ_A is the existing ``attention_temp`` argument on
``Right-to-be-forgotten/my_models/my_llama.py``. τ_M divides the gate
pre-activation before SiLU. Both are no-ops at temperature 1, and the
original MLP ``forward`` is called unchanged in that case.
"""

from __future__ import annotations


class GateTemperatureTeacher:
    def __init__(self, model):
        self.model = model
        self.mlp_gate_temp = 1.0
        self.mlp_layers_id = None
        self._patched = []
        layers = model.model.layers
        for index, layer in enumerate(layers):
            mlp = layer.mlp
            original = mlp.forward

            def patched(hidden, *, _mlp=mlp, _index=index, _original=original, _owner=self):
                temperature = _owner.mlp_gate_temp
                selected = _owner.mlp_layers_id is None or _index in _owner.mlp_layers_id
                if temperature is None or float(temperature) == 1.0 or not selected:
                    return _original(hidden)
                if getattr(_mlp.config, "pretraining_tp", 1) > 1:
                    raise RuntimeError("gate temperature wrapper does not support pretraining_tp > 1")
                gate = _mlp.gate_proj(hidden) / float(temperature)
                return _mlp.down_proj(_mlp.act_fn(gate) * _mlp.up_proj(hidden))

            mlp.forward = patched
            self._patched.append((mlp, original))

    def restore(self):
        for mlp, original in self._patched:
            mlp.forward = original
        self._patched.clear()

    def forward(self, input_ids, attention_mask, attention_temp=1.0, mlp_gate_temp=1.0, layers_id=None, mlp_layers_id=None):
        self.mlp_gate_temp = 1.0 if mlp_gate_temp is None else float(mlp_gate_temp)
        self.mlp_layers_id = mlp_layers_id
        attention_layers = None if layers_id is None else list(layers_id)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_temp=None if attention_temp is None else float(attention_temp),
            layers_id=attention_layers,
            use_cache=False,
        )
