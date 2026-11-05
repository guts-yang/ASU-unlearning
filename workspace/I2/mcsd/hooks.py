"""Activation boundaries shared by the ASU Llama editor and the fake model."""

from __future__ import annotations

import torch

from I2.mcsd.errors import HookBoundaryError


def path_key(module, layer, token, head=None):
    if module not in {"attention", "mlp"}:
        raise ValueError("path module must be attention or mlp, got %r" % (module,))
    if module == "attention" and head is None:
        raise ValueError("attention path requires a head id")
    if module == "mlp" and head is not None:
        raise ValueError("mlp path does not take a head id")
    return (module, int(layer), int(token), None if head is None else int(head))


def parse_path(item):
    if isinstance(item, tuple):
        if len(item) != 4:
            raise ValueError("path tuple must be (module, layer, token, head)")
        return path_key(*item)
    return path_key(item["module"], item["layer"], item["token"], item.get("head"))


def path_to_dict(path):
    module, layer, token, head = path
    return {"module": module, "layer": layer, "token": token, "head": head}


def mix_activations(source, donor, weight):
    """A(weight) = (1 - weight) * source + weight * donor. Weight must broadcast."""
    if source.shape != donor.shape:
        raise ValueError(
            "source shape %s does not match donor shape %s"
            % (tuple(source.shape), tuple(donor.shape))
        )
    if not torch.is_tensor(weight):
        weight = torch.tensor(weight, dtype=source.dtype, device=source.device)
    elif weight.dtype != source.dtype or weight.device != source.device:
        weight = weight.to(dtype=source.dtype, device=source.device)
    try:
        broadcast = torch.broadcast_shapes(tuple(weight.shape), tuple(source.shape))
    except RuntimeError as exc:
        raise ValueError(
            "weight shape %s does not broadcast to %s"
            % (tuple(weight.shape), tuple(source.shape))
        ) from exc
    if broadcast != tuple(source.shape):
        raise ValueError(
            "weight shape %s does not broadcast to %s"
            % (tuple(weight.shape), tuple(source.shape))
        )
    return (1 - weight) * source + weight * donor


def module_layer_index(module):
    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is None:
        attn = getattr(module, "self_attn", None)
        layer_idx = getattr(attn, "layer_idx", None) if attn is not None else None
    if layer_idx is None:
        raise HookBoundaryError("%s has no layer_idx" % type(module).__name__)
    return int(layer_idx)


def head_spec(module):
    heads = getattr(module, "num_heads", None)
    dim = getattr(module, "head_dim", None)
    if heads is None or dim is None:
        attn = getattr(module, "self_attn", None)
        if attn is not None:
            heads = getattr(attn, "num_heads", None)
            dim = getattr(attn, "head_dim", None)
    if heads is None or dim is None:
        raise HookBoundaryError(
            "%s does not expose num_heads and head_dim" % type(module).__name__
        )
    return int(heads), int(dim)


def to_canonical_attention(tensor, layout, num_heads, head_dim, module_name):
    """Return [batch, tokens, heads, head_dim] or fail. Never flatten a fused vector."""
    if layout == "bhsd":
        if tensor.ndim != 4 or tensor.shape[1] != num_heads or tensor.shape[-1] != head_dim:
            raise HookBoundaryError(
                "%s attention layout bhsd expected [batch, %s, tokens, %s], got %s"
                % (module_name, num_heads, head_dim, tuple(tensor.shape))
            )
        return tensor.permute(0, 2, 1, 3)
    if layout == "flash":
        if (
            tensor.ndim == 4
            and tensor.shape[2] == num_heads
            and tensor.shape[3] == head_dim
        ):
            return tensor
        raise HookBoundaryError(
            "%s FlashAttention output shape %s cannot be viewed as "
            "[batch, tokens, heads=%s, head_dim=%s] before concat; refusing to patch "
            "a fused residual as if it were a head output"
            % (module_name, tuple(tensor.shape), num_heads, head_dim)
        )
    raise HookBoundaryError("unknown attention layout %r" % (layout,))


def from_canonical_attention(canonical, layout):
    if layout == "bhsd":
        return canonical.permute(0, 2, 1, 3)
    if layout == "flash":
        return canonical
    raise HookBoundaryError("unknown attention layout %r" % (layout,))


class ActivationCache(object):
    def __init__(self):
        self.attention = {}
        self.mlp = {}
        self.shapes = []
        self.sample_ids = []


def _weight_view(live):
    if live.ndim == 4:
        return (live.shape[0], live.shape[1], live.shape[2], 1)
    if live.ndim == 3:
        return (live.shape[0], live.shape[1], 1)
    raise HookBoundaryError("activation rank %s is not a head or MLP output" % live.ndim)


def edit_sites(live, source, donor, placements):
    """Overwrite selected sites with a mix of clean caches. Other sites stay live."""
    if not placements:
        return live
    if source.shape != donor.shape:
        raise ValueError(
            "source shape %s does not match donor shape %s"
            % (tuple(source.shape), tuple(donor.shape))
        )
    if live.shape[0] != source.shape[0]:
        raise HookBoundaryError(
            "batch size %s does not match donor batch %s; refusing to reuse token indices"
            % (live.shape[0], source.shape[0])
        )
    tokens = source.shape[1]
    if live.shape[1] < tokens:
        raise HookBoundaryError("scored sequence is shorter than the donor cache")
    if live.ndim != source.ndim:
        raise HookBoundaryError(
            "live rank %s does not match donor rank %s" % (live.ndim, source.ndim)
        )
    weight_grid = live.new_zeros(_weight_view(live))
    selected = live.new_zeros(_weight_view(live))
    seen = set()
    for token, head, weight in placements:
        marker = (int(token), None if head is None else int(head))
        if marker in seen:
            raise ValueError("duplicate path token=%s head=%s" % marker)
        seen.add(marker)
        if token < 0 or token >= tokens:
            raise HookBoundaryError(
                "path token %s is outside the prompt donor cache of length %s"
                % (token, tokens)
            )
        if torch.is_tensor(weight):
            w = weight if weight.dtype == live.dtype and weight.device == live.device else weight.to(
                dtype=live.dtype, device=live.device
            )
        else:
            w = torch.tensor(float(weight), dtype=live.dtype, device=live.device)
        selector = live.new_zeros(_weight_view(live))
        if live.ndim == 4:
            if head is None:
                raise ValueError("attention path requires a head id")
            selector[:, token, head, :] = 1
        else:
            if head is not None:
                raise ValueError("mlp path has no head id")
            selector[:, token, :] = 1
        selected = selected + selector
        weight_grid = weight_grid + selector * w
    mixed = (1 - weight_grid[:, :tokens]) * source + weight_grid[:, :tokens] * donor
    edited = live[:, :tokens] * (1 - selected[:, :tokens]) + mixed * selected[:, :tokens]
    if live.shape[1] == tokens:
        return edited
    return torch.cat([edited, live[:, tokens:]], dim=1)


class ActivationEditor(object):
    """Record clean donors or patch them. A zero scalar strength writes nothing."""

    def __init__(self, mode, sample_ids, paths=None, source=None, donor=None, strength=1.0, weights=None):
        self.mode = mode
        self.sample_ids = list(sample_ids)
        self.paths = [] if paths is None else [parse_path(path) for path in paths]
        self.source = source
        self.donor = donor
        self.strength = strength
        self.weights = weights
        self.cache = ActivationCache()
        self.shapes = []
        self.last_live = None
        self.last_written = None
        self.snapshots = {}

    def __call__(self, kind, tensor, module, layout):
        if kind not in {"attention", "mlp"}:
            raise HookBoundaryError("unknown activation kind %r" % (kind,))
        layer = module_layer_index(module)
        module_name = type(module).__name__
        if kind == "attention":
            heads, dim = head_spec(module)
            canonical = to_canonical_attention(tensor, layout, heads, dim, module_name)
        else:
            if layout != "bth" or tensor.ndim != 3:
                raise HookBoundaryError(
                    "%s MLP hook expected layout bth and rank 3, got layout %s shape %s"
                    % (module_name, layout, tuple(tensor.shape))
                )
            canonical = tensor
        self.shapes.append(
            {
                "kind": kind,
                "layout": layout,
                "module": module_name,
                "layer": layer,
                "shape": list(tensor.shape),
                "canonical_shape": list(canonical.shape),
            }
        )
        if self.mode == "record":
            self._store(kind, layer, canonical)
            return tensor
        if self.mode != "patch":
            raise HookBoundaryError("unknown editor mode %r" % (self.mode,))
        self.last_live = canonical.detach().clone()
        self.snapshots.setdefault(kind, {})["live"] = self.last_live
        if self._inactive() or not self._placements(kind, layer):
            return tensor
        edited = self._patch(kind, layer, canonical)
        self.last_written = edited.detach().clone()
        self.snapshots[kind]["written"] = self.last_written
        if kind == "mlp":
            return edited
        return from_canonical_attention(edited, layout)

    def _inactive(self):
        if self.weights:
            return False
        if self.strength is None:
            return True
        if torch.is_tensor(self.strength):
            return False
        return float(self.strength) == 0.0

    def _store(self, kind, layer, canonical):
        if canonical.shape[0] != len(self.sample_ids):
            raise HookBoundaryError(
                "batch size %s does not match sample ids %s"
                % (canonical.shape[0], self.sample_ids)
            )
        bucket = self.cache.attention if kind == "attention" else self.cache.mlp
        for row, sample_id in enumerate(self.sample_ids):
            bucket[(sample_id, layer)] = canonical[row].detach().clone()
        self.cache.sample_ids = list(self.sample_ids)
        self.cache.shapes = list(self.shapes)

    def _placements(self, kind, layer):
        found = []
        for path in self.paths:
            if path[0] != kind or path[1] != layer:
                continue
            weight = self.weights[path] if self.weights is not None else self.strength
            found.append((path[2], path[3], weight))
        return found

    def _patch(self, kind, layer, canonical):
        source = _stack_cache(self.source, kind, layer, self.sample_ids)
        donor = _stack_cache(self.donor, kind, layer, self.sample_ids)
        return edit_sites(canonical, source, donor, self._placements(kind, layer))


def _stack_cache(cache, kind, layer, sample_ids):
    if cache is None:
        raise HookBoundaryError("patch is missing a clean activation cache")
    bucket = cache.attention if kind == "attention" else cache.mlp
    rows = []
    for sample_id in sample_ids:
        key = (sample_id, layer)
        if key not in bucket:
            raise HookBoundaryError(
                "missing %s donor for sample %s layer %s; refusing to reuse another row"
                % (kind, sample_id, layer)
            )
        rows.append(bucket[key])
    lengths = {int(row.shape[0]) for row in rows}
    if len(lengths) != 1:
        raise HookBoundaryError("refusing to align donor caches of different token lengths")
    return torch.stack(rows, dim=0)


def iter_layers(model):
    custom = getattr(model, "mcsd_layers", None)
    if custom is not None:
        return list(custom)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)
    raise HookBoundaryError("%s does not expose decoder layers" % type(model).__name__)


def bind_editor(model, editor):
    saved = []
    for layer in iter_layers(model):
        _reject_tensor_parallel(layer)
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            _reject_tensor_parallel(attn)
        modules = (layer, attn) if attn is not None else (layer,)
        for module in modules:
            saved.append((module, getattr(module, "activation_editor", None)))
            module.activation_editor = editor
    return saved


def restore_editor(saved):
    for module, previous in saved:
        module.activation_editor = previous


def _reject_tensor_parallel(module):
    config = getattr(module, "config", None)
    groups = getattr(config, "pretraining_tp", 1) or 1
    if int(groups) > 1:
        raise HookBoundaryError("pretraining_tp > 1 is not supported for activation patching")


def supports_differentiable_mask(model):
    flag = getattr(model, "mcsd_differentiable", None)
    if flag is True:
        return True
    if flag is False:
        return False
    config = getattr(model, "config", None)
    return getattr(config, "_attn_implementation", None) == "eager"


def layer_count(model):
    custom = getattr(model, "layer_count", None)
    if custom is not None:
        return int(custom)
    return len(iter_layers(model))
