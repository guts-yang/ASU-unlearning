"""Pre-registered DA-ASU decisions.

Mechanism kill uses derivation versus function only. The derivation versus
factual contrast is reported and never kills the project. Attention-entropy
curves are not a kill criterion: a global temperature flattens every row.
"""

TAU_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

# |Cohen's d| below this, or a 95% CI on the slope gap that covers 0, kills H1.
EFFECT_SIZE_KILL = 0.2
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
MIN_PAIRED_PROBLEMS = 20

# M1 existence bar, relative to ASU_KL, and only if the logit-control run is lower.
M1_GSM8K_GAIN = 0.03

MECHANISM_CONTRAST = ("derivation", "function")
TAXONOMY_CONTRAST = ("derivation", "factual")


def normalize_layers_id(layers_id):
    """Treat a missing or empty layer list as every layer.

    The model applies temperature only when ``layers_id is None`` or the index
    is in the list. An empty list therefore disables temperature. Callers that
    want the paper default must pass null; this helper makes that explicit.
    """
    if layers_id is None:
        return None
    if isinstance(layers_id, str) and layers_id.strip().lower() in {"", "null", "none"}:
        return None
    try:
        values = list(layers_id)
    except TypeError:
        return layers_id
    if len(values) == 0:
        return None
    return values
