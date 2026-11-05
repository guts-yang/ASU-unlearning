"""Source contract for the default-off Llama activation editor."""

from __future__ import annotations

import unittest
from pathlib import Path


LLAMA = Path(__file__).resolve().parents[3] / "Right-to-be-forgotten" / "my_models" / "my_llama.py"


def _class_body(text, name):
    marker = "class %s" % name
    start = text.index(marker)
    next_class = text.find("\nclass ", start + len(marker))
    if next_class < 0:
        return text[start:]
    return text[start:next_class]


class LlamaEditorContractTests(unittest.TestCase):
    def test_editor_is_a_noop_and_sits_on_the_specified_boundaries(self):
        text = LLAMA.read_text()
        self.assertIn("if editor is None:\n        return tensor", text)
        eager = _class_body(text, "LlamaAttention")
        eager_editor = eager.index('_apply_activation_editor(self, "attention", attn_output, "bhsd")')
        self.assertLess(eager_editor, eager.index("attn_output.transpose(1, 2)"))
        self.assertLess(eager_editor, eager.index("attn_output.reshape(bsz, q_len, -1)"))
        flash = _class_body(text, "LlamaFlashAttention2")
        flash_editor = flash.index('_apply_activation_editor(self, "attention", attn_output, "flash")')
        self.assertLess(flash_editor, flash.index("attn_output.reshape(bsz, q_len, -1)"))
        sdpa = _class_body(text, "LlamaSdpaAttention")
        sdpa_editor = sdpa.index('_apply_activation_editor(self, "attention", attn_output, "bhsd")')
        self.assertLess(sdpa_editor, sdpa.index("attn_output.transpose(1, 2)"))
        decoder = _class_body(text, "LlamaDecoderLayer")
        mlp = decoder.index("self.mlp(hidden_states)")
        editor = decoder.index("_apply_activation_editor", mlp)
        residual = decoder.index("residual + hidden_states", editor)
        self.assertLess(mlp, editor)
        self.assertLess(editor, residual)
        self.assertLess(decoder.index("residual + hidden_states"), mlp)


if __name__ == "__main__":
    unittest.main()
