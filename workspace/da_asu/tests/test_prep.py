import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workspace"))
sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))

from da_asu.anchor import derivation_anchor_loss, derivation_logit_kl, pad_query_index, rows_at_layer
from da_asu.kill_switch import decide
from da_asu.labels import encode_qa, queries_overlapping, result_char_spans
from da_asu.protocol import EFFECT_SIZE_KILL, TAU_GRID, normalize_layers_id
from da_asu.run_m0b import h2_decision
from da_asu.scan import scan_records
from my_models.my_llama import LlamaForCausalLM

TOKENIZER = (
    "/root/autodl-tmp/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct/"
    "snapshots/53346005fb0ef11d3b6a83b12c895cca40156b6c"
)
LLAMA3 = {
    "question_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
    "question_end_tag": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "answer_tag": "",
}


def test_empty_layers_id_means_every_layer():
    assert normalize_layers_id([]) is None
    assert normalize_layers_id(None) is None
    assert normalize_layers_id("null") is None
    assert normalize_layers_id([0, 2]) == [0, 2]


def test_kill_switch_uses_function_not_factual():
    taus = TAU_GRID
    steep = [0.2 * (tau - 1.0) for tau in taus]
    flat = [0.01 * (tau - 1.0) for tau in taus]
    derivation = {i: steep for i in range(30)}
    function = {i: flat for i in range(30)}
    factual = {i: steep for i in range(30)}
    verdict = decide({"derivation": _slopes(derivation), "function": _slopes(function), "factual": _slopes(factual)})
    assert verdict["decision"] == "proceed"
    assert verdict["mechanism"]["kill"] is False
    assert verdict["mechanism"]["cohens_d"] > EFFECT_SIZE_KILL
    assert verdict["taxonomy"]["kill"] is False
    assert verdict["taxonomy"]["role"] == "report_only"
    assert verdict["entropy_used"] is False

    same = decide({
        "derivation": _slopes(function),
        "function": _slopes(function),
        "factual": _slopes({i: steep for i in range(30)}),
    })
    assert same["decision"] == "kill"
    assert same["mechanism"]["kill"] is True
    assert same["taxonomy"]["kill"] is False


def test_h2_stops_when_attention_does_not_blur():
    assert h2_decision([0.0] * 25)["decision"] == "stop_anchor"
    assert h2_decision([0.05] * 25)["decision"] == "proceed"
    assert h2_decision([0.05] * 3)["decision"] == "insufficient_data"


def test_llama3_query_is_the_position_that_predicts_the_result():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    answer = "She sold <<48/2=1234567890>> clips."
    input_ids, labels, attention, query_mask, full_text, offsets = encode_qa(
        tokenizer, "How many?", answer, LLAMA3, max_length=128
    )
    spans = result_char_spans(answer)
    assert spans == [(len("She sold <<48/2="), len("She sold <<48/2=") + len("1234567890"))]
    answer_start = full_text.index(answer)
    expected = queries_overlapping(offsets, [(answer_start + spans[0][0], answer_start + spans[0][1])])
    got = query_mask.nonzero(as_tuple=False).flatten().tolist()
    assert got == expected
    assert len(got) >= 2
    for query in got:
        assert labels[query + 1] != -100
        piece = tokenizer.decode([int(input_ids[query + 1])])
        assert piece.strip() != ""
        start, end = offsets[query + 1]
        assert start < answer_start + spans[0][1] and end > answer_start + spans[0][0]
        assert query != query + 1


def _slopes(curves):
    from da_asu.kill_switch import problem_slopes

    return problem_slopes(curves, TAU_GRID)


def _tiny():
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def test_selected_rows_match_eager_attention():
    torch.manual_seed(0)
    model = _tiny()
    input_ids = torch.randint(0, 128, (1, 12))
    attention_mask = torch.ones(1, 12, dtype=torch.long)
    query_mask = torch.zeros(1, 12, dtype=torch.bool)
    query_mask[0, [2, 5, 9]] = True
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        position_ids = torch.arange(12).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(outputs.hidden_states[0], position_ids)
        query_index = pad_query_index(query_mask)
        for layer_id, layer in enumerate(model.model.layers):
            rows = rows_at_layer(
                layer, outputs.hidden_states[layer_id], position_embeddings, query_index, attention_mask
            )
            full = outputs.attentions[layer_id]
            for slot, query in enumerate(query_index[0].tolist()):
                assert torch.allclose(rows[0, :, slot], full[0, :, query], atol=1e-5)


def test_anchor_backward_and_logit_control_do_not_share_a_graph():
    torch.manual_seed(1)
    student = _tiny()
    base = _tiny()
    student.train()
    input_ids = torch.randint(0, 128, (2, 10))
    labels = input_ids.clone()
    labels[:, :2] = -100
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    query_mask = torch.zeros(2, 10, dtype=torch.bool)
    query_mask[:, 4] = True
    loss, per_layer = derivation_anchor_loss(student, base, input_ids, attention_mask, query_mask)
    loss.backward()
    grad = student.model.layers[0].self_attn.q_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert per_layer.shape[0] == 2
    assert per_layer.grad_fn is None

    student.zero_grad()
    logit = derivation_logit_kl(student, base, input_ids, labels, attention_mask, query_mask)
    logit.backward()
    assert student.lm_head.weight.grad is not None


def test_scan_on_tiny_model_returns_a_verdict():
    torch.manual_seed(2)
    model = _tiny()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    class Narrow:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, text, **kwargs):
            encoded = self.inner(text, **kwargs)
            encoded["input_ids"] = [token % 128 for token in encoded["input_ids"]]
            return encoded

        def tokenize(self, text, add_special_tokens=True):
            return self.inner.tokenize(text, add_special_tokens=add_special_tokens)

        @property
        def eos_token_id(self):
            return 2

        def __getattr__(self, name):
            return getattr(self.inner, name)

    records = []
    for index in range(4):
        records.append(
            {
                "question": f"Q{index}",
                "answer": f"Step <<1+1={index + 10}>> done.",
                "factual_spans": [(0, 4)],
                "function_spans": [(5, 9)],
            }
        )
    result = scan_records(model, Narrow(tokenizer), records, LLAMA3, max_length=64, taus=(1.0, 2.0))
    assert result["verdict"]["decision"] in {"kill", "proceed", "insufficient_data"}
    assert result["verdict"]["entropy_used"] is False
    assert len(result["curves"]) == 4


def test_gsm8k_spans_keep_derivation_out_of_function_and_factual():
    from da_asu.prepare_records import records_from_rows
    from da_asu.spans import label_answer

    answer = "She sold <<48/2=24>>24 clips in May.\n#### 24"
    factual, function = label_answer(answer)
    assert factual == [(answer.index("####") + len("#### "), answer.index("####") + len("#### 24"))]
    assert any(answer[start:end] == "in" for start, end in function)
    for start, end in factual + function:
        assert not (start >= answer.index("<<") and end <= answer.index(">>") + 2)
    records = records_from_rows([{"question": "How many?", "answer": answer}], "test")
    assert records[0]["factual_spans"][0][0] < records[0]["factual_spans"][0][1]
    assert records[0]["id"] == "test-0"
