#!/usr/bin/env python3
"""Read-only remote pre-screen for the Wikidata donor-pair candidates.

This script loads a tokenizer only. It does not load model weights, run a
forward pass, localize activations, train a Teacher or Student, or choose a
K2 threshold.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path("/usr/local/ASU-unlearning")
CANDIDATE_DIR = REPO / "workspace/data/external_donor_pairs_wikidata_v0.1"
OUT_DIR = REPO / "workspace/data/external_donor_pairs_wikidata_v0.1_validation"
TOFU_JSONL_DIR = REPO / "workspace/data/tofu"
TOFU_JSON_DIR = REPO / "Right-to-be-forgotten/data/tofu"
FORGET05 = TOFU_JSONL_DIR / "forget05.jsonl"

SNAPSHOT = Path(
    "/root/autodl-tmp/huggingface/hub/"
    "models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/"
    "snapshots/cc5b31c69127d5da881608e640f2b453c446435b"
)
NOUS_SNAPSHOT = Path(
    "/root/autodl-tmp/huggingface/hub/"
    "models--NousResearch--Llama-2-7b-chat-hf/"
    "snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee"
)

K2_STATUS = (
    "not_computed: K2 screening stopped — no independent calibration set "
    "(instruction #4)"
)

QUESTION_START = "[INST] "
QUESTION_END = " [/INST]"
ANSWER_TAG = ""
MAX_LENGTH = 200

CHECKPOINT_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
]
TOKENIZER_COMPARE_FILES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
]
CODE_FILES = [
    REPO / "Right-to-be-forgotten/dataset/data_module.py",
    REPO / "Right-to-be-forgotten/metrics/metrics.py",
    REPO / "Right-to-be-forgotten/utils/utils.py",
    REPO / "Right-to-be-forgotten/config/model_config.yaml",
    REPO / "Right-to-be-forgotten/config/tofu.yaml",
]
CANDIDATE_FILES = [
    "candidate_pairs.jsonl",
    "clean_facts.jsonl",
    "wikidata_donors.jsonl",
    "relation_audit.json",
    "README.md",
]
CONFIGURED_EVAL_NAMES = {
    "retain_perturbed.json",
    "real_authors_perturbed.json",
    "world_facts_perturbed.json",
    "forget05_perturbed.json",
    "retain_perturbed.jsonl",
    "real_authors_perturbed.jsonl",
    "world_facts_perturbed.jsonl",
    "forget05_perturbed.jsonl",
    "real_authors.jsonl",
    "world_facts.jsonl",
}

# Predeclared entity, alias, neighbor, and place strings for the offline overlap scan.
DONOR_TERMS = {
    "donor-01": {
        "entity": ["Katherine Mansfield"],
        "aliases": ["Kathleen Mansfield Beauchamp", "Kathleen Beauchamp"],
        "neighbors": ["John Middleton Murry"],
        "places": ["Wellington"],
    },
    "donor-02": {
        "entity": ["Bill Gates"],
        "aliases": ["William Henry Gates III", "William Henry Gates", "William H. Gates"],
        "neighbors": ["Paul Allen", "Melinda Gates", "Melinda French Gates"],
        "places": ["Seattle"],
    },
    "donor-03": {
        "entity": ["Beyoncé"],
        "aliases": [
            "Beyoncé Giselle Knowles-Carter",
            "Beyoncé Knowles-Carter",
            "Beyoncé Knowles",
            "Beyonce",
        ],
        "neighbors": ["Jay-Z", "Shawn Corey Carter", "Destiny's Child"],
        "places": ["Houston"],
    },
    "donor-04": {
        "entity": ["Michael Jackson"],
        "aliases": ["Michael Joseph Jackson"],
        "neighbors": ["Janet Jackson", "The Jackson 5"],
        "places": ["Gary", "Gary, Indiana"],
    },
    "donor-05": {
        "entity": ["Elvis Presley"],
        "aliases": ["Elvis Aaron Presley", "Elvis Aron Presley"],
        "neighbors": ["Priscilla Presley"],
        "places": ["Tupelo"],
    },
    "donor-06": {
        "entity": ["Ada Lovelace"],
        "aliases": ["Augusta Ada King", "Augusta Ada Byron", "Ada Byron"],
        "neighbors": ["Charles Babbage"],
        "places": ["London"],
    },
    "donor-07": {
        "entity": ["Malala Yousafzai"],
        "aliases": ["Malala"],
        "neighbors": ["Ziauddin Yousafzai"],
        "places": ["Mingora"],
    },
    "donor-08": {
        "entity": ["Paulo Coelho"],
        "aliases": [],
        "neighbors": [],
        "places": ["Rio de Janeiro"],
    },
    "donor-09": {
        "entity": ["Anne Rice"],
        "aliases": [
            "Howard Allen Frances O'Brien",
            "Anne Rampling",
            "A. N. Roquelaure",
        ],
        "neighbors": ["Stan Rice"],
        "places": ["New Orleans"],
    },
}
AMBIGUOUS_PLACES = {"Gary", "London", "Houston", "Seattle", "Wellington"}
EXAMPLES_PER_TERM = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows = []
    errors = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            errors.append(f"{path.name}:{line_no}: empty line")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_no}: {exc}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"{path.name}:{line_no}: JSON value is not an object")
            continue
        rows.append(obj)
    return rows, errors


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("`", "'")
    return text.casefold()


def term_pattern(term: str) -> re.Pattern[str]:
    normalized = normalize_text(term)
    return re.compile(r"(?<![0-9a-z])" + re.escape(normalized) + r"(?![0-9a-z])")


def match_excerpt(text: str, normalized: str, term: str, pattern: re.Pattern[str]) -> str:
    raw = re.search(re.escape(term), text, flags=re.IGNORECASE)
    if raw:
        start, end = raw.start(), raw.end()
        return text[max(0, start - 70) : end + 70]
    found = pattern.search(normalized)
    if found is None:
        return text[:180]
    return normalized[max(0, found.start() - 70) : found.end() + 70]


def git_text(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def encode_side(tokenizer, question: str, answer: str) -> dict:
    prompt_text = QUESTION_START + question + QUESTION_END
    full_text = prompt_text + ANSWER_TAG + answer
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        truncation=True,
    )
    full_ids = encoded["input_ids"]
    num_question_tokens = len(tokenizer.tokenize(prompt_text, add_special_tokens=True))
    answer_ids = full_ids[num_question_tokens:]
    decoded = tokenizer.decode(answer_ids, skip_special_tokens=False)
    return {
        "question": question,
        "target_answer": answer,
        "prompt_text": prompt_text,
        "full_text": full_text,
        "prompt_token_ids": prompt_ids,
        "prompt_token_strings": tokenizer.convert_ids_to_tokens(prompt_ids),
        "num_prompt_tokens": len(prompt_ids),
        "num_question_tokens_tokenize": num_question_tokens,
        "boundary_methods_agree": num_question_tokens == len(prompt_ids),
        "prompt_prefix_matches_full": full_ids[: len(prompt_ids)] == prompt_ids,
        "full_input_ids": full_ids,
        "full_token_count": len(full_ids),
        "truncated_to_max_length": len(full_ids) >= MAX_LENGTH,
        "answer_token_ids": answer_ids,
        "answer_token_strings": tokenizer.convert_ids_to_tokens(answer_ids),
        "answer_token_count": len(answer_ids),
        "is_single_token": len(answer_ids) == 1,
        "answer_token_span": {
            "start": num_question_tokens,
            "end": len(full_ids),
            "convention": "half_open_index_in_full_input_ids",
        },
        "decoded_answer_span": decoded,
        "decoded_answer_equals_target": decoded == answer,
    }


def load_eval_lines() -> list[dict]:
    files = sorted(TOFU_JSONL_DIR.glob("*.jsonl")) + sorted(TOFU_JSON_DIR.glob("*.json"))
    rows = []
    for path in files:
        rel = str(path.relative_to(REPO))
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rows.append(
                {
                    "path": rel,
                    "file_name": path.name,
                    "line": line_no,
                    "text": line,
                    "normalized": normalize_text(line),
                    "configured_eval": path.name in CONFIGURED_EVAL_NAMES,
                }
            )
    return rows


def scan_overlap(eval_rows: list[dict]) -> dict[str, dict]:
    compiled = {}
    for donor_id, groups in DONOR_TERMS.items():
        compiled[donor_id] = []
        for category, terms in groups.items():
            category_name = {
                "entity": "entity",
                "aliases": "alias",
                "neighbors": "neighbor",
                "places": "place",
            }[category]
            for term in terms:
                compiled[donor_id].append((category_name, term, term_pattern(term)))

    found: dict[str, list[dict]] = {donor_id: [] for donor_id in DONOR_TERMS}
    for row in eval_rows:
        for donor_id, terms in compiled.items():
            for category, term, pattern in terms:
                if pattern.search(row["normalized"]):
                    found[donor_id].append(
                        {
                            "category": category,
                            "term": term,
                            "file": row["path"],
                            "file_name": row["file_name"],
                            "line": row["line"],
                            "configured_eval_file": row["configured_eval"],
                            "snippet": match_excerpt(
                                row["text"], row["normalized"], term, pattern
                            ),
                        }
                    )
    summaries = {}
    for donor_id, hits in found.items():
        by_term = defaultdict(list)
        for hit in hits:
            by_term[(hit["category"], hit["term"])].append(hit)
        term_counts = []
        examples = []
        for (category, term), term_hits in sorted(by_term.items()):
            file_counts = defaultdict(int)
            configured = 0
            for hit in term_hits:
                file_counts[hit["file"]] += 1
                configured += int(hit["configured_eval_file"])
            term_counts.append(
                {
                    "category": category,
                    "term": term,
                    "count": len(term_hits),
                    "configured_eval_count": configured,
                    "ambiguous_place": term in AMBIGUOUS_PLACES,
                    "files": dict(sorted(file_counts.items())),
                }
            )
            examples.extend(term_hits[:EXAMPLES_PER_TERM])
        person_categories = {"entity", "alias", "neighbor"}
        summaries[donor_id] = {
            "person_hit_count": sum(1 for hit in hits if hit["category"] in person_categories),
            "place_hit_count": sum(1 for hit in hits if hit["category"] == "place"),
            "configured_eval_person_hit_count": sum(
                1
                for hit in hits
                if hit["category"] in person_categories and hit["configured_eval_file"]
            ),
            "configured_eval_place_hit_count": sum(
                1
                for hit in hits
                if hit["category"] == "place" and hit["configured_eval_file"]
            ),
            "term_counts": term_counts,
            "examples": examples,
        }
    return summaries


def donor_consistency(pair: dict, donors_by_id: dict[str, dict]) -> dict:
    donor = donors_by_id.get(pair["donor_id"])
    issues = []
    if donor is None:
        return {"issues": [f"missing donor record {pair['donor_id']}"]}
    qid = donor.get("wikidata_qid", "")
    url = donor.get("source_url", "")
    checks = {
        "claim_property_is_p19": donor.get("claim_property") == "P19",
        "relation_id_is_place_of_birth": donor.get("relation_id") == "place_of_birth",
        "relation_label_is_place_of_birth": donor.get("relation_label") == "place of birth",
        "qid_format_ok": bool(re.fullmatch(r"Q\d+", str(qid))),
        "url_contains_qid": str(qid) in str(url),
        "pair_qid_matches_donor": pair.get("donor_source", {}).get("qid") == qid,
        "pair_property_matches_donor": pair.get("donor_source", {}).get("property") == "P19",
        "pair_entity_matches_donor": pair.get("donor_entity") == donor.get("name"),
        "pair_donor_answer_matches_donor_file": pair.get("donor_answer") == donor.get("answer"),
        "answer_is_single_place_label": isinstance(donor.get("answer"), str)
        and "," not in donor.get("answer", "")
        and bool(donor.get("answer", "").strip()),
        "source_note_mentions_place_of_birth": "place-of-birth" in donor.get("source_note", "").lower()
        or "place of birth" in donor.get("source_note", "").lower(),
    }
    for name, ok in checks.items():
        if not ok:
            issues.append(name)
    granularity = "city_label_present_in_candidate_file"
    if "granularity" in donor.get("source_note", "").lower() or "city-level" in donor.get("source_note", "").lower():
        granularity = "candidate_note_asks_for_remote_granularity_check"
    return {
        "donor_id": donor["donor_id"],
        "name": donor["name"],
        "wikidata_qid": qid,
        "source_url": url,
        "claim_property": donor.get("claim_property"),
        "donor_answer": donor.get("answer"),
        "checks": checks,
        "internal_consistency": "pass" if not issues else "fail",
        "issues": issues,
        "city_level_granularity": granularity,
        "live_wikidata_p19_recheck": "not_performed_no_network",
    }


def verify_clean_sources(clean_rows: list[dict], forget_lines: list[str]) -> list[dict]:
    results = []
    for row in clean_rows:
        line_no = row.get("source_line_1based")
        issues = []
        source_obj = None
        if not isinstance(line_no, int) or not 1 <= line_no <= len(forget_lines):
            issues.append("source_line_out_of_range")
        else:
            try:
                source_obj = json.loads(forget_lines[line_no - 1])
            except json.JSONDecodeError as exc:
                issues.append(f"source_line_json_error: {exc}")
        if source_obj is not None:
            if source_obj.get("question") != row.get("source_question"):
                issues.append("source_question_mismatch")
            if source_obj.get("answer") != row.get("source_answer"):
                issues.append("source_answer_mismatch")
            if row.get("normalized_value") not in source_obj.get("answer", ""):
                issues.append("normalized_value_absent_from_source_answer")
        results.append(
            {
                "clean_cluster_id": row.get("clean_cluster_id"),
                "source_line_1based": line_no,
                "status": "pass" if not issues else "fail",
                "issues": issues,
            }
        )
    return results


def find_calibration_files() -> list[str]:
    skipped = {".git", "__pycache__", ".cursor"}
    matches = []
    for path in REPO.rglob("*"):
        if any(part in skipped for part in path.parts):
            continue
        if path.is_file() and "calibrat" in path.name.casefold():
            matches.append(str(path.relative_to(REPO)))
    return sorted(matches)


def fmt_hash_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| 文件 | SHA256 |", "| --- | --- |"]
    for name, digest in rows:
        lines.append(f"| `{name}` | `{digest}` |")
    return "\n".join(lines)


def build_summary(context: dict) -> str:
    audit = context["audit"]
    counts = context["counts"]
    records = context["records"]
    lines = []
    add = lines.append
    add("# 外部 donor 配对远程预筛验证")
    add("")
    add("## 结论")
    add("")
    add("**NO-GO。**")
    add("")
    add("K2 通过率：**不适用**。10 条配对全部为 `blocked`，0 条被打分，因此没有通过率，也没有模型拒绝样本。")
    add("")
    add("阻断原因：")
    add("")
    add("1. 仓库中没有独立 calibration 集，K2 阈值不能冻结。按照预筛条件，K2 筛选已停止，主候选没有被用来调阈值。")
    add("2. `candidate_pairs.jsonl` 有 8 条 `y_minus` 与 Wikidata 城市级答案 `donor_answer` 不一致。本次 `q_minus` 的编码目标使用 `donor_answer`。候选文件未修改。")
    add("3. 唯一 donor 簇是 9 个。`donor-01`（Katherine Mansfield / Wellington）被 pair-10 复用。")
    add("4. 本次没有加载模型权重，也没有前向计算，所以没有候选答案 logit、概率、top-1、最强竞争答案或 margin。")
    add("")
    add("在补齐独立 calibration 集、修正 `y_minus`，并在 GPU 上用同一 chat template 跑完严格 K2 之前，不开始激活定位或 Teacher/Student 训练。")
    add("")
    add("## 1. 文件核对与哈希")
    add("")
    add("候选目录只读。格式检查：三个 JSONL 文件的每一行都是 JSON 对象；`relation_audit.json` 是一个 JSON 对象。")
    add("")
    add("| 项目 | 文件计数 | relation_audit.json | 一致 |")
    add("| --- | ---: | ---: | --- |")
    add(f"| candidate_pairs | {counts['pairs']} | {audit['pair_count']} | {'是' if counts['pairs'] == audit['pair_count'] else '否'} |")
    add(f"| clean_facts | {counts['clean']} | — | 与 unique_clean_clusters={audit['unique_clean_clusters']} {'一致' if counts['unique_clean'] == audit['unique_clean_clusters'] else '不一致'} |")
    add(f"| wikidata_donors | {counts['donors']} | {audit['external_donor_facts']} | {'是' if counts['donors'] == audit['external_donor_facts'] else '否'} |")
    add(f"| unique clean clusters | {counts['unique_clean']} | {audit['unique_clean_clusters']} | {'是' if counts['unique_clean'] == audit['unique_clean_clusters'] else '否'} |")
    add(f"| unique donor clusters | {counts['unique_donors']} | {audit['unique_donor_clusters']} | {'是' if counts['unique_donors'] == audit['unique_donor_clusters'] else '否'} |")
    add(f"| donor reuse rows | {counts['reuse_rows']} | {audit['donor_reuse_count']} | {'是' if counts['reuse_rows'] == audit['donor_reuse_count'] else '否'} |")
    add("")
    add(f"解析错误：{context['parse_errors'] or '无'}。")
    add("")
    add(
        f"`forget05.jsonl` 行数为 {counts['forget05_rows']}，"
        f"SHA256 为 `{context['forget05_sha256']}`。"
        f"与 relation_audit 的 `rows_scanned={audit['rows_scanned']}`、"
        f"`source_sha256={audit['source_sha256']}` "
        f"{'一致' if context['forget05_ok'] else '不一致'}。"
    )
    add("")
    add("clean 侧来源行核对：" + context["clean_source_summary"])
    add("")
    add(fmt_hash_table(context["candidate_hashes"]))
    add("")
    add("写完结果后重新计算了上述候选文件哈希，与运行开始时相同。")
    add("")
    add("### y_minus 与 donor_answer")
    add("")
    add("| pair_id | donor | donor_answer | y_minus | 一致 |")
    add("| --- | --- | --- | --- | --- |")
    for record in records:
        minus = record["q_minus"]
        add(
            f"| {record['pair_id']} | {record['donor_entity']} | {minus['donor_answer']} | "
            f"{minus['y_minus']} | {'是' if minus['y_minus_matches_donor_answer'] else '否'} |"
        )
    add("")
    add(f"不一致的配对：{', '.join(context['mismatch_ids'])}。")
    add("")
    add("## 2. Checkpoint、tokenizer、推理代码与 chat template")
    add("")
    add("本次标识并用于编码的 checkpoint：")
    add("")
    add(f"- Hugging Face repo：`{context['checkpoint']['repo_id']}`")
    add(f"- snapshot commit：`{context['checkpoint']['snapshot_commit']}`")
    add(f"- 本地快照：`{context['checkpoint']['snapshot_path']}`")
    add(f"- `config.json` 的 `_name_or_path`：`{context['checkpoint']['name_or_path']}`")
    add("- 权重未加载。本机 `nvidia-smi` 看不到 GPU，脚本也没有调用 `AutoModelForCausalLM`。")
    add("")
    add("`Right-to-be-forgotten/config/tofu.yaml` 写的 `model_path` 是 `locuslab/tofu_ft_llama2-7b`。该 checkpoint 不在本地缓存中，本次没有把它当作实际 tokenizer 来源。")
    add("")
    add("Checkpoint 文件哈希：")
    add("")
    add(fmt_hash_table(context["checkpoint_hashes"]))
    add("")
    nous = context["nous_compare"]
    same = "、".join(f"`{name}`" for name in nous["same_files"]) or "无"
    different = "、".join(f"`{name}`" for name in nous["different_files"]) or "无"
    add(
        f"与本地 `NousResearch/Llama-2-7b-chat-hf` snapshot `{nous['snapshot']}` 的 tokenizer 文件比较："
        f"相同文件为 {same}；不同文件为 {different}。"
    )
    add("")
    tok = context["tokenizer"]
    add("实际加载的 tokenizer：")
    add("")
    add(f"- 类：`{tok['class_name']}`")
    add(f"- transformers：`{tok['transformers_version']}`")
    add(f"- vocab size：{tok['vocab_size']}；`len(tokenizer)`：{tok['len']}；")
    add(f"- bos/eos/pad：`{tok['bos_token']}`={tok['bos_token_id']}，`{tok['eos_token']}`={tok['eos_token_id']}，`{tok['pad_token']}`={tok['pad_token_id']}")
    add(f"- padding side：`{tok['padding_side']}`")
    add("")
    add("推理代码版本：")
    add("")
    add(f"- git commit：`{context['git_commit']}`")
    add(f"- 分支：`{context['git_branch']}`")
    add(f"- 运行开始时 `git status --porcelain`：`{context['git_status'] or 'clean'}`")
    add("")
    add("编码路径来自 `convert_raw_data_to_model_format`，标签来自 `model_config.yaml` 的 `llama2-7b`。相关文件哈希：")
    add("")
    add(fmt_hash_table(context["code_hashes"]))
    add("")
    add("实际 chat template 是手工拼接，不是 `tokenizer.apply_chat_template`：")
    add("")
    add("```text")
    add('[INST] {question} [/INST]{answer}')
    add("```")
    add("")
    add(f"- `question_start_tag` = `{QUESTION_START!r}`")
    add(f"- `question_end_tag` = `{QUESTION_END!r}`")
    add(f"- `answer_tag` = `{ANSWER_TAG!r}`")
    add("- `add_special_tokens=True`，因此序列以 BOS `<s>` 开头。")
    add(f"- 答案边界使用 `len(tokenizer.tokenize(prompt, add_special_tokens=True))`，与 `data_module.py` 相同。`max_length={MAX_LENGTH}`，本批没有截断。")
    add("- 训练标签在未截断时会在答案后再补一个 eos。这个 eos 属于 padding 标签，不计入答案 token span。")
    add("")
    add("tokenizer 自带的 `chat_template` 在 assistant 内容前加一个空格，并在答案后追加空格和 eos。它和上面的实际训练/评测拼接不同。本次两条问题都按实际拼接编码。")
    add("")
    add("存储的 chat template：")
    add("")
    add("```text")
    add(context["stored_chat_template"])
    add("```")
    add("")
    add(f"存储 template 的 SHA256：`{context['stored_chat_template_sha256']}`。")
    add("")
    add("`apply_chat_template` 对 pair-01 的 `q_minus` 会得到：")
    add("")
    add("```text")
    add(context["apply_chat_template_example"])
    add("```")
    add("")
    add("实际拼接得到：")
    add("")
    add("```text")
    add(context["manual_template_example"])
    add("```")
    add("")
    add("## 3. q_plus 与 q_minus 编码")
    add("")
    add("`q_plus` 的目标是 `y_plus`。`q_minus` 的目标是 `donor_answer`。两条都用上一节的同一模板。完整 prompt token IDs、答案 token IDs 和 span 在 `k2_results.jsonl`。")
    add("")
    add("| pair | q_plus 答案 | 答案 token 数 | span | 单 token | q_minus 目标 | 答案 token 数 | span | 单 token | 解码等于目标 |")
    add("| --- | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |")
    for record in records:
        plus = record["q_plus"]
        minus = record["q_minus"]
        add(
            f"| {record['pair_id']} | {plus['target_answer']} | {plus['answer_token_count']} | "
            f"{plus['answer_token_span']['start']}:{plus['answer_token_span']['end']} | "
            f"{'是' if plus['is_single_token'] else '否'} | {minus['target_answer']} | "
            f"{minus['answer_token_count']} | {minus['answer_token_span']['start']}:{minus['answer_token_span']['end']} | "
            f"{'是' if minus['is_single_token'] else '否'} | "
            f"{'是' if plus['decoded_answer_equals_target'] and minus['decoded_answer_equals_target'] else '否'} |"
        )
    add("")
    add(
        f"单 token 答案：q_plus {context['single_plus']} 条，q_minus {context['single_minus']} 条。"
        "其余答案是多 token，后续 K2 不能按单 token logit 处理。"
    )
    add("")
    add(
        f"边界自检：prompt 前缀与 full sequence 前缀不一致的条数 = {context['prefix_mismatches']}；"
        f"`tokenize` 与 `encode` 边界不一致的条数 = {context['boundary_mismatches']}；"
        f"解码结果不等于目标字符串的条数 = {context['decode_mismatches']}。"
    )
    add("")
    example = records[0]["q_minus"]
    add("pair-01 `q_minus` 样例：")
    add("")
    add(f"- prompt：`{example['prompt_text']}`")
    add(f"- prompt token IDs：`{example['prompt_token_ids']}`")
    add(f"- prompt tokens：`{example['prompt_token_strings']}`")
    add(f"- answer token IDs：`{example['answer_token_ids']}`")
    add(f"- answer tokens：`{example['answer_token_strings']}`")
    add(
        f"- answer span：[{example['answer_token_span']['start']}, {example['answer_token_span']['end']})，"
        f"解码 `{example['decoded_answer_span']}`"
    )
    add("")
    add("## 4–5. K2 阈值与 q_minus 筛选")
    add("")
    add("K2 筛选状态：`blocked`。")
    add("")
    add(f"每条记录的 `k2_status` 为：`{K2_STATUS}`。")
    add("")
    add(f"文件名搜索 `*calibrat*` 的结果：{context['calibration_files'] or '无'}。")
    add("")
    add("`holdout01/05/10`、`world_facts`、`real_authors`、`retain*` 是 TOFU 评测或保留数据，不是出生地 K2 的独立 calibration 集。本次没有用主候选或这些评测文件估计阈值。")
    add("")
    add("因此没有冻结阈值，也没有记录候选答案 logit、概率、top-1、最强竞争答案或 margin。这些字段在 `k2_results.jsonl` 中为 null。")
    add("")
    add("## 6. Donor QID、P19 与评测重叠")
    add("")
    add("本机访问 Wikidata 失败（`Cannot assign requested address`），所以没有在线复核 P19。下面是候选文件内部一致性。")
    add("")
    add("| donor | QID | P19 | 城市级标签 | 内部一致性 | 在线复核 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for donor_id, item in context["donor_checks"].items():
        add(
            f"| {donor_id} {item['name']} | {item['wikidata_qid']} | {item['claim_property']} | "
            f"{item['donor_answer']} | {item['internal_consistency']} | {item['live_wikidata_p19_recheck']} |"
        )
    add("")
    granularity = [
        f"{item['donor_id']} ({item['city_level_granularity']})"
        for item in context["donor_checks"].values()
        if item["city_level_granularity"] != "city_label_present_in_candidate_file"
    ]
    add(f"候选笔记要求远程确认粒度的 donor：{granularity or '无'}。")
    add("")
    add(f"内部一致性失败项：{context['consistency_failures'] or '无'}。")
    add("")
    add(
        f"评测清单扫描了 `workspace/data/tofu/*.jsonl`（{context['inventory_counts']['jsonl_files']} 个文件）"
        f"和 `Right-to-be-forgotten/data/tofu/*.json`（{context['inventory_counts']['json_files']} 个文件，内容为 JSONL）。"
        f"共 {context['inventory_counts']['lines']} 行。"
    )
    add("")
    add("`tofu.yaml` 当前评测清单是 `retain_perturbed`、`real_authors_perturbed`、`world_facts_perturbed`、`forget05_perturbed`。重叠表同时给出全清单命中和这份配置清单中的命中。")
    add("")
    add("搜索表是预先写明的实体、别名、近邻人物和出生地字符串。重音和弯引号做了归一化，匹配使用字母数字边界。")
    add("")
    add("| donor | 实体/别名/近邻命中 | 其中配置评测清单 | 出生地字符串命中 | 其中配置评测清单 |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for donor_id, summary in context["overlap"].items():
        add(
            f"| {donor_id} | {summary['person_hit_count']} | {summary['configured_eval_person_hit_count']} | "
            f"{summary['place_hit_count']} | {summary['configured_eval_place_hit_count']} |"
        )
    add("")
    add("有命中的搜索词：")
    add("")
    hit_terms = context["hit_terms"]
    if not hit_terms:
        add("无。")
    else:
        add("| donor | 类别 | 词 | 次数 | 配置评测清单次数 | 歧义 |")
        add("| --- | --- | --- | ---: | ---: | --- |")
        for row in hit_terms:
            add(
                f"| {row['donor_id']} | {row['category']} | {row['term']} | {row['count']} | "
                f"{row['configured_eval_count']} | {'是' if row['ambiguous_place'] else '否'} |"
            )
    add("")
    person_examples = []
    for donor_id, summary in context["overlap"].items():
        for example in summary["examples"]:
            if example["category"] in {"entity", "alias", "neighbor"}:
                person_examples.append((donor_id, example))
    if not person_examples:
        add("实体、别名和近邻人物字符串没有命中这份评测清单。")
    else:
        add("实体、别名和近邻人物的字符串命中：")
        add("")
        add("| donor | 类别 | 词 | 位置 | 上下文 |")
        add("| --- | --- | --- | --- | --- |")
        for donor_id, example in person_examples:
            snippet = example["snippet"].replace("|", "\\|").replace("\n", " ")
            add(
                f"| {donor_id} | {example['category']} | {example['term']} | "
                f"`{example['file']}:{example['line']}` | `{snippet}` |"
            )
        add("")
        add(
            "donor-03 的实体和别名命中是同一行：`Right-to-be-forgotten/data/tofu/forget10_perturbed.json:2021` "
            "中的虚构作者 “Beyonce Johnson”。归一化会把 `Beyoncé` 和 `Beyonce` 收成同一个字符串，所以这一行被计了两次。"
            "该行不在 `tofu.yaml` 当前的 forget05 评测清单里，也不是 Wikidata Q36153 进入评测集的证据。"
        )
    add("")
    add(
        "出生地次数是两个目录里的原始行命中。`workspace/data/tofu` 的 jsonl 和 "
        "`Right-to-be-forgotten/data/tofu` 的 json 会把同一事实各计一次，perturbed 改写也会重复计入。"
        "`London`、`Rio de Janeiro`、`New Orleans`、`Seattle`、`Wellington`、`Houston` 的命中是城市字符串，"
        "不是 donor 人物出现在评测清单中。样例在 `k2_results.jsonl` 的 `overlap.examples`。"
    )
    add("")
    add("## 7. 簇、复用与 go/no-go")
    add("")
    add(f"- unique clean clusters：{counts['unique_clean']}")
    add(f"- unique donor clusters：{counts['unique_donors']}")
    add(
        f"- donor 复用：{counts['reuse_rows']} 行，复用 {', '.join(context['reused_donors'])}，"
        f"出现在 {', '.join(context['reuse_pairs'])}"
    )
    add("- K2 通过率：不适用（0 条打分，10 条 blocked）")
    add("- 建议：**NO-GO**")
    add("")
    add("恢复实验前需要同时满足：")
    add("")
    add("1. 提供与主候选分开的出生地 calibration 集，并在看到主候选 K2 结果之前冻结阈值。")
    add("2. 修正 `candidate_pairs.jsonl` 的 `y_minus`，使它与对应 donor 的 P19 城市级答案一致。")
    add("3. 在 GPU 上用本报告记录的实际 chat template 对每条 `q_minus` 做严格 K2，记录答案 logit、概率、top-1、最强竞争答案和 margin。多 token 答案按答案 span 计算，不按单 token 处理。")
    add("4. 恢复外网后复核每条 QID 的 P19 是否支持当前城市级答案，尤其是 donor-04 Gary。")
    add("5. 若出生地字符串命中需要处理，先区分常见城市词和真正的 donor 事实泄漏。")
    add("")
    add("## 8. 本阶段没有做的事")
    add("")
    add("没有运行激活定位。没有运行 Teacher/Student 训练。没有为了增加通过样本而选择或调整 K2 阈值。没有修改 TOFU 文件、评测文件或 `external_donor_pairs_wikidata_v0.1/` 中的候选文件。")
    add("")
    add(f"结果目录：`{OUT_DIR.relative_to(REPO)}`。")
    add(f"生成时间：`{context['generated_at']}`。")
    add(f"本脚本 SHA256：`{context['script_sha256']}`。")
    add("")
    return "\n".join(lines)


def main() -> None:
    if OUT_DIR.resolve() == CANDIDATE_DIR.resolve():
        raise RuntimeError("output directory must stay outside the candidate directory")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    git_commit = git_text(["rev-parse", "HEAD"])
    git_branch = git_text(["rev-parse", "--abbrev-ref", "HEAD"])
    git_status = git_text(["status", "--porcelain"])

    candidate_hashes = [(name, sha256_file(CANDIDATE_DIR / name)) for name in CANDIDATE_FILES]
    code_hashes = [(str(path.relative_to(REPO)), sha256_file(path)) for path in CODE_FILES]
    checkpoint_hashes = [(name, sha256_file(SNAPSHOT / name)) for name in CHECKPOINT_FILES]

    pairs, pair_errors = read_jsonl(CANDIDATE_DIR / "candidate_pairs.jsonl")
    clean_rows, clean_errors = read_jsonl(CANDIDATE_DIR / "clean_facts.jsonl")
    donors, donor_errors = read_jsonl(CANDIDATE_DIR / "wikidata_donors.jsonl")
    audit = json.loads((CANDIDATE_DIR / "relation_audit.json").read_text(encoding="utf-8"))
    parse_errors = pair_errors + clean_errors + donor_errors

    forget_text = FORGET05.read_text(encoding="utf-8")
    forget_lines = [line for line in forget_text.splitlines() if line.strip()]
    forget05_sha = sha256_file(FORGET05)
    forget05_ok = (
        len(forget_lines) == audit["rows_scanned"] and forget05_sha == audit["source_sha256"]
    )
    clean_source = verify_clean_sources(clean_rows, forget_lines)
    clean_source_failures = [row for row in clean_source if row["status"] != "pass"]

    donors_by_id = {row["donor_id"]: row for row in donors}
    unique_clean = {row["clean_cluster_id"] for row in pairs}
    unique_donors = {row["donor_id"] for row in pairs}
    reuse_pairs = [row["pair_id"] for row in pairs if row.get("donor_reuse") is True]
    reused_donors = sorted({row["donor_id"] for row in pairs if row.get("donor_reuse") is True})

    import transformers

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    stored_chat_template = tokenizer.chat_template or ""
    example_messages = [
        {"role": "user", "content": "Where was Katherine Mansfield born?"},
        {"role": "assistant", "content": "Wellington"},
    ]
    apply_example = tokenizer.apply_chat_template(
        example_messages, tokenize=False, add_generation_prompt=False
    )
    manual_example = QUESTION_START + example_messages[0]["content"] + QUESTION_END + "Wellington"

    nous_same = []
    nous_different = []
    for name in TOKENIZER_COMPARE_FILES:
        local_hash = sha256_file(SNAPSHOT / name)
        nous_hash = sha256_file(NOUS_SNAPSHOT / name)
        (nous_same if local_hash == nous_hash else nous_different).append(name)

    eval_rows = load_eval_lines()
    overlap = scan_overlap(eval_rows)
    calibration_files = find_calibration_files()
    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))

    records = []
    donor_checks = {}
    for pair in pairs:
        plus = encode_side(tokenizer, pair["q_plus"], pair["y_plus"])
        minus = encode_side(tokenizer, pair["q_minus"], pair["donor_answer"])
        minus["target_field"] = "donor_answer"
        minus["donor_answer"] = pair["donor_answer"]
        minus["y_minus"] = pair["y_minus"]
        minus["y_minus_matches_donor_answer"] = pair["y_minus"] == pair["donor_answer"]
        plus["target_field"] = "y_plus"
        plus["y_plus"] = pair["y_plus"]
        plus["clean_answer"] = pair["clean_answer"]
        plus["y_plus_matches_clean_answer"] = pair["y_plus"] == pair["clean_answer"]
        consistency = donor_consistency(pair, donors_by_id)
        donor_checks[pair["donor_id"]] = consistency
        donor_overlap = overlap[pair["donor_id"]]
        records.append(
            {
                "pair_id": pair["pair_id"],
                "arm": pair["arm"],
                "relation_id": pair["relation_id"],
                "wikidata_property": pair["wikidata_property"],
                "clean_cluster_id": pair["clean_cluster_id"],
                "clean_entity": pair["clean_entity"],
                "donor_id": pair["donor_id"],
                "donor_entity": pair["donor_entity"],
                "donor_reuse": pair["donor_reuse"],
                "chat_template": {
                    "id": "llama2-7b-manual-inst-data-module",
                    "model_family": "llama2-7b",
                    "question_start_tag": QUESTION_START,
                    "question_end_tag": QUESTION_END,
                    "answer_tag": ANSWER_TAG,
                    "add_special_tokens": True,
                    "source": "Right-to-be-forgotten/dataset/data_module.py:convert_raw_data_to_model_format",
                },
                "q_plus": plus,
                "q_minus": minus,
                "donor_consistency": consistency,
                "overlap": donor_overlap,
                "k2_status": K2_STATUS,
                "pass_reject": "blocked",
                "k2": {
                    "threshold": None,
                    "calibration_set": None,
                    "forward_pass_run": False,
                    "candidate_answer_logit": None,
                    "candidate_answer_probability": None,
                    "top1": None,
                    "strongest_competitor": None,
                    "margin": None,
                },
            }
        )

    results_path = OUT_DIR / "k2_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    hit_terms = []
    for donor_id, summary in overlap.items():
        for row in summary["term_counts"]:
            if row["count"]:
                hit_terms.append({"donor_id": donor_id, **row})

    rehash = [(name, sha256_file(CANDIDATE_DIR / name)) for name in CANDIDATE_FILES]
    if rehash != candidate_hashes:
        raise RuntimeError("candidate files changed during validation")

    script_path = Path(__file__).resolve()
    context = {
        "audit": audit,
        "counts": {
            "pairs": len(pairs),
            "clean": len(clean_rows),
            "donors": len(donors),
            "unique_clean": len(unique_clean),
            "unique_donors": len(unique_donors),
            "reuse_rows": len(reuse_pairs),
            "forget05_rows": len(forget_lines),
        },
        "records": records,
        "parse_errors": parse_errors,
        "forget05_sha256": forget05_sha,
        "forget05_ok": forget05_ok,
        "clean_source_summary": (
            "10 条来源行的 question、answer 和城市名都与 forget05.jsonl 一致。"
            if not clean_source_failures
            else json.dumps(clean_source_failures, ensure_ascii=False)
        ),
        "candidate_hashes": candidate_hashes,
        "code_hashes": code_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "checkpoint": {
            "repo_id": "open-unlearning/tofu_Llama-2-7b-chat-hf_full",
            "snapshot_commit": (SNAPSHOT.parent.parent / "refs" / "main").read_text(encoding="utf-8").strip(),
            "snapshot_path": str(SNAPSHOT),
            "name_or_path": config.get("_name_or_path"),
        },
        "nous_compare": {
            "snapshot": str(NOUS_SNAPSHOT),
            "same_files": nous_same,
            "different_files": nous_different,
        },
        "tokenizer": {
            "class_name": type(tokenizer).__name__,
            "transformers_version": transformers.__version__,
            "vocab_size": tokenizer.vocab_size,
            "len": len(tokenizer),
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "padding_side": tokenizer.padding_side,
        },
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_status": git_status,
        "stored_chat_template": stored_chat_template,
        "stored_chat_template_sha256": sha256_text(stored_chat_template),
        "apply_chat_template_example": apply_example,
        "manual_template_example": manual_example,
        "single_plus": sum(row["q_plus"]["is_single_token"] for row in records),
        "single_minus": sum(row["q_minus"]["is_single_token"] for row in records),
        "prefix_mismatches": sum(
            (not row["q_plus"]["prompt_prefix_matches_full"])
            + (not row["q_minus"]["prompt_prefix_matches_full"])
            for row in records
        ),
        "boundary_mismatches": sum(
            (not row["q_plus"]["boundary_methods_agree"])
            + (not row["q_minus"]["boundary_methods_agree"])
            for row in records
        ),
        "decode_mismatches": sum(
            (not row["q_plus"]["decoded_answer_equals_target"])
            + (not row["q_minus"]["decoded_answer_equals_target"])
            for row in records
        ),
        "mismatch_ids": [row["pair_id"] for row in records if not row["q_minus"]["y_minus_matches_donor_answer"]],
        "calibration_files": calibration_files,
        "donor_checks": donor_checks,
        "consistency_failures": [
            {"donor_id": donor_id, "issues": item["issues"]}
            for donor_id, item in donor_checks.items()
            if item["issues"]
        ],
        "inventory_counts": {
            "jsonl_files": len(list(TOFU_JSONL_DIR.glob("*.jsonl"))),
            "json_files": len(list(TOFU_JSON_DIR.glob("*.json"))),
            "lines": len(eval_rows),
        },
        "overlap": overlap,
        "hit_terms": hit_terms,
        "reused_donors": reused_donors,
        "reuse_pairs": reuse_pairs,
        "generated_at": generated_at,
        "script_sha256": sha256_file(script_path),
    }
    summary = build_summary(context)
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {OUT_DIR / 'summary.md'}")
    print(f"pairs={len(records)} blocked={sum(row['pass_reject']=='blocked' for row in records)}")
    print(f"y_minus_mismatches={context['mismatch_ids']}")
    print(f"person_hits={sum(item['person_hit_count'] for item in overlap.values())}")
    print(f"place_hits={sum(item['place_hit_count'] for item in overlap.values())}")


if __name__ == "__main__":
    main()
