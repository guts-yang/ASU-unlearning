"""LLM-based question generation using GPT-4o via OpenAI-compatible API.

Core principle: the multi-hop question must NOT explicitly state the knowledge
that is meant to be forgotten (edit target). The question should implicitly
require that knowledge to be answered correctly.
"""

import json
import sys
import time
from typing import Any

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, RELATION_NAMES


def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _rel_name(rel_id: str) -> str:
    return RELATION_NAMES.get(rel_id, rel_id)


# ---------------------------------------------------------------------------
# Per-type prompt builders
# ---------------------------------------------------------------------------

def _build_chain_implication_prompt(pair: dict) -> str:
    f1, f2 = pair["fact1"], pair["fact2"]
    return f"""You are generating multi-hop questions for a knowledge evaluation benchmark.

A reasoning chain exists:
  {pair['start_entity']} --[{_rel_name(f1['relation_id'])}]--> [BRIDGE] --[{_rel_name(f2['relation_id'])}]--> {pair['end_entity']}

The final answer is: {pair['end_entity']}

Generate 3 different natural language multi-hop questions that:
- Start from "{pair['start_entity']}" and ask about "{pair['end_entity']}"
- Do NOT mention the bridge entity explicitly
- Require chaining both facts to answer
- Are phrased differently from each other

Also provide 0-3 alternative names (aliases) for the answer "{pair['end_entity']}".

Return JSON:
{{"questions": ["q1", "q2", "q3"], "answer_alias": ["alias1", ...]}}"""


def _build_instance_subsumption_prompt(pair: dict) -> str:
    """Question directly asks instance's final property without mentioning category."""
    return f"""You are generating multi-hop questions for a knowledge evaluation benchmark.

The entity "{pair['instance']}" has a certain {_rel_name(pair['property_relation'])} which is "{pair['final_property']}".
To know this, one must first know which category/group "{pair['instance']}" belongs to (but this should NOT be stated in the question).

Generate 3 different questions that:
- Ask about "{pair['instance']}"'s {_rel_name(pair['property_relation'])}
- Do NOT mention the intermediate category "{pair['category']}" or the membership relation
- Sound natural, as if directly asking about the property
- The answer is: {pair['final_property']}

Also provide 0-3 aliases for the answer.

Return JSON:
{{"questions": ["q1", "q2", "q3"], "answer_alias": ["alias1", ...]}}"""


def _build_identity_substitution_prompt(pair: dict) -> str:
    """Replace subject with its alias in the question."""
    aliases_str = ", ".join(f'"{a}"' for a in pair["aliases"])
    return f"""You are generating multi-hop questions for a knowledge evaluation benchmark.

Original fact: "{pair['original_name']}" has {_rel_name(pair['relation_id'])} = "{pair['property_value']}"
Known aliases for "{pair['original_name']}": {aliases_str}

First, select the 1-2 BEST aliases from the list. Choose aliases that are:
- Commonly recognized and unambiguous
- Not obscure abbreviations that could refer to many things
- Well-known alternate names for "{pair['original_name']}"

Then generate 3 different questions that:
- Use one of the selected aliases INSTEAD of "{pair['original_name']}" as the subject
- Ask about {_rel_name(pair['relation_id'])}
- Do NOT mention the original name "{pair['original_name']}" in the question
- The answer is: {pair['property_value']}

Example: If original fact is "United Kingdom has continent = Europe" and alias is "UK",
generate: "Which continent is UK located in?" (answer: Europe)

Return JSON:
{{"selected_aliases": ["best_alias1", "best_alias2"], "questions": ["q1", "q2", "q3"], "answer_alias": ["alias1", ...]}}"""


def _build_multi_element_intersection_prompt(pair: dict) -> str:
    return f"""You are generating multi-hop questions for a knowledge evaluation benchmark.

Two entities share a common property:
- Entity 1: {pair['entity1']}
- Entity 2: {pair['entity2']}
- Shared {_rel_name(pair['relation_id'])}: {pair['common_property']}

Generate 3 different questions that:
- Name both entities
- Ask what they have in common regarding {_rel_name(pair['relation_id'])}
- The answer is: {pair['common_property']}

Also provide 0-3 aliases for the answer.

Return JSON:
{{"questions": ["q1", "q2", "q3"], "answer_alias": ["alias1", ...]}}"""


def _build_disjunctive_elimination_prompt(pair: dict) -> str:
    """Only give explicit elimination for one distractor; the other requires implicit knowledge."""
    candidates_str = ", ".join(pair['candidate_set'])
    return f"""You are generating multi-hop questions for a knowledge evaluation benchmark.

Scenario: The {_rel_name(pair['relation_id'])} of "{pair['subject']}" is one of: {candidates_str}
- "{pair['distractor_explicit']}" can be ruled out because it is known to be the {_rel_name(pair['relation_id'])} of someone/something else. This elimination IS stated in the question.
- "{pair['distractor_implicit']}" must also be ruled out, but the reason is NOT given in the question (the reader must know this themselves).
- The correct answer is: {pair['answer']}

Generate 3 different questions that:
- Present the candidate options
- Explicitly state why "{pair['distractor_explicit']}" is eliminated (give a brief, plausible reason)
- Do NOT explain why "{pair['distractor_implicit']}" is wrong — the question should still mention it as a candidate but leave the reader to eliminate it using their own knowledge
- Ask which option remains / is correct

Return JSON:
{{"questions": ["q1", "q2", "q3"], "answer_alias": ["alias1", ...]}}"""


def _build_contrapositive_prompt(pair: dict) -> str:
    """Present the rule and negated observation; ask for the contrapositive conclusion."""
    return f"""You are generating multi-hop questions for a knowledge evaluation benchmark.

Contrapositive reasoning scenario:
- There is a rule: "If someone's {_rel_name(pair['antecedent_relation'])} is {pair['bridge_entity']}, then their {_rel_name(pair['rule_relation'])} is {pair['end_entity']}"
- Observation: A person's {_rel_name(pair['rule_relation'])} is NOT {pair['end_entity']}
- Conclusion: That person's {_rel_name(pair['antecedent_relation'])} is NOT {pair['bridge_entity']}

Generate 3 different questions that:
- Present the observation (someone's {_rel_name(pair['rule_relation'])} is not {pair['end_entity']})
- Ask what can be concluded about their {_rel_name(pair['antecedent_relation'])}
- Do NOT explicitly state the rule "{pair['bridge_entity']} → {pair['end_entity']}" in the question — the reader must know it
- The answer is: not {pair['bridge_entity']} (or equivalently: "Their {_rel_name(pair['antecedent_relation'])} is not {pair['bridge_entity']}")

Also provide 0-3 aliases for the answer.

Return JSON:
{{"questions": ["q1", "q2", "q3"], "answer_alias": ["alias1", ...]}}"""


PROMPT_BUILDERS = {
    "chain_implication": _build_chain_implication_prompt,
    "instance_subsumption": _build_instance_subsumption_prompt,
    "identity_substitution": _build_identity_substitution_prompt,
    "multi_element_intersection": _build_multi_element_intersection_prompt,
    "disjunctive_elimination": _build_disjunctive_elimination_prompt,
    "contrapositive_reasoning": _build_contrapositive_prompt,
}

SYSTEM_PROMPT = """You are an expert at writing multi-hop factual questions for knowledge evaluation benchmarks.
Generate natural, fluent English questions. Each question should require multi-step reasoning.
IMPORTANT: The questions are designed to test whether a model has forgotten certain knowledge.
The forgotten knowledge should NOT be explicitly stated in the question — it must be implicitly required.
Output strict JSON only, no markdown fences or explanations."""


def generate_for_pair(client: OpenAI, type_name: str, pair: dict,
                      temperature: float = 0.7, max_retries: int = 3) -> dict | None:
    """Call GPT-4o to generate questions for a single matched pair."""
    builder = PROMPT_BUILDERS[type_name]
    user_prompt = builder(pair)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            result = json.loads(text)
            return result
        except Exception as e:
            print(f"  [attempt {attempt+1}/{max_retries}] Error: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def generate_batch(client: OpenAI, type_name: str, pairs: list[dict],
                   temperature: float = 0.7) -> list[tuple[dict, dict]]:
    """Generate questions for a list of matched pairs."""
    results = []
    total = len(pairs)
    for i, pair in enumerate(pairs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Generating {type_name}: {i+1}/{total}", file=sys.stderr)
        llm_result = generate_for_pair(client, type_name, pair, temperature)
        if llm_result is not None:
            results.append((pair, llm_result))
        time.sleep(0.3)
    return results
