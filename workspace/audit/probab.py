"""Probabilistic decoding recovery. Inject generate_fn; do not load 7B here."""

import argparse

from .match import hit_rate


def sample_predictions(generate_fn, prompts, temperature, k, seed=0):
    """generate_fn(prompt, temperature, seed_i) -> str."""
    rows = []
    for prompt in prompts:
        samples = [
            generate_fn(prompt, temperature=temperature, seed=seed + j)
            for j in range(k)
        ]
        rows.append(samples)
    return rows


def any_hit_rate(sample_rows, answers, aliases_list=None):
    flattened_preds = []
    flattened_answers = []
    flattened_aliases = []
    for i, samples in enumerate(sample_rows):
        alias = None if aliases_list is None else aliases_list[i]
        for pred in samples:
            flattened_preds.append(pred)
            flattened_answers.append(answers[i])
            flattened_aliases.append(alias)
    return hit_rate(flattened_preds, flattened_answers, flattened_aliases)


def recovery_delta(attack_rate, greedy_rate):
    return attack_rate - greedy_rate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probab channel (no 7B this round).")
    parser.add_argument("--model", default="", help="Checkpoint path for next round.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)
    if not args.model:
        print("dry-run: no model loaded; pass --model next round")
        return 0
    raise RuntimeError("Real generation is deferred. Do not load 7B this round.")


if __name__ == "__main__":
    raise SystemExit(main())
