"""FocusOnKey prompt injection. Greedy generate_fn is injected; no 7B load."""

import argparse

DEFAULT_TEMPLATE = (
    "Question: {question}\n"
    "KeyToken: {keytoken}\n"
    "Answer:"
)
BASE_TEMPLATE = "Question: {question}\nAnswer:"


def inject_keytoken(question, keytoken, template=DEFAULT_TEMPLATE):
    return template.format(question=question, keytoken=keytoken)


def base_prompt(question, template=BASE_TEMPLATE):
    return template.format(question=question)


def greedy_decode(generate_fn, prompts):
    return [generate_fn(p, temperature=0.0, seed=0) for p in prompts]


def main(argv=None):
    parser = argparse.ArgumentParser(description="FocusOnKey channel (no 7B this round).")
    parser.add_argument("--model", default="")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)
    if not args.model:
        print("dry-run: no model loaded; pass --model next round")
        return 0
    raise RuntimeError("Real generation is deferred. Do not load 7B this round.")


if __name__ == "__main__":
    raise SystemExit(main())
