"""Literal / alias matching. Headline metric; no LLM-judge."""


def _norm(text):
    return " ".join(str(text).strip().lower().split())


def answer_or_alias_contained(prediction, answer, aliases=None):
    pred = _norm(prediction)
    if not pred:
        return False
    candidates = [answer]
    if aliases:
        candidates.extend(aliases)
    for cand in candidates:
        gold = _norm(cand)
        if gold and gold in pred:
            return True
    return False


def hit_rate(predictions, answers, aliases_list=None):
    n = len(predictions)
    if n == 0:
        return 0.0
    hits = 0
    for i, pred in enumerate(predictions):
        aliases = None if aliases_list is None else aliases_list[i]
        if answer_or_alias_contained(pred, answers[i], aliases):
            hits += 1
    return hits / n
