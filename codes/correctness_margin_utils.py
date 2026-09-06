import torch

from steering_utils import normalize_yes_no


def single_token_ids(tokenizer, words):
    token_ids = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.append(ids[0])
    return sorted(set(token_ids))


def answer_token_ids(tokenizer, answer):
    normalized = normalize_yes_no(answer)
    if normalized == "yes":
        words = ["yes", "Yes", " yes", " Yes", "YES", " YES"]
    elif normalized == "no":
        words = ["no", "No", " no", " No", "NO", " NO"]
    else:
        words = [str(answer).strip(), " " + str(answer).strip()]

    token_ids = single_token_ids(tokenizer, words)
    if not token_ids:
        raise ValueError(f"Answer must have at least one single-token form: {answer!r}")
    return token_ids


def correct_answer_margin(outputs, answer_ids):
    logits = outputs.logits[0, -1].float()
    answer_index = torch.tensor(answer_ids, device=logits.device, dtype=torch.long)
    answer_logit = logits.index_select(0, answer_index).max()

    competitor_logits = logits.clone()
    competitor_logits.index_fill_(0, answer_index, float("-inf"))
    competitor_logit, competitor_id = competitor_logits.max(dim=0)
    return answer_logit - competitor_logit, answer_logit, competitor_logit, competitor_id


def transform_score(score, mode):
    if mode == "raw":
        return score
    if mode == "tanh":
        return torch.tanh(score)
    raise ValueError(f"Unknown score transform: {mode}")
