import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "mid_file" / "triton_cache"))

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import set_seed

from steering_utils import (
    DEFAULT_TEMP_MODEL_NAME,
    DEFAULT_LOCAL_VISION_TOWER,
    DEFAULT_MODEL_PATH,
    build_prompt,
    load_modal_tensors,
    load_vtlm_model,
    parse_layers,
    tokenize_prompt,
)


SUBMIT_ROOT = CURRENT_DIR.parent
DEFAULT_MID_DIR = str(SUBMIT_ROOT / "mid_file")


def iter_jsonl(path):
    with open(os.path.expanduser(path), "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def single_token_ids(tokenizer, words):
    token_ids = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.append(ids[0])
    return sorted(set(token_ids))


def yes_no_token_ids(tokenizer):
    yes_ids = single_token_ids(tokenizer, ["yes", "Yes", " yes", " Yes", "YES", " YES"])
    no_ids = single_token_ids(tokenizer, ["no", "No", " no", " No", "NO", " NO"])
    if not yes_ids or not no_ids:
        raise RuntimeError(f"Could not find single-token yes/no ids: yes={yes_ids}, no={no_ids}")
    return yes_ids, no_ids


def max_token_logit(logits, token_ids):
    ids = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    return logits.index_select(0, ids).max()


def yes_no_margin(outputs, yes_ids, no_ids):
    logits = outputs.logits[0, -1].float()
    yes_logit = max_token_logit(logits, yes_ids)
    no_logit = max_token_logit(logits, no_ids)
    return yes_logit - no_logit


def apply_variant(image_tensor, touch_tensor, variant):
    image = image_tensor
    touch = touch_tensor
    if variant.get("image_mode") == "blank":
        image = torch.zeros_like(image_tensor)
    if variant.get("touch_mode") == "blank":
        touch = torch.zeros_like(touch_tensor)
    return image, touch


def forward_variant(row, bundle, args, image_tensor, touch_tensor, variant):
    image, touch = apply_variant(image_tensor, touch_tensor, variant)
    prompt = build_prompt(row["question"], args.conv_mode, answer=None)
    input_ids = tokenize_prompt(prompt, bundle.tokenizer)
    with torch.inference_mode():
        return bundle.model(
            input_ids=input_ids,
            images=image,
            touchs=touch,
            output_hidden_states=True,
            use_cache=False,
        )


def transform_score(score, mode):
    if mode == "raw":
        return score
    if mode == "tanh":
        return torch.tanh(score)
    raise ValueError(f"Unknown score transform: {mode}")


def hidden_delta(full_outputs, ablated_outputs, layer, position):
    full_hidden = full_outputs.hidden_states[layer + 1][0]
    ablated_hidden = ablated_outputs.hidden_states[layer + 1][0]
    if position == "last":
        return full_hidden[-1].float() - ablated_hidden[-1].float()
    return full_hidden.float().mean(dim=0) - ablated_hidden.float().mean(dim=0)


def attribution_vector(delta, score, normalize_delta):
    if normalize_delta:
        delta = F.normalize(delta, dim=-1)
    return transform_score(score, "tanh") * delta


def main(args):
    set_seed(args.seed)
    layers = parse_layers(args.layers)
    bundle = load_vtlm_model(args, run_name=DEFAULT_TEMP_MODEL_NAME)
    yes_ids, no_ids = yes_no_token_ids(bundle.tokenizer)
    print(f"yes token ids: {yes_ids}")
    print(f"no token ids: {no_ids}")

    touch_vectors = {layer: [] for layer in layers}
    image_vectors = {layer: [] for layer in layers}
    touch_scores = []
    image_scores = []
    margins = []
    metadata = []
    count = 0
    skipped = 0

    rows = iter_jsonl(args.manifest_file)
    for row in tqdm(rows, total=args.max_samples or None, desc="extract modality attribution"):
        if args.max_samples and count >= args.max_samples:
            break

        try:
            image_tensor, touch_tensor = load_modal_tensors(row, bundle, args.image_folder)
            variants = row["variants"]
            full_outputs = forward_variant(row, bundle, args, image_tensor, touch_tensor, variants["full"])
            no_touch_outputs = forward_variant(row, bundle, args, image_tensor, touch_tensor, variants["no_touch"])
            no_image_outputs = forward_variant(row, bundle, args, image_tensor, touch_tensor, variants["no_image"])

            margin_full = yes_no_margin(full_outputs, yes_ids, no_ids)
            margin_no_touch = yes_no_margin(no_touch_outputs, yes_ids, no_ids)
            margin_no_image = yes_no_margin(no_image_outputs, yes_ids, no_ids)
            touch_score = transform_score(margin_full - margin_no_touch, args.score_transform)
            image_score = transform_score(margin_full - margin_no_image, args.score_transform)

            for layer in layers:
                delta_touch = hidden_delta(full_outputs, no_touch_outputs, layer, args.hidden_position)
                delta_image = hidden_delta(full_outputs, no_image_outputs, layer, args.hidden_position)
                if args.normalize_delta:
                    delta_touch = F.normalize(delta_touch, dim=-1)
                    delta_image = F.normalize(delta_image, dim=-1)
                touch_vectors[layer].append((touch_score * delta_touch).detach().to(dtype=torch.float16).cpu())
                image_vectors[layer].append((image_score * delta_image).detach().to(dtype=torch.float16).cpu())

            touch_scores.append(float(touch_score.detach().cpu()))
            image_scores.append(float(image_score.detach().cpu()))
            margins.append(
                {
                    "full": float(margin_full.detach().cpu()),
                    "no_touch": float(margin_no_touch.detach().cpu()),
                    "no_image": float(margin_no_image.detach().cpu()),
                }
            )
            metadata.append(
                {
                    "sample_id": row.get("sample_id"),
                    "split": row.get("split"),
                    "source_file": row.get("source_file"),
                    "source_index": row.get("source_index"),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "video_path": row.get("video_path", ""),
                    "tactile_path": row.get("tactile_path", ""),
                }
            )
            count += 1
        except Exception as exc:
            skipped += 1
            if args.verbose:
                print(f"[skip] {row.get('sample_id')} {exc}")

    if count == 0:
        raise RuntimeError("No modality attribution examples were processed.")

    payload = {
        "touch_vectors": {layer: torch.stack(touch_vectors[layer]) for layer in layers},
        "image_vectors": {layer: torch.stack(image_vectors[layer]) for layer in layers},
        "touch_scores": torch.tensor(touch_scores, dtype=torch.float32),
        "image_scores": torch.tensor(image_scores, dtype=torch.float32),
        "margins": margins,
        "metadata": metadata,
        "layers": layers,
        "count": count,
        "skipped": skipped,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "score_transform": args.score_transform,
        "normalize_delta": args.normalize_delta,
        "hidden_position": args.hidden_position,
        "manifest_file": args.manifest_file,
    }

    output_file = Path(os.path.expanduser(args.output_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_file)
    print(f"saved modality attribution vectors to {output_file}")
    print(f"count={count} skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--local-vision-tower", default=DEFAULT_LOCAL_VISION_TOWER)
    parser.add_argument("--image-folder", default="")
    parser.add_argument(
        "--manifest-file",
        default=f"{DEFAULT_MID_DIR}/first_manifest.jsonl",
    )
    parser.add_argument(
        "--output-file",
        default=f"{DEFAULT_MID_DIR}/first_attr_vectors_layers_24_31.pt",
    )
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--layers", default="24-31")
    parser.add_argument("--hidden-position", choices=["last", "mean"], default="last")
    parser.add_argument("--score-transform", choices=["raw", "tanh"], default="tanh")
    parser.add_argument("--normalize-delta", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
