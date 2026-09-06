import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "mid_file" / "triton_cache"))

CURRENT_DIR = Path(__file__).resolve().parent
for path in (CURRENT_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import set_seed

from correctness_margin_utils import answer_token_ids, correct_answer_margin, transform_score
from fact_conditioned_utils import hidden_delta, layer_features
from steering_utils import (
    DEFAULT_TEMP_MODEL_NAME,
    DEFAULT_LOCAL_VISION_TOWER,
    DEFAULT_TOUCH_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    build_prompt,
    load_modal_tensors,
    load_vtlm_model,
    parse_layers,
    tokenize_prompt,
)


SUBMIT_ROOT = CURRENT_DIR.parent
DEFAULT_NO_PROMPT_MID_DIR = str(SUBMIT_ROOT / "mid_file")
ANSWER_INSTRUCTION = "Please answer this question with one word."


def iter_jsonl(path):
    with open(os.path.expanduser(path), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def apply_variant(image_tensor, touch_tensor, variant):
    image = torch.zeros_like(image_tensor) if variant.get("image_mode") == "blank" else image_tensor
    touch = torch.zeros_like(touch_tensor) if variant.get("touch_mode") == "blank" else touch_tensor
    return image, touch


def remove_answer_instruction(question):
    question = str(question).rstrip()
    if question.lower().endswith(ANSWER_INSTRUCTION.lower()):
        question = question[:-len(ANSWER_INSTRUCTION)].rstrip()
    return question


def forward_variant(
    row,
    bundle,
    args,
    image_tensor,
    touch_tensor,
    variant,
    keep_answer_instruction,
):
    image, touch = apply_variant(image_tensor, touch_tensor, variant)
    question = row["question"]
    if not keep_answer_instruction:
        question = remove_answer_instruction(question)
    input_ids = tokenize_prompt(
        build_prompt(question, args.conv_mode, answer=None), bundle.tokenizer
    )
    with torch.inference_mode():
        return bundle.model(
            input_ids=input_ids,
            images=image,
            touchs=touch,
            output_hidden_states=True,
            use_cache=False,
        )


def margin(outputs, answer_ids):
    return correct_answer_margin(outputs, answer_ids)[0]


def main(args):
    set_seed(args.seed)
    layers = parse_layers(args.layers)
    bundle = load_vtlm_model(args, run_name=DEFAULT_TEMP_MODEL_NAME)

    features = {layer: [] for layer in layers}
    touch_targets = {layer: [] for layer in layers}
    image_targets = {layer: [] for layer in layers}
    touch_scores, image_scores, metadata = [], [], []
    count = skipped = 0

    for row in tqdm(iter_jsonl(args.manifest_file), total=args.max_samples or None, desc="extract no-prompt-delta attribution"):
        if args.max_samples and count >= args.max_samples:
            break
        try:
            answer_ids = answer_token_ids(bundle.tokenizer, row.get("answer", ""))
            image_tensor, touch_tensor = load_modal_tensors(row, bundle, args.image_folder)
            variants = row["variants"]

            # Predictor inputs and modality contribution scores keep the
            # one-word answer instruction used by the original method.
            score_full_prompted = forward_variant(
                row, bundle, args, image_tensor, touch_tensor, variants["full"], True
            )
            score_no_touch_prompted = forward_variant(
                row, bundle, args, image_tensor, touch_tensor, variants["no_touch"], True
            )
            score_no_image_prompted = forward_variant(
                row, bundle, args, image_tensor, touch_tensor, variants["no_image"], True
            )
            touch_score = transform_score(
                margin(score_full_prompted, answer_ids)
                - margin(score_no_touch_prompted, answer_ids),
                args.score_transform,
            )
            image_score = transform_score(
                margin(score_full_prompted, answer_ids)
                - margin(score_no_image_prompted, answer_ids),
                args.score_transform,
            )

            # Hidden-state changes are computed from the same modality
            # ablations after removing the one-word answer instruction.
            delta_full_no_prompt = forward_variant(
                row, bundle, args, image_tensor, touch_tensor, variants["full"], False
            )
            delta_no_touch_no_prompt = forward_variant(
                row, bundle, args, image_tensor, touch_tensor, variants["no_touch"], False
            )
            delta_no_image_no_prompt = forward_variant(
                row, bundle, args, image_tensor, touch_tensor, variants["no_image"], False
            )
            input_features = layer_features(
                delta_full_no_prompt, layers, args.hidden_position
            )

            for layer in layers:
                delta_touch = hidden_delta(
                    delta_full_no_prompt,
                    delta_no_touch_no_prompt,
                    layer,
                    args.hidden_position,
                )
                delta_image = hidden_delta(
                    delta_full_no_prompt,
                    delta_no_image_no_prompt,
                    layer,
                    args.hidden_position,
                )
                if args.normalize_delta:
                    delta_touch = F.normalize(delta_touch, dim=-1)
                    delta_image = F.normalize(delta_image, dim=-1)
                features[layer].append(input_features[layer].detach().half().cpu())
                touch_targets[layer].append((touch_score * delta_touch).detach().half().cpu())
                image_targets[layer].append((image_score * delta_image).detach().half().cpu())
            touch_scores.append(float(touch_score.cpu()))
            image_scores.append(float(image_score.cpu()))
            metadata.append({key: row.get(key) for key in ("sample_id", "split", "source_file", "source_index", "question", "answer", "video_path", "tactile_path")})
            count += 1
        except Exception as exc:
            skipped += 1
            if args.verbose:
                print(f"[skip] {row.get('sample_id')} {exc}")

    if not count:
        raise RuntimeError("No no-prompt-delta attribution examples were processed.")
    payload = {
        "features": {layer: torch.stack(features[layer]) for layer in layers},
        "touch_targets": {layer: torch.stack(touch_targets[layer]) for layer in layers},
        "image_targets": {layer: torch.stack(image_targets[layer]) for layer in layers},
        "touch_scores": torch.tensor(touch_scores),
        "image_scores": torch.tensor(image_scores),
        "metadata": metadata,
        "layers": layers,
        "count": count,
        "skipped": skipped,
        "alpha_f": 0.0,
        "score_transform": args.score_transform,
        "normalize_delta": args.normalize_delta,
        "hidden_position": args.hidden_position,
        "input_feature_prompt": "without_one_word_instruction",
        "score_prompt": "with_one_word_instruction",
        "hidden_delta_prompt": "without_one_word_instruction",
        "target_definition": "unconditioned_prompted_correct_margin_delta_times_no_prompt_hidden_delta_for_touch_and_image",
    }
    output = Path(os.path.expanduser(args.output_file))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved no-prompt-delta attribution data to {output}")
    print(f"count={count} skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--local-vision-tower", default=DEFAULT_LOCAL_VISION_TOWER)
    parser.add_argument("--touch-encoder-path", default=DEFAULT_TOUCH_ENCODER_PATH)
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--manifest-file", default=f"{DEFAULT_NO_PROMPT_MID_DIR}/first_manifest.jsonl")
    parser.add_argument(
        "--output-file",
        default=f"{DEFAULT_NO_PROMPT_MID_DIR}/first_attr_train_no_prompt_delta.pt",
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
