import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "mid_file" / "triton_cache"))

CURRENT_DIR = Path(__file__).resolve().parent
for path in (CURRENT_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from tqdm import tqdm
from transformers import set_seed

from extract_label_steering_vector_vtlm import answer_hidden_states, expanded_prefix_len
from steering_utils import (
    DEFAULT_TEMP_MODEL_NAME,
    DEFAULT_FIRST_HALLU,
    DEFAULT_FIRST_NORMAL,
    DEFAULT_LOCAL_VISION_TOWER,
    DEFAULT_MODEL_PATH,
    DEFAULT_TOUCH_ENCODER_PATH,
    build_prompt,
    iter_yes_no_examples,
    load_modal_tensors,
    load_vtlm_model,
    parse_layers,
    tokenize_prompt,
)

SUBMIT_ROOT = CURRENT_DIR.parent
DEFAULT_PROMPT_DATA_ROOT = SUBMIT_ROOT / "datasets" / "obtain_steering_datasets"
DEFAULT_FIRST_HALLU_PROMPT = str(DEFAULT_PROMPT_DATA_ROOT / "first_0514_hallu.json")
DEFAULT_FIRST_NORMAL_PROMPT = str(DEFAULT_PROMPT_DATA_ROOT / "first_0514_normal.json")
DEFAULT_MID_DIR = str(SUBMIT_ROOT / "mid_file")
ANSWER_INSTRUCTION = "Please answer this question with one word."


def remove_answer_instruction(question):
    question = str(question).rstrip()
    if question.lower().endswith(ANSWER_INSTRUCTION.lower()):
        question = question[:-len(ANSWER_INSTRUCTION)].rstrip()
    return question


def context_hidden_states(row, layers, bundle, image_tensor, touch_tensor, conv_mode, context_position):
    prompt = build_prompt(row["question"], conv_mode, answer=None)
    input_ids = tokenize_prompt(prompt, bundle.tokenizer)
    with torch.inference_mode():
        outputs = bundle.model(
            input_ids=input_ids,
            images=image_tensor,
            touchs=touch_tensor,
            output_hidden_states=True,
            use_cache=False,
        )

    if context_position == "last":
        return {layer: outputs.hidden_states[layer + 1][0, -1].detach().float().cpu() for layer in layers}

    return {layer: outputs.hidden_states[layer + 1][0].mean(dim=0).detach().float().cpu() for layer in layers}


def main(args):
    set_seed(args.seed)
    layers = parse_layers(args.layers)
    bundle = load_vtlm_model(args, run_name=DEFAULT_TEMP_MODEL_NAME)

    features = {layer: [] for layer in layers}
    targets = {layer: [] for layer in layers}
    metadata = []
    count = 0
    skipped = 0

    examples = iter_yes_no_examples(args.train_files)
    for row in tqdm(examples, total=args.max_samples or None):
        if args.max_samples and count >= args.max_samples:
            break

        try:
            no_prompt_row = dict(row)
            no_prompt_row["question"] = remove_answer_instruction(row["question"])
            image_tensor, touch_tensor = load_modal_tensors(row, bundle, args.image_folder)
            prefix_len = expanded_prefix_len(
                no_prompt_row, bundle, image_tensor, touch_tensor, args.conv_mode
            )
            ctx = context_hidden_states(
                no_prompt_row,
                layers,
                bundle,
                image_tensor,
                touch_tensor,
                args.conv_mode,
                args.context_position,
            )
            correct = answer_hidden_states(
                no_prompt_row,
                row["answer"],
                prefix_len,
                layers,
                bundle,
                image_tensor,
                touch_tensor,
                args.conv_mode,
                args.answer_position,
            )
            wrong = answer_hidden_states(
                no_prompt_row,
                row["wrong_answer"],
                prefix_len,
                layers,
                bundle,
                image_tensor,
                touch_tensor,
                args.conv_mode,
                args.answer_position,
            )
        except Exception as exc:
            skipped += 1
            if args.verbose:
                print(f"[skip] {row.get('source_file')}:{row.get('source_index')} {exc}")
            continue

        for layer in layers:
            features[layer].append(ctx[layer])
            targets[layer].append(correct[layer] - wrong[layer])

        metadata.append(
            {
                "source_file": row.get("source_file"),
                "source_index": row.get("source_index"),
                "answer": row.get("answer"),
                "wrong_answer": row.get("wrong_answer"),
                "question": no_prompt_row.get("question", ""),
                "original_question": row.get("question", ""),
                "video_path": row.get("video_path", ""),
                "tactile_path": row.get("tactile_path", ""),
            }
        )
        count += 1

    if count == 0:
        raise RuntimeError("No examples were processed.")

    payload = {
        "features": {layer: torch.stack(features[layer]) for layer in layers},
        "targets": {layer: torch.stack(targets[layer]) for layer in layers},
        "layers": layers,
        "count": count,
        "skipped": skipped,
        "answer_position": args.answer_position,
        "context_position": args.context_position,
        "input_prompt": "without_one_word_instruction",
        "answer_prompt": "without_one_word_instruction",
        "train_files": args.train_files,
        "model_path": args.model_path,
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(os.path.expanduser(args.output_file)), exist_ok=True)
    torch.save(payload, os.path.expanduser(args.output_file))
    print(f"saved L2S training data for layers {layers} to {args.output_file}")
    print(f"count={count} skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--local-vision-tower", default=DEFAULT_LOCAL_VISION_TOWER)
    parser.add_argument("--touch-encoder-path", default=DEFAULT_TOUCH_ENCODER_PATH)
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--train-files", nargs="+", default=[DEFAULT_FIRST_HALLU_PROMPT, DEFAULT_FIRST_NORMAL_PROMPT])
    parser.add_argument(
        "--output-file",
        default=f"{DEFAULT_MID_DIR}/l2s_train_layers_24_31.pt",
    )
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--layers", default="24-31")
    parser.add_argument("--answer-position", choices=["first", "last", "mean"], default="first")
    parser.add_argument("--context-position", choices=["last", "mean"], default="last")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
