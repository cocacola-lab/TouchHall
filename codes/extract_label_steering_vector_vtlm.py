import argparse
import os
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "mid_file" / "triton_cache"))

import torch
from tqdm import tqdm
from transformers import set_seed

from steering_utils import (
    DEFAULT_FIRST_HALLU,
    DEFAULT_FIRST_NORMAL,
    DEFAULT_LOCAL_VISION_TOWER,
    DEFAULT_MODEL_PATH,
    build_prompt,
    iter_yes_no_examples,
    load_modal_tensors,
    load_vtlm_model,
    parse_layers,
    prepare_touch_inputs,
    tokenize_prompt,
)


def answer_hidden_states(row, answer, prefix_len, layers, bundle, image_tensor, touch_tensor, conv_mode, answer_position):
    prompt = build_prompt(row["question"], conv_mode, answer=answer)
    input_ids = tokenize_prompt(prompt, bundle.tokenizer)
    with torch.inference_mode():
        outputs = bundle.model(
            input_ids=input_ids,
            images=image_tensor,
            touchs=touch_tensor,
            output_hidden_states=True,
            use_cache=False,
        )

    full_len = outputs.hidden_states[0].shape[1]
    if prefix_len >= full_len:
        raise ValueError(f"prefix_len={prefix_len} >= full_len={full_len}")

    if answer_position == "first":
        pos = prefix_len
        return {layer: outputs.hidden_states[layer + 1][0, pos].detach().float().cpu() for layer in layers}
    if answer_position == "last":
        pos = full_len - 1
        return {layer: outputs.hidden_states[layer + 1][0, pos].detach().float().cpu() for layer in layers}

    return {
        layer: outputs.hidden_states[layer + 1][0, prefix_len:full_len].mean(dim=0).detach().float().cpu()
        for layer in layers
    }


def expanded_prefix_len(row, bundle, image_tensor, touch_tensor, conv_mode):
    prompt = build_prompt(row["question"], conv_mode, answer=None)
    input_ids = tokenize_prompt(prompt, bundle.tokenizer)
    with torch.inference_mode():
        _, _, _, _, inputs_embeds, _ = prepare_touch_inputs(bundle.model, input_ids, image_tensor, touch_tensor)
    return inputs_embeds.shape[1]


def main(args):
    set_seed(args.seed)
    layers = parse_layers(args.layers)
    bundle = load_vtlm_model(args)
    sums = {layer: None for layer in layers}
    count = 0
    skipped = 0

    examples = iter_yes_no_examples(args.train_files)
    for row in tqdm(examples, total=args.max_samples or None):
        if args.max_samples and count >= args.max_samples:
            break
        try:
            image_tensor, touch_tensor = load_modal_tensors(row, bundle, args.image_folder)
            prefix_len = expanded_prefix_len(row, bundle, image_tensor, touch_tensor, args.conv_mode)
            correct = answer_hidden_states(
                row, row["answer"], prefix_len, layers, bundle, image_tensor, touch_tensor, args.conv_mode, args.answer_position
            )
            wrong = answer_hidden_states(
                row, row["wrong_answer"], prefix_len, layers, bundle, image_tensor, touch_tensor, args.conv_mode, args.answer_position
            )
        except Exception as exc:
            skipped += 1
            if args.verbose:
                print(f"[skip] {row.get('source_file')}:{row.get('source_index')} {exc}")
            continue

        for layer in layers:
            diff = correct[layer] - wrong[layer]
            sums[layer] = diff if sums[layer] is None else sums[layer] + diff
        count += 1

    if count == 0:
        raise RuntimeError("No steering examples were processed.")

    vectors = {layer: (sums[layer] / count) for layer in layers}
    payload = {
        "vectors": vectors,
        "layers": layers,
        "count": count,
        "skipped": skipped,
        "answer_position": args.answer_position,
        "train_files": args.train_files,
        "model_path": args.model_path,
    }

    os.makedirs(os.path.dirname(os.path.expanduser(args.output_file)), exist_ok=True)
    torch.save(payload, os.path.expanduser(args.output_file))
    print(f"saved steering vectors for layers {layers} to {args.output_file}")
    print(f"count={count} skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--local-vision-tower", default=DEFAULT_LOCAL_VISION_TOWER)
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--train-files", nargs="+", default=[DEFAULT_FIRST_HALLU, DEFAULT_FIRST_NORMAL])
    parser.add_argument(
        "--output-file",
        default=str(Path(__file__).resolve().parents[1] / "mid_file" / "vtlm_label_flip_vectors.pt"),
    )
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--layers", default="16,18,20,22,24")
    parser.add_argument("--answer-position", choices=["first", "last", "mean"], default="first")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
