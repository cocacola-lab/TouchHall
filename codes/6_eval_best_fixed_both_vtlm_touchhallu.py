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
from tqdm import tqdm
from transformers import set_seed

from attribution_l2s_predictor import load_attribution_l2s_checkpoint
from fact_conditioned_utils import (
    combine_components,
    layer_features,
    predict_components,
    register_last_token_hooks,
    remove_hooks,
)
from l2s_predictor import load_l2s_checkpoint
from steering_utils import (
    DEFAULT_TEMP_MODEL_NAME,
    DEFAULT_LOCAL_VISION_TOWER,
    DEFAULT_TOUCH_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    build_prompt,
    get_stop_str,
    load_json,
    load_modal_tensors,
    load_vtlm_model,
    normalize_yes_no,
    parse_layers,
    prepare_touch_inputs,
    remove_answer_instruction,
    tokenize_prompt,
    trim_stop,
)


SUBMIT_ROOT = CURRENT_DIR.parent
DEFAULT_PROMPT_DATA_ROOT = SUBMIT_ROOT / "datasets" / "touchhall"
DEFAULT_SECOND_HALLU_PROMPT = str(DEFAULT_PROMPT_DATA_ROOT / "VT_inconsistent.json")
DEFAULT_SECOND_NORMAL_PROMPT = str(DEFAULT_PROMPT_DATA_ROOT / "VT_consistent.json")
DEFAULT_NO_PROMPT_MID_DIR = str(SUBMIT_ROOT / "mid_file")
DEFAULT_NO_PROMPT_OUTPUT_DIR = str(SUBMIT_ROOT / "output" / "touchhall")


def split_name(source_file):
    name = os.path.basename(str(source_file))
    if "hallu" in name:
        return "hallu"
    if "inconsistent" in name:
        return "hallu"
    if "normal" in name:
        return "normal"
    if "consistent" in name:
        return "normal"
    return os.path.splitext(name)[0]


def resolve_test_file(test_split: str) -> str:
    if test_split == "hallu":
        return DEFAULT_SECOND_HALLU_PROMPT
    if test_split == "norm":
        return DEFAULT_SECOND_NORMAL_PROMPT
    raise ValueError(f"Unknown test split: {test_split}")


def parse_configs(spec, mean_touch_weight, mean_image_weight):
    available = {
        "fact": (0.0, 0.0),
        "touch": (1.0, 0.0),
        "image": (0.0, 1.0),
        "both": (1.0, 1.0),
        "fixed_mean": (float(mean_touch_weight), float(mean_image_weight)),
    }
    names = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown fixed ablation configs: {unknown}")
    return {name: available[name] for name in names}


def normalize_prediction(text):
    pred = normalize_yes_no(text)
    return pred if pred in {"yes", "no"} else "unknown"


def full_features(row, bundle, args, image_tensor, touch_tensor, layers):
    question = remove_answer_instruction(row["question"])
    input_ids = tokenize_prompt(
        build_prompt(question, args.conv_mode, answer=None), bundle.tokenizer
    )
    with torch.inference_mode():
        outputs = bundle.model(
            input_ids=input_ids,
            images=image_tensor,
            touchs=touch_tensor,
            output_hidden_states=True,
            use_cache=False,
        )
    return layer_features(outputs, layers, args.hidden_position)


def generate_with_vectors(row, bundle, args, image_tensor, touch_tensor, vectors, layers):
    handles = register_last_token_hooks(bundle.model, vectors, layers)
    try:
        input_ids = tokenize_prompt(
            build_prompt(
                row["question"],
                args.conv_mode,
                answer=None,
                add_answer_instruction=False,
            ),
            bundle.tokenizer,
        )
        with torch.inference_mode():
            _, position_ids, attention_mask, _, inputs_embeds, _ = prepare_touch_inputs(
                bundle.model, input_ids, image_tensor, touch_tensor
            )
            output_ids = bundle.model.generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
    finally:
        remove_hooks(handles)
    text = bundle.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return trim_stop(text, get_stop_str(args.conv_mode))


def main(args):
    set_seed(args.seed)
    configs = parse_configs(args.configs, args.mean_touch_weight, args.mean_image_weight)
    if len(configs) != 1:
        raise ValueError("This clean evaluator writes one prediction file per run, so --configs should contain one config only.")
    config_name, (touch_weight, image_weight) = next(iter(configs.items()))

    source_file = os.path.expanduser(args.test_file) if args.test_file else resolve_test_file(args.test_split)
    source_split = split_name(source_file)
    if source_split == "normal":
        source_split = "norm"

    layers = parse_layers(args.layers)
    fact_predictor, fact_payload = load_l2s_checkpoint(os.path.expanduser(args.fact_predictor_file))
    modality_predictor, modality_payload = load_attribution_l2s_checkpoint(os.path.expanduser(args.modality_predictor_file))
    fact_predictor = fact_predictor.cuda().eval()
    modality_predictor = modality_predictor.cuda().eval()
    bundle = load_vtlm_model(
        args,
        run_name=DEFAULT_TEMP_MODEL_NAME,
    )

    rows = load_json(source_file)
    correct = 0
    total = 0
    predictions = []
    for index, row in enumerate(tqdm(rows, desc=f"fixed {config_name} {source_split}")):
        if args.max_samples and index >= args.max_samples:
            break
        image_tensor, touch_tensor = load_modal_tensors(row, bundle, args.image_folder)
        features = full_features(row, bundle, args, image_tensor, touch_tensor, layers)
        components = predict_components(
            fact_predictor,
            fact_payload,
            modality_predictor,
            modality_payload,
            features,
            layers,
            args.normalize_fact,
            args.normalize_modality,
        )
        answer_norm = normalize_yes_no(row.get("answer", ""))
        vectors = combine_components(
            components,
            layers,
            args.alpha_f,
            args.alpha_t,
            args.alpha_v,
            touch_weight,
            image_weight,
        )
        text = generate_with_vectors(
            row,
            bundle,
            args,
            image_tensor,
            touch_tensor,
            vectors,
            layers,
        )
        pred_norm = normalize_prediction(text)
        correct += int(pred_norm == answer_norm)
        total += 1
        predictions.append(
            {
                "question_id": row.get("question_id", index + 1),
                "question": row.get("question", ""),
                "image_path": row.get("video_path", ""),
                "tactile_path": row.get("tactile_path", ""),
                "label": row.get("answer", ""),
                "text": text,
            }
        )

    prediction_dir = Path(os.path.expanduser(args.prediction_dir))
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = Path(os.path.expanduser(args.prediction_file)) if args.prediction_file else prediction_dir / f"{source_split}.jsonl"
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    with prediction_file.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")

    accuracy = correct / total if total else 0.0
    print(f"no_prompt_delta_fixed_{config_name} {source_split}: {correct}/{total} acc={accuracy:.6f}")
    print(f"wrote predictions to {prediction_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--local-vision-tower", default=DEFAULT_LOCAL_VISION_TOWER)
    parser.add_argument("--touch-encoder-path", default=DEFAULT_TOUCH_ENCODER_PATH)
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--fact-predictor-file", default=f"{DEFAULT_NO_PROMPT_MID_DIR}/l2s_predictor_layers_24_31.pt")
    parser.add_argument(
        "--modality-predictor-file",
        default=f"{DEFAULT_NO_PROMPT_MID_DIR}/attr_predictor_no_prompt_delta.pt",
    )
    parser.add_argument("--test-split", choices=["hallu", "norm"], default="norm")
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--configs", default="both")
    parser.add_argument("--mean-touch-weight", type=float, default=0.528)
    parser.add_argument("--mean-image-weight", type=float, default=0.530)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--prediction-dir", default=DEFAULT_NO_PROMPT_OUTPUT_DIR)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--layers", default="24-31")
    parser.add_argument("--hidden-position", choices=["last", "mean"], default="last")
    parser.add_argument("--alpha-f", type=float, default=0.2)
    parser.add_argument("--alpha-t", type=float, default=0.2)
    parser.add_argument("--alpha-v", type=float, default=0.2)
    parser.add_argument("--normalize-fact", action="store_true")
    parser.add_argument("--normalize-modality", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
