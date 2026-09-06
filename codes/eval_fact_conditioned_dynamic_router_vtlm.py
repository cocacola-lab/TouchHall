import argparse
import json
import os
import sys
from collections import Counter
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
from dynamic_utility_router import destandardize_utilities, load_router_checkpoint, route_from_utilities
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
    DEFAULT_SECOND_HALLU,
    DEFAULT_SECOND_NORMAL,
    build_prompt,
    get_stop_str,
    load_json,
    load_modal_tensors,
    load_vtlm_model,
    normalize_yes_no,
    parse_layers,
    prepare_touch_inputs,
    tokenize_prompt,
    trim_stop,
)


SUBMIT_ROOT = CURRENT_DIR.parent
DEFAULT_VALID_OUTPUT_ROOT = str(SUBMIT_ROOT / "mid_file")


def full_features(row, bundle, args, image_tensor, touch_tensor, layers):
    input_ids = tokenize_prompt(build_prompt(row["question"], args.conv_mode, answer=None), bundle.tokenizer)
    with torch.inference_mode():
        outputs = bundle.model(
            input_ids=input_ids,
            images=image_tensor,
            touchs=touch_tensor,
            output_hidden_states=True,
            use_cache=False,
        )
    return layer_features(outputs, layers, args.hidden_position)


def generate_one(row, bundle, fact_predictor, fact_payload, modality_predictor, modality_payload, router, router_payload, args, layers):
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
    with torch.inference_mode():
        router_input = features[args.router_layer].to(next(router.parameters()).device).unsqueeze(0)
        utilities = destandardize_utilities(router(router_input)[0], router_payload)
    route = route_from_utilities(utilities, args.route_temperature, args.weight_floor)
    vectors = combine_components(
        components,
        layers,
        args.alpha_f,
        args.alpha_t,
        args.alpha_v,
        route["touch_weight"],
        route["image_weight"],
    )

    handles = register_last_token_hooks(bundle.model, vectors, layers)
    try:
        prompt = build_prompt(row["question"], args.conv_mode, answer=None)
        input_ids = tokenize_prompt(prompt, bundle.tokenizer)
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
    route["predicted_utilities"] = {
        name: float(value.detach().cpu()) for name, value in zip(router_payload["action_names"], utilities)
    }
    return trim_stop(text, get_stop_str(args.conv_mode)), route


def evaluate_file(path, bundle, fact_predictor, fact_payload, modality_predictor, modality_payload, router, router_payload, args, layers):
    rows = load_json(path)
    correct = errors = 0
    action_counts = Counter()
    touch_weight_sum = image_weight_sum = 0.0
    predictions = []
    for index, row in enumerate(tqdm(rows, desc=f"fact-conditioned router {os.path.basename(path)}")):
        if args.max_samples and index >= args.max_samples:
            break
        try:
            text, route = generate_one(
                row, bundle, fact_predictor, fact_payload, modality_predictor, modality_payload, router, router_payload, args, layers
            )
            pred_norm = normalize_yes_no(text)
            answer_norm = normalize_yes_no(row.get("answer", ""))
            is_correct = pred_norm == answer_norm
        except Exception as exc:
            errors += 1
            text = pred_norm = ""
            answer_norm = normalize_yes_no(row.get("answer", ""))
            is_correct = False
            route = {"action": "error", "touch_weight": 0.0, "image_weight": 0.0, "error": str(exc)}
            if args.verbose:
                print(f"[error] {path}:{index} {exc}")
        correct += int(is_correct)
        action_counts[route["action"]] += 1
        touch_weight_sum += route["touch_weight"]
        image_weight_sum += route["image_weight"]
        if args.save_predictions:
            predictions.append({
                "source_file": path,
                "question_id": index + 1,
                "prompt": row.get("question", ""),
                "text": text,
                "pred_norm": pred_norm,
                "answer": row.get("answer", ""),
                "answer_norm": answer_norm,
                "correct": is_correct,
                **route,
            })
    total = min(len(rows), args.max_samples) if args.max_samples else len(rows)
    return {
        "source_file": path,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "action_counts": dict(action_counts),
        "action_rates": {name: count / total for name, count in action_counts.items()},
        "mean_touch_weight": touch_weight_sum / total if total else 0.0,
        "mean_image_weight": image_weight_sum / total if total else 0.0,
        "error_count": errors,
    }, predictions


def main(args):
    set_seed(args.seed)
    fact_predictor, fact_payload = load_l2s_checkpoint(os.path.expanduser(args.fact_predictor_file))
    modality_predictor, modality_payload = load_attribution_l2s_checkpoint(os.path.expanduser(args.modality_predictor_file))
    router, router_payload = load_router_checkpoint(os.path.expanduser(args.router_file))
    layers = parse_layers(args.layers) if args.layers else [int(layer) for layer in router_payload["layers"]]
    args.router_layer = args.router_layer if args.router_layer is not None else int(router_payload["router_layer"])
    args.alpha_f = args.alpha_f if args.alpha_f is not None else float(router_payload["alpha_f"])
    args.alpha_t = args.alpha_t if args.alpha_t is not None else float(router_payload["alpha_t"])
    args.alpha_v = args.alpha_v if args.alpha_v is not None else float(router_payload["alpha_v"])
    bundle = load_vtlm_model(args, run_name=DEFAULT_TEMP_MODEL_NAME)
    fact_predictor, modality_predictor, router = fact_predictor.cuda().eval(), modality_predictor.cuda().eval(), router.cuda().eval()

    summaries, predictions = [], []
    for path in args.test_files:
        summary, rows = evaluate_file(
            path, bundle, fact_predictor, fact_payload, modality_predictor, modality_payload, router, router_payload, args, layers
        )
        summaries.append(summary)
        predictions.extend(rows)
        print(f"{path}: {summary['correct']}/{summary['total']} acc={summary['accuracy']:.6f} actions={summary['action_counts']}")
    overall_correct = sum(row["correct"] for row in summaries)
    overall_total = sum(row["total"] for row in summaries)
    actions = Counter()
    for row in summaries:
        actions.update(row["action_counts"])
    summaries.append({
        "source_file": "overall",
        "correct": overall_correct,
        "total": overall_total,
        "accuracy": overall_correct / overall_total,
        "action_counts": dict(actions),
        "action_rates": {name: count / overall_total for name, count in actions.items()},
        "alpha_f": args.alpha_f,
        "alpha_t": args.alpha_t,
        "alpha_v": args.alpha_v,
        "route_temperature": args.route_temperature,
        "weight_floor": args.weight_floor,
        "layers": layers,
    })
    summary_file = Path(os.path.expanduser(args.summary_file))
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if args.save_predictions:
        prediction_file = Path(os.path.expanduser(args.prediction_file))
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        with prediction_file.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
    print(f"wrote summary to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--local-vision-tower", default=DEFAULT_LOCAL_VISION_TOWER)
    parser.add_argument("--touch-encoder-path", default=DEFAULT_TOUCH_ENCODER_PATH)
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--fact-predictor-file", default=f"{DEFAULT_VALID_OUTPUT_ROOT}/steering/l2s_predictor_layers_24_31.pt")
    parser.add_argument("--modality-predictor-file", default=f"{DEFAULT_VALID_OUTPUT_ROOT}/fact_conditioned_dynamic_router/fact_conditioned_attr_predictor.pt")
    parser.add_argument("--router-file", default=f"{DEFAULT_VALID_OUTPUT_ROOT}/fact_conditioned_dynamic_router/dynamic_router.pt")
    parser.add_argument("--test-files", nargs="+", default=[DEFAULT_SECOND_HALLU, DEFAULT_SECOND_NORMAL])
    parser.add_argument("--summary-file", default=f"{DEFAULT_VALID_OUTPUT_ROOT}/fact_conditioned_dynamic_router/dynamic_router_summary.json")
    parser.add_argument("--prediction-file", default=f"{DEFAULT_VALID_OUTPUT_ROOT}/fact_conditioned_dynamic_router/dynamic_router_predictions.jsonl")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--layers", default="24-31")
    parser.add_argument("--router-layer", type=int, default=None)
    parser.add_argument("--hidden-position", choices=["last", "mean"], default="last")
    parser.add_argument("--alpha-f", type=float, default=None)
    parser.add_argument("--alpha-t", type=float, default=None)
    parser.add_argument("--alpha-v", type=float, default=None)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--weight-floor", type=float, default=0.2)
    parser.add_argument("--normalize-fact", action="store_true")
    parser.add_argument("--normalize-modality", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
