import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

CURRENT_DIR = Path(__file__).resolve().parent
for path in (CURRENT_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import set_seed

from attribution_l2s_predictor import LayerwiseAttributionPredictor
from split_utils import load_aligned_split


SUBMIT_ROOT = CURRENT_DIR.parent
DEFAULT_NO_PROMPT_MID_DIR = str(SUBMIT_ROOT / "mid_file")


def vector_loss(pred, target, cosine_weight):
    mse = F.mse_loss(pred, target)
    if cosine_weight <= 0:
        return mse
    cos = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()
    return mse + cosine_weight * cos


def evaluate(model, data, indices, device, cosine_weight):
    if len(indices) == 0:
        return {}
    model.eval()
    metrics = {}
    with torch.inference_mode():
        for layer in model.layers:
            x = data["features"][layer][indices].to(device)
            touch_y = data["touch_targets"][layer][indices].to(device)
            image_y = data["image_targets"][layer][indices].to(device)
            pred = model.predict_layer(layer, x)
            touch_loss = vector_loss(pred["touch"], touch_y, cosine_weight)
            image_loss = vector_loss(pred["image"], image_y, cosine_weight)
            touch_cos = F.cosine_similarity(pred["touch"].float(), touch_y.float(), dim=-1).mean()
            image_cos = F.cosine_similarity(pred["image"].float(), image_y.float(), dim=-1).mean()
            metrics[layer] = {
                "loss": float(((touch_loss + image_loss) / 2).cpu()),
                "touch_cosine": float(touch_cos.cpu()),
                "image_cosine": float(image_cos.cpu()),
            }
    return metrics


def main(args):
    set_seed(args.seed)
    payload = torch.load(os.path.expanduser(args.train_data), map_location="cpu")
    layers = [int(layer) for layer in payload["layers"]]
    data = {
        "features": {layer: payload["features"][layer].float() for layer in layers},
        "touch_targets": {layer: payload["touch_targets"][layer].float() for layer in layers},
        "image_targets": {layer: payload["image_targets"][layer].float() for layer in layers},
    }
    hidden_size = data["features"][layers[0]].shape[-1]
    train_idx, val_idx, split = load_aligned_split(args.split_file, payload["metadata"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayerwiseAttributionPredictor(hidden_size, layers, args.bottleneck, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)

    best_val = None
    best_epoch = None
    best_state = None
    history = []

    for epoch in trange(1, args.epochs + 1, desc="train no-prompt-delta attribution L2S"):
        model.train()
        running = 0.0
        steps = 0
        for (batch_idx,) in loader:
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for layer in layers:
                x = data["features"][layer][batch_idx].to(device)
                touch_y = data["touch_targets"][layer][batch_idx].to(device)
                image_y = data["image_targets"][layer][batch_idx].to(device)
                pred = model.predict_layer(layer, x)
                touch_loss = vector_loss(pred["touch"], touch_y, args.cosine_weight)
                image_loss = vector_loss(pred["image"], image_y, args.cosine_weight)
                loss = loss + args.touch_loss_weight * touch_loss + args.image_loss_weight * image_loss
            loss = loss / (len(layers) * (args.touch_loss_weight + args.image_loss_weight))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            running += float(loss.detach().cpu())
            steps += 1

        val_metrics = evaluate(model, data, val_idx, device, args.cosine_weight)
        val_loss = sum(metric["loss"] for metric in val_metrics.values()) / max(1, len(val_metrics))
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, steps),
            "val_loss": val_loss,
            "touch_cosine": {str(layer): val_metrics[layer]["touch_cosine"] for layer in layers if layer in val_metrics},
            "image_cosine": {str(layer): val_metrics[layer]["image_cosine"] for layer in layers if layer in val_metrics},
        }
        history.append(row)
        if args.verbose or epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(row))

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    output = {
        "ablation": "no_prompt_hidden_delta_without_fact_conditioning",
        "target_definition": payload.get(
            "target_definition",
            "unconditioned_prompted_correct_margin_delta_times_no_prompt_hidden_delta_for_touch_and_image",
        ),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "layers": layers,
        "hidden_size": hidden_size,
        "bottleneck": args.bottleneck,
        "dropout": args.dropout,
        "train_data": args.train_data,
        "split_file": args.split_file,
        "split_seed": split["seed"],
        "split_val_ratio": split["val_ratio"],
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
        "target_norms": {
            "touch": {layer: float(data["touch_targets"][layer].norm(dim=-1).mean()) for layer in layers},
            "image": {layer: float(data["image_targets"][layer].norm(dim=-1).mean()) for layer in layers},
        },
    }

    output_file = Path(os.path.expanduser(args.output_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_file)
    print(f"saved no-prompt-delta attribution L2S predictor to {output_file}")
    print(f"best_epoch={best_epoch} best_val_loss={best_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        default=f"{DEFAULT_NO_PROMPT_MID_DIR}/first_attr_train_no_prompt_delta.pt",
    )
    parser.add_argument(
        "--output-file",
        default=f"{DEFAULT_NO_PROMPT_MID_DIR}/attr_predictor_no_prompt_delta.pt",
    )
    parser.add_argument(
        "--split-file",
        default=f"{DEFAULT_NO_PROMPT_MID_DIR}/train_val_split.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--touch-loss-weight", type=float, default=1.0)
    parser.add_argument("--image-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
