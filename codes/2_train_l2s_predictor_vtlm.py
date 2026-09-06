import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

CURRENT_DIR = Path(__file__).resolve().parent
for path in (CURRENT_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import set_seed

from l2s_predictor import LayerwiseSteeringPredictor
from split_utils import load_aligned_split

SUBMIT_ROOT = CURRENT_DIR.parent
DEFAULT_MID_DIR = str(SUBMIT_ROOT / "mid_file")


def layer_loss(pred, target, cosine_weight):
    mse = F.mse_loss(pred, target)
    if cosine_weight <= 0:
        return mse
    cos = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()
    return mse + cosine_weight * cos


def evaluate(model, features, targets, indices, device, cosine_weight):
    if len(indices) == 0:
        return {}
    model.eval()
    metrics = {}
    with torch.inference_mode():
        for layer in model.layers:
            x = features[layer][indices].to(device)
            y = targets[layer][indices].to(device)
            pred = model.predictors[str(layer)](x)
            loss = layer_loss(pred, y, cosine_weight)
            cos = F.cosine_similarity(pred.float(), y.float(), dim=-1).mean()
            metrics[layer] = {"loss": float(loss.cpu()), "cosine": float(cos.cpu())}
    return metrics


def main(args):
    set_seed(args.seed)
    payload = torch.load(os.path.expanduser(args.train_data), map_location="cpu")
    layers = [int(layer) for layer in payload["layers"]]
    features = {layer: payload["features"][layer].float() for layer in layers}
    targets = {layer: payload["targets"][layer].float() for layer in layers}
    hidden_size = features[layers[0]].shape[-1]
    train_idx, val_idx, split = load_aligned_split(args.split_file, payload["metadata"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayerwiseSteeringPredictor(hidden_size, layers, args.bottleneck, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)

    best_val = None
    best_epoch = None
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in trange(1, args.epochs + 1, desc="train L2S"):
        model.train()
        running = 0.0
        steps = 0
        for (batch_idx,) in loader:
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for layer in layers:
                x = features[layer][batch_idx].to(device)
                y = targets[layer][batch_idx].to(device)
                pred = model.predictors[str(layer)](x)
                loss = loss + layer_loss(pred, y, args.cosine_weight)
            loss = loss / len(layers)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            running += float(loss.detach().cpu())
            steps += 1

        val_metrics = evaluate(model, features, targets, val_idx, device, args.cosine_weight)
        val_loss = sum(metric["loss"] for metric in val_metrics.values()) / max(1, len(val_metrics))
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, steps),
            "val_loss": val_loss,
            "val_cosine": {
                str(layer): val_metrics[layer]["cosine"] for layer in layers if layer in val_metrics
            },
        }
        history.append(row)
        if args.verbose or epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(row))

        improved = best_val is None or val_loss < best_val - args.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_loss={best_val}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    output = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "layers": layers,
        "hidden_size": hidden_size,
        "bottleneck": args.bottleneck,
        "dropout": args.dropout,
        "target_norms": {layer: float(targets[layer].norm(dim=-1).mean()) for layer in layers},
        "train_data": args.train_data,
        "split_file": args.split_file,
        "split_seed": split["seed"],
        "split_val_ratio": split["val_ratio"],
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
    }

    os.makedirs(os.path.dirname(os.path.expanduser(args.output_file)), exist_ok=True)
    torch.save(output, os.path.expanduser(args.output_file))
    print(f"saved L2S predictor to {args.output_file}")
    print(f"best_epoch={best_epoch} best_val_loss={best_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        default=f"{DEFAULT_MID_DIR}/l2s_train_layers_24_31.pt",
    )
    parser.add_argument(
        "--output-file",
        default=f"{DEFAULT_MID_DIR}/l2s_predictor_layers_24_31.pt",
    )
    parser.add_argument(
        "--split-file",
        default=f"{DEFAULT_MID_DIR}/train_val_split.pt",
    )
    parser.add_argument("--epochs", type=int, default=100) #将来换100试试。先按30测一致性
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
