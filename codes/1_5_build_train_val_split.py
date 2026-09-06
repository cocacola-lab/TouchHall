import argparse
from pathlib import Path

import torch
from transformers import set_seed

from split_utils import metadata_keys


CURRENT_DIR = Path(__file__).resolve().parent
SUBMIT_ROOT = CURRENT_DIR.parent
DEFAULT_MID_DIR = SUBMIT_ROOT / "mid_file"


def main(args):
    set_seed(args.seed)
    payload = torch.load(Path(args.train_data).expanduser(), map_location="cpu")
    metadata = payload.get("metadata")
    if metadata is None:
        raise ValueError(f"Training data has no metadata: {args.train_data}")

    keys = metadata_keys(metadata)
    count = len(keys)
    if count != int(payload["count"]):
        raise ValueError(f"metadata count {count} != payload count {payload['count']}")

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(count, generator=generator).tolist()
    val_count = max(1, int(round(count * args.val_ratio))) if count > 1 else 0
    val_positions = permutation[:val_count]
    train_positions = permutation[val_count:]

    split = {
        "train_keys": [keys[index] for index in train_positions],
        "val_keys": [keys[index] for index in val_positions],
        "train_count": len(train_positions),
        "val_count": len(val_positions),
        "sample_count": count,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "source_train_data": str(Path(args.train_data).expanduser()),
    }

    output_file = Path(args.output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(split, output_file)
    print(f"saved shared split to {output_file}")
    print(f"train_count={split['train_count']} val_count={split['val_count']} seed={args.seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        default=str(DEFAULT_MID_DIR / "l2s_train_layers_24_31.pt"),
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_MID_DIR / "train_val_split.pt"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
