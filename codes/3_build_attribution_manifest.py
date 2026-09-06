import argparse
import json
import os
from pathlib import Path
from transformers import set_seed
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

CODE_DIR = Path(__file__).resolve().parent
SUBMIT_ROOT = CODE_DIR.parent
DEFAULT_PROMPT_DATA_ROOT = SUBMIT_ROOT / "datasets" / "obtain_steering_datasets"
DEFAULT_MID_DIR = str(SUBMIT_ROOT / "mid_file")
DEFAULT_FILES = {
    "first_hallu": f"{DEFAULT_PROMPT_DATA_ROOT}/first_0514_hallu.json",
    "first_normal": f"{DEFAULT_PROMPT_DATA_ROOT}/first_0514_normal.json",
    # "second_hallu": str(SUBMIT_ROOT / "datasets" / "touchhall" / "VT_inconsistent.json"),
    # "second_normal": str(SUBMIT_ROOT / "datasets" / "touchhall" / "VT_consistent.json"),
}


def load_json(path):
    with open(os.path.expanduser(path), "r") as f:
        return json.load(f)


def normalize_yes_no(text):
    text = str(text).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text.split()[0] if text.split() else text


def build_row(split_name, source_file, source_index, row):
    answer = normalize_yes_no(row.get("answer", ""))
    return {
        "sample_id": f"{split_name}:{source_index}",
        "split": split_name,
        "source_file": source_file,
        "source_index": source_index,
        "question": row.get("question", ""),
        "answer": answer,
        "video_path": row.get("video_path", ""),
        "tactile_path": row.get("tactile_path", ""),
        "variants": {
            "full": {
                "image_mode": "original",
                "touch_mode": "original",
            },
            "no_touch": {
                "image_mode": "original",
                "touch_mode": "blank",
            },
            "no_image": {
                "image_mode": "blank",
                "touch_mode": "original",
            },
        },
    }


def main(args):
    set_seed(args.seed)
    rows = []
    for split_name in args.splits:
        path = DEFAULT_FILES[split_name] if split_name in DEFAULT_FILES else split_name
        data = load_json(path)
        for idx, row in enumerate(data):
            answer = normalize_yes_no(row.get("answer", ""))
            if args.yes_no_only and answer not in {"yes", "no"}:
                continue
            rows.append(build_row(split_name, path, idx, row))

    output_file = Path(os.path.expanduser(args.output_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["first_hallu", "first_normal"],
        help="Split names from DEFAULT_FILES or explicit JSON paths.",
    )
    parser.add_argument(
        "--output-file",
        default=f"{DEFAULT_MID_DIR}/first_manifest.jsonl",
    )
    parser.add_argument("--yes-no-only", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
