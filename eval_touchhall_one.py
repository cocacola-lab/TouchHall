import argparse
import json
import os
import re
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output" / "touchhall"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "datasets" / "touchhall"
DEFAULT_HALLU_QUESTION_FILE = DEFAULT_DATA_DIR / "VT_inconsistent.json"
DEFAULT_NORM_QUESTION_FILE = DEFAULT_DATA_DIR / "VT_consistent.json"
ONE_WORD_PROMPT = "Please answer this question with one word."


def load_json_any(path):
    path = Path(path).expanduser()
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pope_normalize_prediction(text):
    text = str(text).replace("</s>", "").strip().lower()
    match = re.match(r"^\s*([a-zA-Z]+)", text)
    if not match:
        return "unknown"

    first_word = match.group(1)
    if first_word == "yes":
        return "yes"
    if first_word == "no":
        return "no"
    return "unknown"


def normalize_label(label):
    label = str(label).strip().lower()
    if label.startswith("no"):
        return "no"
    if label.startswith("yes"):
        return "yes"
    raise ValueError(f"Expected yes/no label, got: {label!r}")


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def row_id(row, index):
    return row.get("question_id", row.get("id", index + 1))


def normalize_text(text):
    return " ".join(str(text).strip().split())


def normalize_question(text):
    text = normalize_text(text)
    if text.endswith(ONE_WORD_PROMPT):
        text = text[: -len(ONE_WORD_PROMPT)]
    return normalize_text(text)


def normalize_path_value(path):
    path = str(path).strip()
    return os.path.normpath(path) if path else ""


def pred_question(row):
    return row.get("question", row.get("prompt", ""))


def pred_image_path(row):
    return row.get("image_path", row.get("video_path", ""))


def assert_same_field(index, field_name, pred_value, question_value, normalizer=lambda x: x):
    pred_norm = normalizer(pred_value)
    question_norm = normalizer(question_value)
    if pred_norm != question_norm:
        raise ValueError(
            f"{field_name} mismatch at index {index}:\n"
            f"  prediction: {pred_value!r}\n"
            f"  question:   {question_value!r}"
        )


def pair_predictions_with_labels(prediction_file, question_file):
    predictions = load_json_any(prediction_file)
    questions = load_json_any(question_file)
    if len(predictions) != len(questions):
        raise ValueError(
            f"Prediction/question length mismatch: {prediction_file} has {len(predictions)}, "
            f"{question_file} has {len(questions)}"
        )

    paired = []
    validation = {
        "prediction_file": str(prediction_file),
        "question_file": str(question_file),
        "num_predictions": len(predictions),
        "num_questions": len(questions),
        "question_id_match": True,
        "question_text_match": True,
        "image_path_match": True,
        "tactile_path_match": True,
        "label_match": True,
    }
    for index, (pred_row, question_row) in enumerate(zip(predictions, questions)):
        pred_id = row_id(pred_row, index)
        question_id = row_id(question_row, index)
        assert_same_field(index, "question_id", pred_id, question_id, lambda x: str(x))
        assert_same_field(index, "question", pred_question(pred_row), question_row.get("question", ""), normalize_question)
        assert_same_field(index, "image_path", pred_image_path(pred_row), question_row.get("video_path", ""), normalize_path_value)
        assert_same_field(index, "tactile_path", pred_row.get("tactile_path", ""), question_row.get("tactile_path", ""), normalize_path_value)
        if "label" in pred_row or "answer" in pred_row:
            assert_same_field(
                index,
                "label",
                pred_row.get("label", pred_row.get("answer", "")),
                question_row.get("answer", question_row.get("label", "")),
                normalize_label,
            )
        paired.append({
            "question_id": question_id,
            "text": pred_row.get("text", pred_row.get("prediction", "")),
            "label": question_row.get("answer", question_row.get("label", "")),
        })
    return paired, validation


def compute_pope_metrics(rows):
    pred_list = []
    label_list = []
    unknown_count = 0
    for row in rows:
        pred = pope_normalize_prediction(row.get("text", ""))
        label = normalize_label(row.get("label", ""))
        if pred == "unknown":
            pred_list.append(-1)
            unknown_count += 1
        else:
            pred_list.append(1 if pred == "yes" else 0)
        label_list.append(1 if label == "yes" else 0)

    tp = fp = tn = fn = 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1
        elif pred == -1 and label == 1:
            fn += 1
        elif pred == -1 and label == 0:
            fp += 1

    total = len(pred_list)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, total)
    return {
        "ACC": acc,
        "F1": f1,
        "P": precision,
        "R": recall,
        "unknown": safe_div(unknown_count, total),
        "unknown_count": unknown_count,
        "total": total,
    }


def print_pope_metrics(title, metrics):
    print(
        f"{title}: "
        f"ACC={metrics['ACC']:.6f} "
        f"F1={metrics['F1']:.6f} "
        f"P={metrics['P']:.6f} "
        f"R={metrics['R']:.6f} "
        f"unknown={metrics['unknown']:.6f} "
        f"({metrics['unknown_count']}/{metrics['total']}) "
        f"total={metrics['total']}"
    )


def default_question_file(split):
    return DEFAULT_HALLU_QUESTION_FILE if split == "hallu" else DEFAULT_NORM_QUESTION_FILE


def main(args):
    prediction_file = Path(args.prediction_file).expanduser() if args.prediction_file else DEFAULT_OUTPUT_DIR / f"{args.split}.jsonl"
    question_file = Path(args.question_file).expanduser() if args.question_file else default_question_file(args.split)
    rows, validation = pair_predictions_with_labels(prediction_file, question_file)
    metrics = compute_pope_metrics(rows)
    metrics["split"] = args.split
    metrics["prediction_file"] = str(prediction_file)
    metrics["question_file"] = str(question_file)
    metrics["validation"] = validation

    print_pope_metrics(args.split, metrics)

    if args.summary_file:
        summary_file = Path(args.summary_file).expanduser()
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote summary to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["hallu", "norm"], default="hallu")
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--question-file", default=None)
    parser.add_argument("--summary-file", default=None)

    main(parser.parse_args())
