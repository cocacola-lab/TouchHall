import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CODE_DIR = Path(__file__).resolve().parent
SUBMIT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SUBMIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMIT_ROOT))

import torch
from PIL import Image

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_Touch_TOKEN = "<touch>"
IMAGE_TOKEN_INDEX = -200


DEFAULT_MODEL_PATH = "/data/pengyuzhao/project_hallu_25/tactile_hallu/my_steering/pretrain_models/llava_merged_lora_touch2"
DEFAULT_LOCAL_VISION_TOWER = "/data/pengyuzhao/project_hallu_25/tactile_hallu/my_steering/pretrain_models/clip_vit_large_patch14_336"
DEFAULT_TOUCH_ENCODER_PATH = "/data/pengyuzhao/project_hallu_25/tactile_hallu/my_steering/pretrain_models/touch_encoder.pt"
DEFAULT_TEMP_MODEL_ROOT = str(SUBMIT_ROOT / "mid_file" / "model_cache")
ANSWER_INSTRUCTION = "Please answer this question with one word."


def remove_answer_instruction(question: str) -> str:
    question = str(question).rstrip()
    if question.lower().endswith(ANSWER_INSTRUCTION.lower()):
        question = question[:-len(ANSWER_INSTRUCTION)].rstrip()
    return question
DEFAULT_TEMP_MODEL_NAME = "llava_merged_lora_touch2_local_vision"
DEFAULT_FIRST_HALLU = str(SUBMIT_ROOT / "datasets" / "obtain_steering_datasets" / "first_0514_hallu.json")
DEFAULT_FIRST_NORMAL = str(SUBMIT_ROOT / "datasets" / "obtain_steering_datasets" / "first_0514_normal.json")
DEFAULT_SECOND_HALLU = str(SUBMIT_ROOT / "datasets" / "touchhall" / "VT_inconsistent.json")
DEFAULT_SECOND_NORMAL = str(SUBMIT_ROOT / "datasets" / "touchhall" / "VT_consistent.json")


@dataclass
class VTLMBundle:
    tokenizer: object
    model: torch.nn.Module
    image_processor: object
    touch_processor: object
    model_name: str


def normalize_yes_no(text: str) -> str:
    text = str(text).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text.split()[0] if text.split() else text


def flip_yes_no(answer: str) -> str:
    answer = normalize_yes_no(answer)
    if answer == "yes":
        return "no"
    if answer == "no":
        return "yes"
    raise ValueError(f"Expected yes/no answer, got: {answer!r}")


def parse_layers(spec: str) -> List[int]:
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_json(path: str) -> List[dict]:
    with open(os.path.expanduser(path), "r") as f:
        return json.load(f)


def iter_yes_no_examples(paths: Sequence[str]) -> Iterable[dict]:
    for path in paths:
        for idx, row in enumerate(load_json(path)):
            answer = normalize_yes_no(row.get("answer", ""))
            if answer not in {"yes", "no"}:
                continue
            item = dict(row)
            item["answer"] = answer
            item["wrong_answer"] = flip_yes_no(answer)
            item["source_file"] = path
            item["source_index"] = idx
            yield item


def build_local_model_path(model_path: str, local_vision_tower: str, run_name: str = DEFAULT_TEMP_MODEL_NAME) -> str:
    model_path = os.path.expanduser(model_path)
    local_vision_tower = os.path.expanduser(local_vision_tower)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path) or not os.path.isdir(local_vision_tower):
        return model_path

    with open(config_path, "r") as f:
        config = json.load(f)

    if config.get("mm_vision_tower") == local_vision_tower:
        return model_path

    # Use one stable temporary model-config directory for all experiments.
    run_model_path = os.path.join(DEFAULT_TEMP_MODEL_ROOT, DEFAULT_TEMP_MODEL_NAME)
    os.makedirs(run_model_path, exist_ok=True)

    for name in os.listdir(model_path):
        src = os.path.join(model_path, name)
        dst = os.path.join(run_model_path, name)
        if name == "config.json" or os.path.exists(dst):
            continue
        os.symlink(src, dst)

    config["mm_vision_tower"] = local_vision_tower
    with open(os.path.join(run_model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return run_model_path


def load_vtlm_model(args, run_name: str = DEFAULT_TEMP_MODEL_NAME) -> VTLMBundle:
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_path = build_local_model_path(args.model_path, args.local_vision_tower, run_name)
    # The stable temporary model-cache path may not contain "llava" or "touch".
    # Builder dispatches on model_name, so keep the identity from the real checkpoint path.
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, touch_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        touch_enc_path=getattr(args, "touch_encoder_path", DEFAULT_TOUCH_ENCODER_PATH),
        attn_implementation="eager",
        eval_test=getattr(args, "eval_test", True),
    )
    tokenizer.add_tokens("<touch>")
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return VTLMBundle(tokenizer, model, image_processor, touch_processor, model_name)


def build_prompt(
    question: str,
    conv_mode: str,
    answer: Optional[str] = None,
    add_answer_instruction: bool = False,
) -> str:
    from llava.conversation import conv_templates

    qs = DEFAULT_IMAGE_TOKEN + "\n " + DEFAULT_Touch_TOKEN + "\n" + question
    if add_answer_instruction:
        qs = qs + " Please answer this question with one word."
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], answer)
    return conv.get_prompt()


def get_stop_str(conv_mode: str) -> str:
    from llava.conversation import SeparatorStyle, conv_templates

    conv = conv_templates[conv_mode].copy()
    return conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


def load_modal_tensors(row: dict, bundle: VTLMBundle, image_folder: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    from llava.mm_utils import process_images

    image_file = row["video_path"]
    touch_file = row["tactile_path"]
    if image_file is None:
        image_file = row["video_path"]
    if not os.path.exists(image_file):
        image_file = image_file.replace(".png", ".jpg")
    if not os.path.exists(touch_file):
        touch_file = touch_file.replace(".png", ".jpg")

    image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
    image_tensor = process_images([image], bundle.image_processor, bundle.model.config)[0]
    touch_img = Image.open(os.path.join(image_folder, touch_file)).convert("RGB")
    touch_tensor = bundle.touch_processor(touch_img)
    return image_tensor.unsqueeze(0).half().cuda(), touch_tensor.unsqueeze(0).half().cuda()


def tokenize_prompt(prompt: str, tokenizer) -> torch.Tensor:
    from llava.mm_utils import tokenizer_image_token

    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()


def prepare_touch_inputs(model, input_ids: torch.Tensor, image_tensor: torch.Tensor, touch_tensor: torch.Tensor):
    return model.prepare_inputs_labels_for_touch_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        image_tensor,
        touch_tensor,
    )


def trim_stop(text: str, stop_str: str) -> str:
    text = text.strip()
    if stop_str and text.endswith(stop_str):
        text = text[: -len(stop_str)]
    return text.strip()


def save_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.expanduser(path)), exist_ok=True)
    with open(os.path.expanduser(path), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
