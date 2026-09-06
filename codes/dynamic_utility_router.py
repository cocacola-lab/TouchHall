import torch
import torch.nn as nn


ACTION_NAMES = ["fact", "touch", "image", "both"]


class DynamicUtilityRouter(nn.Module):
    def __init__(self, hidden_size, bottleneck=256, dropout=0.05):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(bottleneck)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(bottleneck), len(ACTION_NAMES)),
        )

    def forward(self, hidden_state):
        return self.network(hidden_state)


def load_router_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model = DynamicUtilityRouter(payload["hidden_size"], payload["bottleneck"], dropout=0.0)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def destandardize_utilities(prediction, payload):
    mean = torch.tensor(payload["utility_mean"], device=prediction.device, dtype=prediction.dtype)
    std = torch.tensor(payload["utility_std"], device=prediction.device, dtype=prediction.dtype)
    return prediction * std.clamp_min(1e-6) + mean


def route_from_utilities(utilities, temperature=1.0, weight_floor=0.0):
    action_index = int(utilities.argmax().detach().cpu())
    action = ACTION_NAMES[action_index]
    temperature = max(float(temperature), 1e-6)
    touch_gate = float(torch.sigmoid(utilities[1] / temperature).detach().cpu())
    image_gate = float(torch.sigmoid(utilities[2] / temperature).detach().cpu())

    touch_weight = touch_gate if action in ("touch", "both") else 0.0
    image_weight = image_gate if action in ("image", "both") else 0.0
    if touch_weight > 0:
        touch_weight = max(touch_weight, float(weight_floor))
    if image_weight > 0:
        image_weight = max(image_weight, float(weight_floor))
    return {
        "action": action,
        "action_index": action_index,
        "touch_weight": touch_weight,
        "image_weight": image_weight,
    }
