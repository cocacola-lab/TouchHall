import torch
import torch.nn as nn


class LayerwiseAttributionPredictor(nn.Module):
    def __init__(self, hidden_size, layers, bottleneck=256, dropout=0.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layers = [int(layer) for layer in layers]
        self.touch_heads = nn.ModuleDict()
        self.image_heads = nn.ModuleDict()
        for layer in self.layers:
            self.touch_heads[str(layer)] = self._make_head(bottleneck, dropout)
            self.image_heads[str(layer)] = self._make_head(bottleneck, dropout)

    def _make_head(self, bottleneck, dropout):
        return nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, self.hidden_size),
        )

    def predict_layer(self, layer, feature):
        key = str(layer)
        return {
            "touch": self.touch_heads[key](feature),
            "image": self.image_heads[key](feature),
        }


def load_attribution_l2s_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model = LayerwiseAttributionPredictor(
        hidden_size=payload["hidden_size"],
        layers=payload["layers"],
        bottleneck=payload["bottleneck"],
        dropout=0.0,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
