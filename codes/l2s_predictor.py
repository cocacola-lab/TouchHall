import torch
import torch.nn as nn


class LayerwiseSteeringPredictor(nn.Module):
    def __init__(self, hidden_size, layers, bottleneck=256, dropout=0.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layers = [int(layer) for layer in layers]
        self.predictors = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, bottleneck),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(bottleneck, self.hidden_size),
                )
                for layer in self.layers
            }
        )

    def forward(self, features):
        return {layer: self.predictors[str(layer)](features[layer]) for layer in self.layers}


def load_l2s_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model = LayerwiseSteeringPredictor(
        hidden_size=payload["hidden_size"],
        layers=payload["layers"],
        bottleneck=payload["bottleneck"],
        dropout=0.0,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
