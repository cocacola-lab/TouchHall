import torch
import torch.nn.functional as F


def layer_features(outputs, layers, hidden_position):
    features = {}
    for layer in layers:
        hidden = outputs.hidden_states[layer + 1][0]
        if hidden_position == "last":
            features[layer] = hidden[-1].detach().float()
        else:
            features[layer] = hidden.float().mean(dim=0).detach()
    return features


def hidden_delta(full_outputs, ablated_outputs, layer, hidden_position):
    full = full_outputs.hidden_states[layer + 1][0]
    ablated = ablated_outputs.hidden_states[layer + 1][0]
    if hidden_position == "last":
        return (full[-1] - ablated[-1]).detach().float()
    return (full.float().mean(dim=0) - ablated.float().mean(dim=0)).detach()


def normalize_with_target_norm(vector, payload, layer, key=None):
    norms = payload.get("target_norms", {})
    if key is not None:
        norms = norms.get(key, {})
    target_norm = norms.get(layer, norms.get(str(layer)))
    if target_norm is None:
        return vector
    return F.normalize(vector.float(), dim=-1) * float(target_norm)


def predict_components(
    fact_predictor,
    fact_payload,
    modality_predictor,
    modality_payload,
    features,
    layers,
    normalize_fact=False,
    normalize_modality=False,
):
    device = next(fact_predictor.parameters()).device
    components = {}
    with torch.inference_mode():
        for layer in layers:
            feature = features[layer].to(device).unsqueeze(0)
            fact = fact_predictor.predictors[str(layer)](feature)[0]
            modality = modality_predictor.predict_layer(layer, feature)
            touch = modality["touch"][0]
            image = modality["image"][0]
            if normalize_fact:
                fact = normalize_with_target_norm(fact, fact_payload, layer)
            if normalize_modality:
                touch = normalize_with_target_norm(touch, modality_payload, layer, "touch")
                image = normalize_with_target_norm(image, modality_payload, layer, "image")
            components[layer] = {
                "fact": fact.detach(),
                "touch": touch.detach(),
                "image": image.detach(),
            }
    return components


def combine_components(components, layers, alpha_f, alpha_t, alpha_v, touch_weight, image_weight):
    return {
        layer: (
            alpha_f * components[layer]["fact"]
            + alpha_t * float(touch_weight) * components[layer]["touch"]
            + alpha_v * float(image_weight) * components[layer]["image"]
        )
        for layer in layers
    }


def register_last_token_hooks(model, vectors, layers):
    handles = []
    for layer in layers:
        vector = vectors[layer]

        def make_hook(v):
            def hook(_module, _inputs, output):
                hidden = output[0]
                steered = hidden.clone()
                steered[:, -1, :] = steered[:, -1, :] + v.to(device=hidden.device, dtype=hidden.dtype)
                return (steered,) + output[1:]

            return hook

        handles.append(model.model.layers[layer].register_forward_hook(make_hook(vector)))
    return handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()
