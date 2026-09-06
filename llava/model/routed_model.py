import torch


class RoutedTouchVisionLM(torch.nn.Module):

    def __init__(self, vtlm, tlm, threshold=0.3):
        super().__init__()
        self.vtlm = vtlm
        self.tlm = tlm
        self.threshold = threshold

    @torch.no_grad()
    def compute_inconsistency(self, images, touchs):
        z_v, z_t = self.vtlm.extract_modality_embeddings(images, touchs)

        z_v = z_v.mean(dim=1)
        z_t = z_t.mean(dim=1)

        sim = torch.nn.functional.cosine_similarity(z_v, z_t, dim=-1)
        distance = 1 - sim
        return distance

    def forward(self, input_ids, images=None, touchs=None, **kwargs):

        if images is not None and touchs is not None:

            distance = self.compute_inconsistency(images, touchs)

            if distance.mean() > self.threshold:
                # 视觉不可信 → 走 TLM
                return self.tlm(
                    input_ids=input_ids,
                    touchs=touchs,
                    **kwargs
                )
            else:
                # 正常 → 走 VTLM
                return self.vtlm(
                    input_ids=input_ids,
                    images=images,
                    touchs=touchs,
                    **kwargs
                )

        else:
            return self.tlm(input_ids=input_ids, touchs=touchs, **kwargs)