from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    reconstruction: float = 1.0
    structure: float = .05
    lut_smoothness: float = .001
    local_energy: float = .001
    gate_penalty: float = 0.0
    range_penalty: float = .01
    style_invariance: float = .01
    style_sensitivity: float = .01
    projection_penalty: float = 0.0


def _smooth(volume):
    if not volume.numel(): return volume.sum()
    return sum(volume.diff(dim=d).square().mean() for d in [-1, -2, -3]) / 3


class ToneLoss(nn.Module):
    """Supervised baseline. No learned reward or assertion of perfect style labels."""
    def __init__(self, config=None):
        super().__init__(); self.config = config or LossConfig()

    def forward(self, prediction, target, content, aux):
        zero = prediction.new_zeros(())
        reconstruction = F.l1_loss(prediction, target)
        # Match target gradients: original content's exact gradients would penalize
        # legitimate tone/contrast edits. Target artifacts still require review.
        dx = F.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1]) if prediction.shape[-1] > 1 else zero
        dy = F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]) if prediction.shape[-2] > 1 else zero
        structure = (dx + dy) / 2
        lut_smoothness = _smooth(aux['global_delta']) + _smooth(aux['local_luts'])
        local_energy = aux['proxy_local'].square().mean()
        gate_penalty = aux['gates'].mean() if aux['gates'].numel() else zero
        raw = aux['preclamp_output']; range_penalty = (F.relu(-raw) + F.relu(raw-1)).mean()
        basis = aux['proxy_basis']; local = aux['proxy_local'].flatten(2).transpose(1,2)
        projection_penalty = (basis.transpose(1,2) @ local / basis.shape[1]).square().mean()
        pieces = dict(reconstruction=reconstruction, structure=structure, lut_smoothness=lut_smoothness,
                      local_energy=local_energy, gate_penalty=gate_penalty, range_penalty=range_penalty,
                      style_invariance=aux.get('style_invariance',zero),
                      style_sensitivity=aux.get('style_sensitivity',zero),
                      projection_penalty=projection_penalty)
        total = sum(getattr(self.config, key) * value for key, value in pieces.items())
        return {'total': total, **pieces}
