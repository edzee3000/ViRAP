"""Small set-style encoder and input-anchored, color-projected residual renderer.

This is an unvalidated research hypothesis, not a pretrained model. Projection
properties apply to proxy RGB residuals before final output clipping only.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    style_size: int = 128
    patch_size: int = 16
    token_dim: int = 128
    style_dim: int = 128
    condition_dim: int = 192
    context_size: int = 64
    lut_size: int = 9
    local_rank: int = 2
    projection: str = 'color'
    global_limit: float = .3
    local_limit: float = .1
    local_output_limit: float = .1
    render_chunk_rows: int = 128
    projection_rcond: float = 1e-4
    eval_gate_threshold: float = 0.0
    style_aux: bool = True
    bound_local: bool = True

    def __post_init__(self):
        if self.style_size % self.patch_size or self.patch_size < 2:
            raise ValueError('style_size must be divisible by patch_size >= 2')
        if self.projection not in ('none', 'mean', 'color'):
            raise ValueError('projection must be none, mean or color')
        if self.local_rank < 0 or min(self.context_size, self.lut_size) < 2:
            raise ValueError('invalid context/lut/rank')
        if self.render_chunk_rows < 1 or not 0 <= self.eval_gate_threshold < 1:
            raise ValueError('invalid rendering options')
        if min(self.global_limit, self.local_limit, self.local_output_limit) <= 0:
            raise ValueError('Mapping limits must be positive')


def rgb_to_normalized_lab(rgb):
    """Differentiable sRGB/D65 -> roughly normalized Lab; input in [0,1]."""
    value = rgb.float().clamp(0, 1)
    linear = torch.where(value <= .04045, value / 12.92,
                         ((value + .055) / 1.055).clamp_min(1e-8).pow(2.4))
    matrix = value.new_tensor([[.4124564, .3575761, .1804375],
                               [.2126729, .7151522, .0721750],
                               [.0193339, .1191920, .9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear)
    xyz = xyz / value.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    delta = 6 / 29
    f = torch.where(xyz > delta**3, xyz.clamp_min(1e-8).pow(1 / 3),
                    xyz / (3 * delta**2) + 4 / 29)
    light = (116 * f[:, 1] - 16) / 100
    a = (500 * (f[:, 0] - f[:, 1]) + 128) / 255
    b = (200 * (f[:, 1] - f[:, 2]) + 128) / 255
    return torch.stack([light, a, b], dim=1).clamp(0, 1)


class StyleSetEncoder(nn.Module):
    """No absolute positions, inter-patch convolution or pretrained backbone.

    Exactly invariant (up to rounding) to permutations of already extracted
    patch tokens. This is NOT semantic independence or arbitrary crop invariance.
    """
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.patch_mlp = nn.Sequential(nn.Linear(3 * cfg.patch_size**2, cfg.token_dim),
                                       nn.GELU(), nn.Linear(cfg.token_dim, cfg.token_dim))
        self.project = nn.Sequential(nn.Linear(2 * cfg.token_dim + 54, 256),
                                     nn.GELU(), nn.Linear(256, cfg.style_dim))
        self.register_buffer('hist_centers', torch.linspace(0, 1, 16))

    def patches(self, image):
        image = F.interpolate(image, (self.cfg.style_size, self.cfg.style_size),
                              mode='bilinear', align_corners=False)
        lab = rgb_to_normalized_lab(image)
        return F.unfold(lab, self.cfg.patch_size, stride=self.cfg.patch_size).transpose(1, 2)

    def encode_patches(self, patches):
        tokens = self.patch_mlp(patches)
        pixels = patches.reshape(patches.shape[0], patches.shape[1], 3, -1)
        pixels = pixels.permute(0, 2, 1, 3).flatten(2)
        hist = (1 - (pixels[..., None] - self.hist_centers).abs() * 15).clamp_min(0).mean(2)
        moments = torch.cat([pixels.mean(2), pixels.std(2, correction=0)], dim=1)
        stats = torch.cat([tokens.mean(1), tokens.std(1, correction=0),
                           hist.flatten(1), moments], dim=1)
        return self.project(stats)

    def forward(self, image):
        return self.encode_patches(self.patches(image))


def color_basis(image):
    """Ten fixed RGB polynomial basis functions, NOT all global LUT functions."""
    r, g, b = (image.float() * 2 - 1).unbind(1)
    return torch.stack([torch.ones_like(r), r, g, b, r*r, g*g, b*b,
                        r*g, r*b, g*b], dim=-1)


def identity_lut(size):
    axis = torch.linspace(0, 1, size)
    blue, green, red = torch.meshgrid(axis, axis, axis, indexing='ij')
    return torch.stack([red, green, blue], dim=0).unsqueeze(0)


def sample_lut(volume, image):
    """Axis order is volume[B,RGB,blue,green,red], lookup grid[r,g,b]."""
    grid = (image * 2 - 1).permute(0, 2, 3, 1).unsqueeze(1)
    return F.grid_sample(volume, grid, mode='bilinear', padding_mode='border',
                         align_corners=True).squeeze(2)


def project_residual(raw, image, mode='color', rcond=1e-4):
    """Differentiable in raw. Fixed input basis has no gradient through its SVD.

    SVD truncation gives the numerical column space. No ridge approximation.
    Keep this in FP32 for AMP; pinv coefficients have shape [B,10,3].
    """
    raw32 = raw.float(); basis = color_basis(image.detach()).flatten(1, 2)
    values = raw32.flatten(2).transpose(1, 2)
    coefficients = values.new_zeros(values.shape[0], 10, 3)
    if mode == 'color':
        with torch.no_grad(): inverse = torch.linalg.pinv(basis, rtol=rcond)
        coefficients = inverse @ values
    elif mode == 'mean':
        coefficients = torch.cat([values.mean(1, keepdim=True), coefficients[:, 1:]], 1)
    elif mode != 'none':
        raise ValueError(mode)
    corrected = values - basis @ coefficients
    return corrected.transpose(1, 2).reshape_as(raw32), coefficients, basis


class ContentContext(nn.Module):
    """Reference-free, coordinate-free maps; replicate padding preserves flat fields."""
    def __init__(self, rank):
        super().__init__()
        layers = []
        for a, b in [(6, 48), (48, 64), (64, 64), (64, 64)]:
            layers += [nn.Conv2d(a, b, 3, padding=1, padding_mode='replicate'),
                       nn.GroupNorm(8, b), nn.SiLU()]
        layers += [nn.Conv2d(64, rank, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, image):
        light = (image * image.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)
        padded = F.pad(light, (1, 1, 1, 1), mode='replicate')
        smooth = F.avg_pool2d(padded, 3, stride=1)
        dx = F.pad(light[:, :, :, 1:] - light[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(light[:, :, 1:, :] - light[:, :, :-1, :], (0, 0, 0, 1))
        return self.net(torch.cat([image, light, light - smooth, dx.abs() + dy.abs()], 1))


@dataclass
class RenderPlan:
    global_lut: torch.Tensor
    local_luts: torch.Tensor
    maps: torch.Tensor
    gates: torch.Tensor
    coefficients: torch.Tensor
    local_scale: torch.Tensor
    active: tuple
    aux: dict


class ToneTransferModel(nn.Module):
    def __init__(self, config=None):
        super().__init__(); self.config = config or ModelConfig(); c = self.config
        self.style_encoder = StyleSetEncoder(c)
        self.condition = nn.Sequential(nn.Linear(4 * c.style_dim, 256), nn.GELU(),
                                       nn.Linear(256, c.condition_dim), nn.GELU())
        n = 3 * c.lut_size**3
        self.global_head = nn.Linear(c.condition_dim, n)
        nn.init.zeros_(self.global_head.weight); nn.init.zeros_(self.global_head.bias)
        self.register_buffer('identity', identity_lut(c.lut_size))
        if c.local_rank:
            self.context = ContentContext(c.local_rank)
            self.local_head = nn.Linear(c.condition_dim, c.local_rank * n)
            self.gate_head = nn.Linear(c.condition_dim, c.local_rank)
            nn.init.zeros_(self.local_head.weight); nn.init.zeros_(self.local_head.bias)
            nn.init.zeros_(self.gate_head.weight); nn.init.constant_(self.gate_head.bias, -2.)
        self.local_enabled = True

    def set_local_enabled(self, enabled=True):
        self.local_enabled = bool(enabled)

    def prepare(self, content, reference):
        """Plan is for this content only; safe to reuse at its native resolution."""
        c = self.config; batch = content.shape[0]
        if content.ndim != 4 or reference.ndim != 4 or content.shape[1] != 3 or reference.shape[1] != 3:
            raise ValueError('Expected RGB BCHW input and reference')
        if content.shape[0] != reference.shape[0]: raise ValueError('Batch sizes differ')
        zc = self.style_encoder(content); zr = self.style_encoder(reference)
        z = self.condition(torch.cat([zc, zr, zr-zc, zc*zr], dim=1))
        global_delta = c.global_limit * self.global_head(z).tanh().reshape(batch, 3, c.lut_size, c.lut_size, c.lut_size)
        global_lut = self.identity + global_delta
        proxy = F.interpolate(content, (c.context_size, c.context_size), mode='bilinear', align_corners=False)
        if c.local_rank and self.local_enabled:
            maps = self.context(proxy)
            local_luts = c.local_limit * self.local_head(z).tanh().reshape(batch, c.local_rank, 3, c.lut_size, c.lut_size, c.lut_size)
            gates = self.gate_head(z).sigmoid()
            if not self.training and c.eval_gate_threshold > 0:
                gates = gates * (gates >= c.eval_gate_threshold)
                active = tuple(k for k in range(c.local_rank) if bool((gates[:, k] != 0).any()))
            else: active = tuple(range(c.local_rank))
            raw = sum((sample_lut(local_luts[:, k], proxy) * maps[:, k:k+1] * gates[:, k, None, None, None]
                       for k in active), torch.zeros_like(proxy))
        else:
            maps = proxy.new_zeros(batch, 0, c.context_size, c.context_size)
            local_luts = proxy.new_zeros(batch, 0, 3, c.lut_size, c.lut_size, c.lut_size)
            gates = proxy.new_zeros(batch, 0); active = (); raw = torch.zeros_like(proxy)
        # The projection itself must not be autocast to low precision.
        with torch.autocast(device_type=content.device.type, enabled=False):
            projected, coefficients, basis = project_residual(raw, proxy, c.projection, c.projection_rcond)
            # A polynomial fitted to the proxy can extrapolate on native colors.
            # For RGB in [0,1], |basis_j|<=1 and |raw_c|<=local_limit*sum(gates).
            # One scalar per image bounds the WHOLE residual while preserving
            # proxy orthogonality. Per-pixel clipping here would destroy it.
            coefficient_bound = coefficients.abs().sum(1).amax(1)
            bound = c.local_limit * gates.float().sum(1) + coefficient_bound
            local_scale = (c.local_output_limit / bound.clamp_min(1e-8)).clamp(max=1)
            if not c.bound_local: local_scale = torch.ones_like(local_scale)
            projected = projected * local_scale[:, None, None, None]
        aux = {'global_delta': global_delta, 'local_luts': local_luts, 'maps': maps,
               'gates': gates, 'proxy_raw_local': raw, 'proxy_local': projected,
               'proxy_basis': basis, 'style_content': zc, 'style_reference': zr,
               'local_scale': local_scale, 'coefficient_bound': coefficient_bound,
               'active_local_branches': len(active)}
        return RenderPlan(global_lut, local_luts, maps, gates, coefficients, local_scale, active, aux)

    @staticmethod
    def _map_rows(maps, height, width, start, end):
        yy = (torch.arange(start, end, device=maps.device, dtype=maps.dtype) + .5) * (2 / height) - 1
        xx = (torch.arange(width, device=maps.device, dtype=maps.dtype) + .5) * (2 / width) - 1
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(maps.shape[0], -1, -1, -1)
        return F.grid_sample(maps, grid, align_corners=False, padding_mode='border')

    def _render_window(self, pixels, plan, height, width, start, end):
        base = sample_lut(plan.global_lut, pixels)
        if plan.active:
            maps = self._map_rows(plan.maps, height, width, start, end)
            raw = sum((sample_lut(plan.local_luts[:, k], pixels) * maps[:, k:k+1]
                       * plan.gates[:, k, None, None, None] for k in plan.active), torch.zeros_like(base))
            correction = torch.einsum('bhwm,bmc->bchw', color_basis(pixels), plan.coefficients)
            base = base + (raw - correction) * plan.local_scale[:, None, None, None]
        return base

    def _render_raw(self, content, plan):
        height, width = content.shape[-2:]; chunks = []
        for start in range(0, height, self.config.render_chunk_rows):
            end = min(height, start + self.config.render_chunk_rows)
            chunks.append(self._render_window(content[:, :, start:end], plan, height, width, start, end))
        return torch.cat(chunks, dim=2)

    @torch.inference_mode()
    def iter_render_rows(self, content_cpu, plan):
        """CPU native raster -> device row blocks -> CPU float blocks.

        CPU still stores the original and output. Only lookup temporaries and
        one image strip are on the accelerator; conditioning is prepared once.
        """
        if content_cpu.device.type != 'cpu': raise ValueError('Native source must remain on CPU')
        height, width = content_cpu.shape[-2:]
        for start in range(0, height, self.config.render_chunk_rows):
            end = min(height, start + self.config.render_chunk_rows)
            pixels = content_cpu[:, :, start:end].to(plan.global_lut.device)
            pixels = pixels.float() / 255 if pixels.dtype == torch.uint8 else pixels.float()
            block = self._render_window(pixels, plan, height, width, start, end)
            yield start, end, block.clamp(0, 1).cpu()

    def render(self, content, plan):
        """Row chunks bound temporary LUT/map lookup memory; no spatial RGB warp."""
        return self._render_raw(content, plan).clamp(0, 1)

    def forward(self, content, reference, return_aux=False):
        plan = self.prepare(content, reference); raw = self._render_raw(content, plan)
        result = raw.clamp(0, 1)
        if not return_aux: return result
        aux = {**plan.aux, 'preclamp_output': raw}
        # Generic weak regularizers, NOT a new style-disentanglement algorithm.
        # Never assume unrelated photos have identical styles or that teacher
        # outputs perfectly match their references.
        zero = content.new_zeros(())
        aux['style_invariance'] = zero; aux['style_sensitivity'] = zero
        if self.training and self.config.style_aux:
            flipped = self.style_encoder(reference.flip(-1))
            aux['style_invariance'] = (1-F.cosine_similarity(flipped, aux['style_reference'], dim=1)).mean()
            gains = torch.exp((torch.rand(content.shape[0], 3, 1, 1, device=content.device)-.5)*.4)
            exposure = torch.exp((torch.rand(content.shape[0], 1, 1, 1, device=content.device)-.5)*.5)
            edited = (content*gains*exposure).clamp(0, 1)
            actual_change = (edited-content).abs().mean((1,2,3)).detach()
            distance = 1-F.cosine_similarity(self.style_encoder(edited), aux['style_content'], dim=1)
            aux['style_sensitivity'] = F.relu(actual_change*.5-distance).mean()
        return result, aux
