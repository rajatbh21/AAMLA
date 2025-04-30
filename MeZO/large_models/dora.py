import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math

# ==============================
# 1) Module Search Utility
# ==============================
def find_module(root_module: nn.Module, key: str):
    """
    Find a sub-module in a Transformer model by its key name.
    Returns (parent module, attribute name, module instance).
    """
    sub_keys = key.split(".")
    parent = root_module
    for sub in sub_keys[:-1]:
        parent = getattr(parent, sub)
    return parent, sub_keys[-1], getattr(parent, sub_keys[-1])


# ==============================
# 2) LoRA Adapter (Original Implementation)
# ==============================
class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1,
                 lora_dropout=0., fan_in_fan_out=False, merge_weights=False, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r if r > 0 else 1
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else (lambda x: x)
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False

        if r > 0:
            # LoRA low-rank parameters
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.weight.requires_grad = False  # Freeze the original weight

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        def T(w): return w.transpose(0, 1) if self.fan_in_fan_out else w
        super().train(mode)
        if self.merge_weights:
            if mode and self.merged:
                # Unmerge weights when returning to training mode
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
            elif not mode and not self.merged:
                # Merge weights in evaluation mode
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x):
        def T(w): return w.transpose(0, 1) if self.fan_in_fan_out else w
        # Base linear transformation
        result = F.linear(x, T(self.weight), bias=self.bias)
        # Apply LoRA adapter if not merged
        if self.r > 0 and not self.merged:
            result = result + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return result


# ==============================
# 3) DoRA Adapter
# ==============================
class DoRALinear(LoRALinear):
    """
    DoRA: Decomposes weight into magnitude and direction,
    training a scalar parameter for magnitude updates and a LoRA (direction) adapter together.
    """
    def __init__(self, in_features, out_features, r=0, alpha=1,
                 lora_dropout=0., fan_in_fan_out=False, merge_weights=False, eps=1e-6, **kwargs):
        
        super().__init__(in_features, out_features,
                         r=r, lora_alpha=alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         merge_weights=merge_weights,
                         **kwargs)
        # Record the original weight's magnitude (L2 norm) per output dimension
        with torch.no_grad():
            mag = self.weight.data.norm(dim=1)
        self.register_buffer('orig_mag', mag)

        # Learnable parameter for magnitude changes per output dimension
        self.mag_delta = nn.Parameter(torch.zeros(out_features))

        self.eps = eps

    def forward(self, x):
        # 1) baseline output: W0 x
        base = F.linear(x,
                        self.weight.transpose(0, 1) if self.fan_in_fan_out else self.weight,
                        bias=self.bias)
        # 2) LoRA output: W0 x + Δ_dir x
        lora_out = super().forward(x)

        # 3) Compute magnitude scale: new_mag/orig_mag = 1 + mag_delta / (orig_mag + eps)
        scale = 1 + self.mag_delta / (self.orig_mag + self.eps)  # shape: [out_features]

        # 4) Apply scale to base output (unsqueeze to match batch dimension)
        scaled_base = base * scale.unsqueeze(0)

        # 5) Final output: scaled_base + (lora_out - base)
        return scaled_base + (lora_out - base)


# ==============================
# 4) DoRA Model Injector
# ==============================
class DoRA:
    """
    Helper class to inject DoRALinear modules into a Transformer model.
    Replaces q_proj, v_proj, or qkv_proj similarly to LoRA.
    """
    def __init__(self, model: nn.Module, r: int, alpha: int, float16: bool = False):
        self.model = model
        self.r = r
        self.alpha = alpha
        self.float16 = float16

        # Determine attention module key based on model type
        t = model.config.model_type
        if t in ["opt", "codegen"]:    attn_key = "attn"
        elif t == "roberta":           attn_key = "attention"
        elif t == "llama":             attn_key = "self_attn"
        else: raise NotImplementedError(f"DoRA: unsupported model_type={t}")

        # Iterate through all modules to find and replace attention blocks
        for name, _ in model.named_modules():
            if name.endswith(attn_key):
                logging.info(f"[DoRA] Inject into: {name}")
                parent, attr, attn = find_module(model, name)

                # OPT/LLAMA families: replace q_proj and v_proj separately
                if t in ["opt", "llama"]:
                    for proj in ["q_proj", "v_proj"]:
                        orig = getattr(attn, proj)
                        new = DoRALinear(
                            orig.in_features, orig.out_features,
                            r=self.r, alpha=self.alpha,
                            lora_dropout=orig.lora_dropout.p if hasattr(orig, "lora_dropout") else 0.,
                            fan_in_fan_out=getattr(orig, "fan_in_fan_out", False),
                            merge_weights=getattr(orig, "merge_weights", False),
                            bias=orig.bias is not None
                        ).to(orig.weight.device)
                        if self.float16: new.half()
                        # Copy original weight/bias
                        new.weight.data = orig.weight.data.clone()
                        if orig.bias is not None:
                            new.bias.data = orig.bias.data.clone()
                        setattr(attn, proj, new)

                # CodeGen family: apply to qkv_proj as a single module
                elif t == "codegen":
                    orig = attn.qkv_proj
                    new = DoRALinear(
                        orig.in_features, orig.out_features,
                        r=self.r, alpha=self.alpha,
                        lora_dropout=orig.lora_dropout.p if hasattr(orig, "lora_dropout") else 0.,
                        fan_in_fan_out=getattr(orig, "fan_in_fan_out", False),
                        merge_weights=getattr(orig, "merge_weights", False),
                        bias=orig.bias is not None
                    ).to(orig.weight.device)
                    if self.float16: new.half()
                    new.weight.data = orig.weight.data.clone()
                    if orig.bias is not None:
                        new.bias.data = orig.bias.data.clone()
                    attn.qkv_proj = new

                else:
                    raise NotImplementedError(f"DoRA injection for model_type={t}")

        # Freeze all parameters except those of LoRA/DoRA adapters
        for n, p in model.named_parameters():
            if "mag_delta" not in n and "lora_" not in n:
                p.requires_grad = False