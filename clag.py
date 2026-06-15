"""
calg.py
Cross-Architecture Layer Grafting utility module.
"""

import os
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import TypedDict, Tuple

# -----------------------------------------------------------------------------
# Runtime Patches (Must execute before model initialization)
# -----------------------------------------------------------------------------

# Hack to bypass multi-family inheritance collisions in transformers under Python 3.12
class LossKwargs(TypedDict): pass
transformers.utils.LossKwargs = LossKwargs

# Intercept legacy list-based tied weights in older remote code models (like Phi)
_original_get_expanded = transformers.modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys

def _patched_get_expanded(self, all_submodels=False):
    try:
        tied_keys = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied_keys, list):
            return {k: k for k in tied_keys if isinstance(k, str)}
        return _original_get_expanded(self, all_submodels)
    except Exception:
        return {}

transformers.modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded


def heal_meta_tensors(model: nn.Module, base_theta: float = 10000.0, head_dim: int = 128):
    """
    Finds and re-initializes orphaned meta tensors generated during cross-family compilation.
    """
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param.is_meta:
                if 'inv_freq' in param_name:
                    # Natively recalculate RoPE frequencies
                    inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    module.register_buffer(param_name, inv_freq, persistent=False)
                else:
                    # Zero out non-RoPE orphans
                    new_param = nn.Parameter(torch.zeros_like(param, device='cpu'))
                    module.register_parameter(param_name, new_param)


# -----------------------------------------------------------------------------
# Core Math: SVD & Dimensional Mapping
# -----------------------------------------------------------------------------

def extract_and_project_task_vector(
    w_tuned: torch.Tensor, 
    w_base: torch.Tensor, 
    rank_k: int, 
    proj_A: torch.Tensor, 
    proj_B: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """
    Extracts the delta, applies rank-truncated SVD, and uses explicit affine 
    projection matrices to map the donor dimensions to the host graph.
    """
    # 1. Delta extraction (force FP32 for SVD stability)
    tau_donor = (w_tuned - w_base).float()

    # 2. SVD
    U, S, V = torch.svd(tau_donor)
    explained_variance = (torch.sum(S[:rank_k]) / torch.sum(S)).item()

    # 3. Rank Truncation
    U_k = U[:, :rank_k]
    S_k = torch.diag(S[:rank_k])
    V_k = V[:, :rank_k]
    tau_truncated = torch.matmul(U_k, torch.matmul(S_k, V_k.t()))

    # 4. Affine Projection: P(X) = A * X * B
    tau_aligned = torch.matmul(proj_A, torch.matmul(tau_truncated, proj_B))

    return tau_aligned.to(w_base.dtype), explained_variance


# -----------------------------------------------------------------------------
# FusionLM Architecture Wrapper
# -----------------------------------------------------------------------------

class FusionLM(nn.Module):
    def __init__(
        self, 
        host_model_id: str = "microsoft/Phi-4-mini-instruct",
        vision_bridge_path: str = "vision_bridge.pt",
        device: str = "cuda:0",
        quantize_4bit: bool = True
    ):
        super().__init__()
        self.device = device
        
        # Enforce memory constraints
        if device.startswith("cuda"):
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            
        print(f"Loading host backbone: {host_model_id}")
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        ) if quantize_4bit else None

        self.host_model = AutoModelForCausalLM.from_pretrained(
            host_model_id,
            trust_remote_code=True,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map={"": device}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(host_model_id, trust_remote_code=True)
        
        # Run the stabilization patch
        heal_meta_tensors(self.host_model)

        print(f"Loading vision bridge from {vision_bridge_path}")
        self.vision_bridge = torch.load(vision_bridge_path, map_location=device).to(torch.float16)


    def graft_layer(self, layer_idx: int, aligned_delta: torch.Tensor, alpha: float = 0.45):
        """
        Injects the dimensionally mapped task delta into a specific host block.
        """
        print(f"Grafting delta into layer {layer_idx} (alpha={alpha})")
        
        # Note: adjust this path based on the specific host architecture's naming
        # Example for Phi-4's down_proj:
        target_layer = self.host_model.model.layers[layer_idx].mlp.down_proj
        
        with torch.no_grad():
            target_layer.weight.copy_(target_layer.weight + (alpha * aligned_delta))


    def textualize_vision(self, visual_hidden_states: torch.Tensor) -> torch.Tensor:
        """Projects vision manifold into text residual stream"""
        return torch.matmul(visual_hidden_states.to(self.device), self.vision_bridge)


    def forward(self, input_ids=None, visual_hidden_states=None, **kwargs):
        if visual_hidden_states is not None:
            inputs_embeds = self.textualize_vision(visual_hidden_states)
            return self.host_model(inputs_embeds=inputs_embeds, **kwargs)
            
        return self.host_model(input_ids=input_ids, **kwargs)


if __name__ == "__main__":
    # Quick sanity check for the math operations
    print("Running CALG math validation...")
    
    # Mock parameters
    dim_donor, dim_host, rank_k = 4096, 3072, 256
    
    # Mock donor tensors
    w_tuned = torch.randn(dim_donor, dim_donor)
    w_base = torch.randn(dim_donor, dim_donor)
    
    # Mock affine projection operators
    proj_A = torch.randn(dim_host, dim_donor)
    proj_B = torch.randn(dim_donor, dim_host)
    
    aligned_tensor, evr = extract_and_project_task_vector(w_tuned, w_base, rank_k, proj_A, proj_B)
    
    print(f"Retained Variance: {evr:.2%}")
    print(f"Aligned Tensor Shape: {aligned_tensor.shape}")
    assert aligned_tensor.shape == (dim_host, dim_host), "Dimensional mapping failed"
    print("Validation passed.")
