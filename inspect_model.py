from diffusers import DiffusionPipeline
import torch

# Load Stable Diffusion v1.4
model_id = "CompVis/stable-diffusion-v1-4"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Print UNet structure and find cross-attention modules
print("UNet Modules:")
for name, module in pipe.unet.named_modules():
    # Cross-attention in SD 1.x is usually in Transformer2DModel -> BasicTransformerBlock -> Attention (attn2)
    # attn1 is self-attention, attn2 is cross-attention
    if "attn2" in name and "to_k" in str(module): # Check if it seems to be an Attention mechanism
       # to_k or similar is part of Attention
       pass

    # A better way is to look for the class name
    if module.__class__.__name__ == "Attention":
        # Check if it's cross attention. 
        # In diffusers Attention, if query_dim == cross_attention_dim (or checks context_dim/encoder_hidden_states presence in forward)
        # But name check is usually easier for standard UNet.
        pass

# Let's just print names containing 'attn' to filter
for name, module in pipe.unet.named_modules():
    if "attn2" in name:
        print(f"Layer: {name}, Module: {module.__class__.__name__}")
