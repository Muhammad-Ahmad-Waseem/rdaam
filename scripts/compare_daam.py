import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_generation import daam_heatmap
from my_daam import run_manual_daam

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)

prompt = "A cat with blue eyes and wearing a red hat"
word = "eyes"
seed = 42
steps = 30
tau = 0.5

print("Running Manual DAAM...")
img_manual, heatmap_manual, mask_manual = run_manual_daam(pipe, prompt, word, seed=seed, steps=steps, tau=tau)

print("Running Library DAAM...")
img_lib, _, heatmap_lib = daam_heatmap(pipe, prompt, word, seed=seed, steps=steps)

# Compare
print("Visualizing comparison...")
plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
plt.imshow(img_manual)
plt.title("Manual Image")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(heatmap_manual.numpy(), cmap='jet')
plt.title("Manual Heatmap")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(heatmap_lib, cmap='jet')
plt.title("Library Heatmap")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(img_lib)
plt.title("Library Image")
plt.axis("off")

# Calculate Diff
if heatmap_lib.shape != heatmap_manual.shape:
    print(f"Shape mismatch: Lib {heatmap_lib.shape} vs Manual {heatmap_manual.shape}")
    # Resize Lib to match Manual
    # Expected lib shape: (H_latent, W_latent) -> (64, 64)
    # Manual shape: (H, W) -> (512, 512)
    
    # Convert lib to tensor
    lib_tensor = torch.tensor(heatmap_lib).float().unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)
    target_size = heatmap_manual.shape # (512, 512)
    
    lib_resized = torch.nn.functional.interpolate(lib_tensor, size=target_size, mode='bicubic', align_corners=False).squeeze()
    
    print(f"Resized Lib to {lib_resized.shape}")
    
    heatmap_lib_resized = lib_resized
else:
    heatmap_lib_resized = torch.tensor(heatmap_lib)

diff = heatmap_lib_resized - heatmap_manual
mse = (diff ** 2).mean().item()
print(f"MSE between heatmaps: {mse}")

plt.subplot(1, 5, 5)
plt.imshow(diff.numpy(), cmap='bwr')
plt.title(f"Diff (MSE: {mse:.4f})")
plt.colorbar()
plt.axis("off")

plt.savefig("daam_comparison_result.png")
print("Saved daam_comparison_result.png")
