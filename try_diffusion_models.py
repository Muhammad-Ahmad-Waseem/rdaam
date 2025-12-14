from diffusers import DiffusionPipeline#, Kandinsky3Pipeline
import torch
from image_generation import daam_heatmap
import matplotlib.pyplot as plt

# Stable Diffusion v1.5 (512×512 output)
sd_pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                torch_dtype=torch.float16)
sd_pipeline.to("cuda")
# kandinsky_pipeline = Kandinsky3Pipeline.from_pretrained(
#     "kandinsky-community/kandinsky-3", torch_dtype=torch.float16, variant="fp16"
# )
# prompt = "A cat with a red hat"
prompt = "A cat with blue eyes and wearing a red hat"
word = "eyes"
img, word_map, heat_np = daam_heatmap(sd_pipeline, prompt, word)
# img = kandinsky_pipeline(prompt).images[0]
img.save("sd1.5_sample.png")
plt.figure(figsize=(6, 6))
word_map.plot_overlay(img)
plt.title(f"DAAM Overlay for '{word}'")
plt.axis("off")
plt.savefig("daam_overlay.png")
plt.close()
# image1 = sd_pipeline("A cat with a red hat").images[0]
# image1.save("sd1.5_sample.png")
