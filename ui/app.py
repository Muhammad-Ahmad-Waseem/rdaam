import gradio as gr
import numpy as np
from daam_analyzer import ReverseDAAMAnalyzer
from PIL import Image

# Initialize Analyzer
# Note: This loads the model on startup.
analyzer = ReverseDAAMAnalyzer()

def generate_image(prompt, seed):
    if not prompt:
        return None
    print(f"Generating for prompt: '{prompt}' with seed {seed}")
    img = analyzer.generate(prompt, seed=int(seed))
    # Return image for the sketchpad
    return img

def analyze_drawing(sketch_info):
    # sketch_info is a dict with "image" (original) and "mask" (drawn mask)
    # in newer Gradio versions or "composite" etc.
    # If type="numpy" in Image, for tool="sketch":
    # It returns a dictionary: {'image': array, 'mask': array}
    
    if sketch_info is None:
        return None
        
    mask = sketch_info['mask']
    # Mask is likely (H, W, 4) or (H, W) or (H, W, 3)
    
    fig = analyzer.analyze_mask(mask)
    return fig

with gr.Blocks(title="DAAM Analyzer") as demo:
    gr.Markdown("# Deep Attention Attribution Maps (DAAM) Analyzer")
    gr.Markdown("1. Select Model & Enter Prompt -> Generate.\n2. Draw on the generated image to select a region.\n3. See which words in the prompt contributed to that region.")
    
    with gr.Row():
        with gr.Column():
            model_dd = gr.Dropdown(choices=["CompVis/stable-diffusion-v1-4"], value="CompVis/stable-diffusion-v1-4", label="Model")
            prompt_input = gr.Textbox(label="Prompt", value="A cat with blue eyes and wearing a red hat")
            seed_input = gr.Number(label="Seed", value=42, precision=0)
            gen_btn = gr.Button("Generate Image", variant="primary")
            
        with gr.Column():
            # Image component with sketch tool
            # type="numpy" returns dict with mask
            image_output = gr.Image(label="Generated Image (Draw here)", tool="sketch", type="numpy", interactive=True, height=512)
            
            analyze_btn = gr.Button("Analyze Selected Region")
            plot_output = gr.Plot(label="Attention Distribution")

    # Interactions
    gen_btn.click(fn=generate_image, inputs=[prompt_input, seed_input], outputs=image_output)
    
    # Analyze when button clicked (or could be on change)
    analyze_btn.click(fn=analyze_drawing, inputs=image_output, outputs=plot_output)
    
    # Also trigger analysis when user finishes drawing? 
    # image_output.edit(fn=analyze_drawing, inputs=image_output, outputs=plot_output)

if __name__ == "__main__":
    demo.launch(share=True)
