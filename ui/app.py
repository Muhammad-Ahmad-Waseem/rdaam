import gradio as gr
import numpy as np
from rdaam import ReverseDAAMAnalyzer
from PIL import Image

# Initialize Analyzer
# Note: This loads the model on startup.
analyzer = ReverseDAAMAnalyzer()

def generate_image(prompt, seed, steps):
    if not prompt:
        return None
    print(f"Generating for prompt: '{prompt}' with seed {seed}, steps {steps}")
    img = analyzer.generate(prompt, seed=int(seed), steps=int(steps))
    # Return image for the sketchpad
    return img

def analyze_drawing(editor_data):
    # editor_data is a dict from ImageEditor
    # keys: 'background', 'layers', 'composite'
    if editor_data is None:
        return None
    
    # Check if there are layers (user drawing)
    layers = editor_data.get("layers", [])
    if not layers:
        print("No layers found.")
        return None
        
    # The drawing is usually in the first layer (or last?)
    # For now assume single drawing layer
    # Layer is (H, W, 4) if RGBA
    mask_layer = layers[0]
    
    # Create mask from alpha channel or non-zero pixels
    # If RGBA, take alpha. If RGB, check non-zero.
    if mask_layer.ndim == 3 and mask_layer.shape[2] == 4:
        mask = mask_layer[:, :, 3]
    else:
        # Fallback: assume any non-zero pixel is part of mask
        mask = np.any(mask_layer > 0, axis=-1).astype(np.uint8) * 255

    # Analyzer expects mask where > 0 is selected region
    fig = analyzer.analyze_mask(mask)
    return fig

with gr.Blocks(title="DAAM Analyzer") as demo:
    gr.Markdown("# Reverse Deep Attention Attribution Maps (rDAAM) Analyzer")
    gr.Markdown("1. Select Model & Enter Prompt -> Generate.\n2. Draw on the generated image to select a region.\n3. See which words in the prompt contributed to that region.")
    
    with gr.Row():
        with gr.Column():
            model_dd = gr.Dropdown(choices=["CompVis/stable-diffusion-v1-4"], value="CompVis/stable-diffusion-v1-4", label="Model")
            prompt_input = gr.Textbox(label="Prompt", value="A cat with blue eyes and wearing a red hat")
            seed_input = gr.Number(label="Seed", value=42, precision=0)
            steps_input = gr.Number(label="Inference Steps", value=30, precision=0)
            gen_btn = gr.Button("Generate Image", variant="primary")
            
        with gr.Column():
            # Image component with sketch tool replaced by ImageEditor
            image_output = gr.ImageEditor(
                label="Generated Image (Draw here)", 
                type="numpy", 
                interactive=True, 
                height=512,
                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed") # White brush for visibility? Or default.
            )
            
            analyze_btn = gr.Button("Analyze Selected Region")
            plot_output = gr.Plot(label="Attention Distribution")

    # Interactions
    gen_btn.click(fn=generate_image, inputs=[prompt_input, seed_input, steps_input], outputs=image_output)
    
    # Analyze when button clicked (or could be on change)
    analyze_btn.click(fn=analyze_drawing, inputs=image_output, outputs=plot_output)
    
    # Also trigger analysis when user finishes drawing? 
    # image_output.edit(fn=analyze_drawing, inputs=image_output, outputs=plot_output)

if __name__ == "__main__":
    demo.launch(share=True)
