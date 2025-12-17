# rdaam: Reverse Deep Attention Attribution Maps

A Python package for analyzing Stable Diffusion attention maps to understand word-to-region and region-to-word attributions in generated images.

![Demo](assets/demo.gif)

## Quick Start

```bash
# 1. Activate environment
conda activate rdaam

# 2. Navigate to directory where you have placed rdaam
cd path/to/parent/directory

# 3. Use the package
python
>>> from rdaam import run_reverse_daam
>>> # Ready to analyze!
```

## Package Structure

```
rdaam/                         # Main package directory
├── __init__.py                # Package entry point
│
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── hooks.py               # AttentionHookManager
│   ├── processing.py          # Attention map processing
│   └── constants.py           # Configuration constants
│
├── analysis/                   # Analysis implementations
│   ├── __init__.py
│   ├── reverse.py             # Reverse DAAM (bbox → words)
│   └── forward.py             # Forward DAAM (word → regions)
│
├── ui/                         # User interface components
│   ├── __init__.py
│   ├── analyzer.py            # ReverseDAAMAnalyzer class
│   └── app.py                 # Gradio web interface
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   └── logging.py             # Logging configuration
│
└── scripts/                    # Standalone scripts
    ├── quantitative_analysis.py
    └── compare_daam.py
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create a new conda environment (named rdaam)
conda create -n rdaam python=3.10 -y
conda activate rdaam

# Navigate to rdaam package directory
cd /path/to/rdaam/directory

# Install the package in editable mode
pip install -e .
```

### Verify Installation

```python
# Test imports
python -c "from rdaam import run_reverse_daam, run_forward_daam, ReverseDAAMAnalyzer; print('✓ Installation successful!')"
```

### Important Usage Notes

> [!IMPORTANT]
> **Always activate the conda environment before using rdaam:**
> ```bash
> # Environment name is 'rdaam', package name is 'rdaam'
> conda activate rdaam
> ```

> [!NOTE]
> **Import Location:** Due to Python's editable install behavior, import `rdaam` from **outside** the package directory:
> ```bash
> # ✓ Correct - from parent directory or elsewhere
> cd /path/to/parent/directory
> python
> >>> import rdaam  # Works!
> 
> # ✗ Incorrect - from inside rdaam/ directory
> cd path/to/rdaam/directory
> python
> >>> import rdaam  # Will fail
> ```

### Dependencies

Core dependencies (automatically installed):
- torch >= 2.0.0
- diffusers >= 0.21.0
- transformers >= 4.30.0
- matplotlib
- pillow
- numpy
- tqdm

Optional dependencies:
```bash
# For Gradio UI
pip install gradio>=3.50.0
```

## Usage

### Basic Usage - Reverse DAAM

```python
from rdaam import run_reverse_daam

# Analyze which words contributed to a bounding box region
image, tokens, scores = run_reverse_daam(
    prompt="A cat with blue eyes",
    seed=42,
    steps=50,
    model_id="CompVis/stable-diffusion-v1-4",
    bbox=[100, 100, 300, 300]  # [x_min, y_min, x_max, y_max]
)

# tokens: ['cat', 'with', 'blue', 'eyes']
# scores: [0.25, 0.15, 0.45, 0.15]  # attention scores
```

### Forward DAAM

```python
from rdaam import run_forward_daam
from diffusers import DiffusionPipeline

# Load pipeline once
pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

# Analyze which regions a word contributed to
image, heatmap, mask = run_forward_daam(
    pipe=pipe,
    prompt="A cat with blue eyes",
    word="eyes",
    seed=42
)

# heatmap: (H, W) tensor showing attention for "eyes"
# mask: binary mask after thresholding
```

### Interactive Analysis

```python
from rdaam import ReverseDAAMAnalyzer

# For interactive applications (e.g., Gradio)
analyzer = ReverseDAAMAnalyzer()

# Generate image
image = analyzer.generate("A cat with a hat", seed=42)

# Analyze a drawn mask
fig = analyzer.analyze_mask(mask_array)
```

## API Reference

### Main Functions

**`run_reverse_daam(prompt, seed, steps, model_id, bbox)`**
- Analyze word contributions to a bounding box region
- Returns: (image, token_list, attention_scores)

**`run_forward_daam(pipe, prompt, word, seed, steps, tau)`**
- Analyze spatial regions a word contributed to
- Returns: (image, heatmap, binary_mask)

### Classes

**`ReverseDAAMAnalyzer(model_id, device)`**
- Stateful analyzer for interactive applications
- Methods:
  - `generate(prompt, seed, steps)` → Image
  - `analyze_mask(mask)` → matplotlib Figure

**`AttentionHookManager`** (Advanced)
- Manages attention hook registration and collection
- Methods:
  - `register_hooks(unet)` → hook_handles
  - `reset()` → clear stored maps

## Examples

### Run Standalone Script

```bash
cd rdaam
python -m analysis.reverse
```

### Launch Gradio UI

```bash
cd rdaam/ui
python app.py
```

### Run Quantitative Analysis

```bash
cd rdaam/scripts
python quantitative_analysis.py
```

## Configuration

Edit `core/constants.py` to change default parameters:

```python
DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_SEED = 42
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
```

## Logging

```python
from r_daam.utils.logging import setup_logger
import logging

# Enable debug logging
logger = setup_logger(__name__, level=logging.DEBUG)

# Log to file
logger = setup_logger(__name__, log_file="daam_analysis.log")
```

## Development

```bash
# Run tests (when available)
pytest tests/

# Check code style
flake8 rdaam/
mypy rdaam/
```

## Citation

If you use this package in your research, please cite:
```bibtex
@software{rdaam2024,
  title={r-daam: Reverse Deep Attention Attribution Maps},
  author={DAAM Analysis Team},
  year={2024}
}
```

## License

MIT License

---

For more information, see the documentation in each module's docstrings.
