# ShapeForge

Local AI-powered text-to-3D generation for desktop Linux.

ShapeForge provides a lightweight desktop interface for generating,
previewing and exporting 3D assets using OpenAI Shap-E.

## Features

- Local text-to-3D generation
- NVIDIA GPU acceleration through PyTorch
- Interactive 3D mesh preview
- OBJ and PLY export
- No cloud account required

## Requirements

- Python 3.10+
- NVIDIA GPU recommended
- CUDA-compatible PyTorch installation
- Separate Shap-E installation

## Installation

Clone this repository:

    git clone https://github.com/somore100/ShapeForge.git
    cd ShapeForge

Install the interface dependencies:

    python3 -m pip install -r requirements.txt

Install Shap-E separately and follow its dependency instructions:

    git clone https://github.com/openai/shap-e.git ~/shap-e
    python3 -m pip install -e ~/shap-e
    python3 -m pip install blobfile PyYAML ipywidgets
    python3 -m pip install git+https://github.com/openai/CLIP.git

Ensure that the same Python interpreter can import Shap-E and
PyTorch with CUDA support.

## Run

Set SHAPE_E_HOME to the location of your Shap-E installation:

    SHAPE_E_HOME="$HOME/shap-e" python3 main.py

## Output

Generated models are saved in the local generated/ directory.

Supported exports:

- Wavefront OBJ
- PLY

## Project status

Early development. Features and interfaces are subject to change.

## License

License not yet selected.
