# ShapeForge

Local AI text-to-3D for the desktop, built on [OpenAI Shap-E](https://github.com/openai/shap-e).

ShapeForge is a focused 3D asset tool, not just a prompt box. Its main idea is
**Batch Build**: write a list of prompts, let your GPU work through them one
after another, and get every model in its own folder, previewed, cleaned up and
ready to export. Everything runs on your machine. No cloud account needed.

## Features

**Generation**
- Single prompt generation
- **Batch Build**: up to 10 prompts, run sequentially, one output folder per model
- Persistent engine: Shap-E is loaded once and reused, so each model after the
  first skips the model-loading time
- Failed jobs never stop the rest of a batch; **Stop After Current** is available

**Smart prompt input** (Batch Build)

| You type | You get |
|---|---|
| One prompt per line | One job per line |
| A numbered or bulleted list (`1.`, `2)`, `-`, `*`) | One job per item, wrapped lines joined |
| `a red cube; a blue sphere` | Split on `;` |
| `a {wooden\|iron} sword` | Two prompts (several groups multiply out) |
| `a low-poly tree x3` | Three separate generations, for variations |
| `# note` or `// note` | Ignored |

A live counter shows how many prompts were detected before you start.

**Prompt enhancement**
- **Enhance** button for single and batch prompts, undoable with Ctrl+Z
- Built-in cleanup (strips "please generate..." wrappers) and a **Style** hint
  (low-poly, stylized, realistic, cartoon, voxel) that works with no extra software
- Optional **AI rewrite** through a local [Ollama](https://ollama.com) model.
  ShapeForge detects Ollama, picks a small suitable chat model automatically, and
  can download one for you. Without Ollama, everything else still works.

**Preview**
- 3D viewport with Smooth, Flat, Wireframe and Solid + Edges modes
- Mesh info line (faces, vertices, size, closed or open surface)
- Models rest on the grid; Reset Camera button
- Clickable thumbnail strip of finished batch results

**Mesh cleanup**
- Remove stray fragments (adjustable size cutoff)
- Fill holes
- Volume-preserving smoothing
- Decimate to a target face count (optional dependency, see below)
- The original is never modified: results are saved as `generated_clean.obj/.ply`,
  with an **Original / Cleaned** toggle to compare
- **Auto-clean batch** applies your settings to every model as it finishes

**Export**
- OBJ, PLY, STL (printing) and GLB (game engines, Blender)
- Shap-E vertex colors are preserved through cleanup and export (OBJ, PLY, GLB;
  STL has no color)
- Orientation control: GLB is written Y-up as glTF expects; other formats keep the
  generated orientation unless you choose otherwise

**Blender add-on**
- Type a prompt in Blender's sidebar and the generated model appears in your scene
  (see [Blender add-on](#blender-add-on) below)

## Requirements

- Linux (developed on Pop!_OS / Ubuntu 24.04)
- Python 3.10+
- NVIDIA GPU with a CUDA-enabled PyTorch (developed on an RTX 3060 12 GB)
- A separate Shap-E installation

Optional:
- [Ollama](https://ollama.com/download) for AI prompt rewriting
- `fast-simplification` for mesh decimation

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

Make sure the same Python interpreter can import Shap-E and PyTorch with CUDA.

Optional extras:

    python3 -m pip install --no-deps fast-simplification   # enables Decimate

For AI prompt rewriting, install Ollama from https://ollama.com/download and make
sure it is running. ShapeForge's **Set up AI...** button walks you through the rest.

## Run

Set `SHAPE_E_HOME` to your Shap-E folder:

    SHAPE_E_HOME="$HOME/shap-e" python3 main.py

The Shap-E model files are downloaded on first use into
`$SHAPE_E_HOME/shap_e_model_cache/` (a few GB).

## Blender add-on

`blender/shapeforge_blender.py` adds a **ShapeForge** tab to the 3D Viewport
sidebar: type a prompt, press Generate, and the model is imported into your scene.

It is a thin client. Blender's bundled Python cannot host PyTorch, so the add-on
starts ShapeForge's own worker (`engine/server.py`) with your normal Python install,
sends it prompts, and imports the result. Shap-E stays loaded between generations.
The ShapeForge desktop app does not need to be open.

Install: Blender > Edit > Preferences > Add-ons > *Install from Disk...* >
`shapeforge_blender.py`, then enable it. In the add-on preferences set:

| Setting | Default |
|---|---|
| Python | `~/.local/python3.10/bin/python3` if present, else `python3` (must be the one with PyTorch + CUDA) |
| ShapeForge folder | `~/shap-e/ShapeForge` |
| Shap-E folder | `~/shap-e` |

Then press **N** in the 3D Viewport, open the **ShapeForge** tab, type a prompt and
click **Generate**. The first generation loads Shap-E (about 10 seconds); later ones
reuse it. Import options: size, place at the 3D cursor, up axis, shade smooth,
vertex-color material (visible in Material Preview and Render), stray-fragment
removal, and an optional Decimate modifier. **Engine > Stop Engine** frees the
roughly 3 GB of GPU memory.

Tested against Blender 5.0.1's Python API. Flatpak/Snap builds of Blender are
sandboxed and cannot start a host Python, so use a regular Blender install.

## Output

Single generations are saved in `generated/`. Batch runs get their own folders:

    outputs/
    └── batch_YYYYMMDD_HHMMSS/
        ├── 01_a_low_poly_penguin/
        │   ├── generated.obj
        │   ├── generated.ply
        │   ├── generated_clean.obj    (after cleanup)
        │   └── generated_clean.ply
        └── 02_a_wooden_chest/
            └── ...

Previous batches are never overwritten.

## Project layout

| File | Role |
|---|---|
| `main.py` | Main window, single and batch generation, viewport |
| `engine/server.py` | Persistent Shap-E worker (JSON lines over stdin/stdout) |
| `engine/client.py` | Starts and talks to the worker; falls back to the one-shot generator |
| `engine/generate.py` | One-shot generator (fallback) |
| `engine/batchtools.py` | Prompt parsing and result thumbnails |
| `engine/enhance.py`, `engine/enhance_ui.py` | Prompt enhancement and the optional Ollama integration |
| `engine/meshtools.py`, `engine/cleanup_ui.py` | Mesh cleanup and export |
| `blender/shapeforge_blender.py` | Blender add-on (client for `engine/server.py`) |

## Notes and limitations

- Shap-E is a research model. Meshes are often lumpy and may contain small stray
  pieces; that is what the cleanup tools are for.
- After the first generation the worker keeps Shap-E in GPU memory (roughly 3 GB)
  until you close the app.
- Decimating to very low face counts can open holes in a mesh; ShapeForge warns
  when that happens.
- The AI rewrite only offers small chat models, so it fits in VRAM next to Shap-E.

## Roadmap

- [x] v0.1 Basic text-to-3D
- [x] v0.2 Persistent engine and improved preview
- [x] v0.3 Batch Build
- [x] v0.4 Prompt enhancement
- [x] v0.5 Mesh cleanup and optimization

## License

MIT. See [LICENSE](LICENSE). Shap-E is a separate project by OpenAI; see its
repository for its own license and terms.
