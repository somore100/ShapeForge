# ShapeForge

Local AI text-to-3D for the desktop, built on [OpenAI Shap-E](https://github.com/openai/shap-e).

ShapeForge is a focused 3D asset tool, not just a prompt box. Its main idea is
**Batch Build**: write a list of prompts, let your computer work through them one
after another, and get every model in its own folder, previewed, cleaned up and
ready to export. Everything runs on your machine. No cloud account needed.

## Features

**Generation**
- Single prompt generation
- **Batch Build**: up to 10 prompts, run sequentially, one output folder per model
- Persistent engine: Shap-E is loaded once and reused, so each model after the
  first skips the model-loading time
- **Quality** presets: Draft (fastest), Standard, High detail. They set the number
  of sampling steps and the mesh resolution. Draft is the default on computers
  without an NVIDIA GPU
- Failed jobs never stop the rest of a batch; **Stop After Current** is available
- **Engine setup assistant**: if the Shap-E engine is missing, ShapeForge offers to
  download and install it (see [Installation](#installation))

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
- **Enhance Prompt** and **Enhance Batch** buttons, undoable with Ctrl+Z. Batch
  enhancement leaves comments, `{a|b}` variants and `;` lines alone and keeps list
  markers and `xN` repeats
- **Rules (no AI)** mode, instant and offline. It removes things Shap-E ignores
  (render tags such as "4k" or "masterpiece", background and lighting phrases,
  filler words, repeated words), trims very long prompts, fixes "a/an", and adds
  missing parts to bare vehicle and furniture nouns ("car" becomes "A car with four
  wheels")
- **Style** hint added to enhanced prompts: Low-poly, Cartoon, Realistic, Game
  asset, Toy
- Optional **AI rewrite** through a local [Ollama](https://ollama.com) model.
  ShapeForge detects Ollama, picks a small suitable chat model automatically, and
  can download one for you. The rewrite keeps your subject, mentions recognizable
  parts and avoids inventing details. Without Ollama everything else still works
- Whether enhanced prompts actually produce better models is something to test on
  your own results: run the original and the enhanced prompt as a batch (`x4` each)
  and compare

**Preview**
- 3D viewport with Smooth, Flat, Wireframe and Solid + Edges modes
- **Colors** toggle: shows the model's own colors
- Mesh info line (faces, vertices, size, closed or open surface)
- Models rest on the grid; Reset Camera button
- Clickable thumbnail strip of finished batch results (hidden until a batch runs)
- Hover help: every control has a plain-language tooltip, and a hint line under the
  viewport shows the same text as you move the mouse
- Opens maximized so the whole window always fits the screen

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
  STL cannot store color)
- Orientation control: GLB is written Y-up as glTF expects; other formats keep the
  generated orientation unless you choose otherwise

**About colors**
Shap-E does not make image textures. It paints one color per vertex, so color
detail is limited by how many vertices the mesh has (the High detail preset uses
a finer mesh). Shap-E stores those colors in linear space, which looks dark in
normal viewers, so ShapeForge converts them to sRGB when a model is generated (and
back to linear for GLB, as glTF expects). Models generated with versions older than
v0.7 keep their original dark colors; generate them again.

**Blender add-on**
- Type a prompt in Blender's sidebar and the generated model appears in your scene
  (see [Blender add-on](#blender-add-on) below)

## Requirements

- Linux, Windows or macOS
- Python 3.10 to 3.12 when running from source (PyTorch may not offer builds for the
  newest Python versions yet)
- **NVIDIA GPU recommended** (developed on an RTX 3060 12 GB). Without one,
  generation runs on the CPU: it works, but expect minutes per model, so use the
  Draft preset. AMD and Apple GPUs are not used by ShapeForge's engine
- About 3 GB of disk for the Shap-E models, plus PyTorch (roughly 200 MB for the
  CPU build, 2 GB or more with CUDA)

Optional:
- [Ollama](https://ollama.com/download) for AI prompt rewriting
- `fast-simplification` for mesh decimation

## Installation

### Packaged builds

Download the AppImage (Linux), zip (Windows) or dmg (macOS) from the
[Releases](https://github.com/somore100/ShapeForge/releases) page, once a release is
published. The builds are not code-signed, so Windows SmartScreen and macOS
Gatekeeper will warn you; choose "Run anyway" or right-click > Open. The Windows
and macOS builds have not been tested on real machines yet.

On first use the app offers to set up the engine. It keeps its own private Python
environment, so nothing is installed system-wide. On Linux, the AppImage needs a
normal desktop with working OpenGL drivers.

### From source

    git clone https://github.com/somore100/ShapeForge.git
    cd ShapeForge
    python3 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    python3 -m pip install -r requirements.txt
    python3 main.py

Press **Generate 3D Model**. If the engine is missing, the setup window offers to
download the Shap-E code, install PyTorch (a small CPU build, or CUDA where an
NVIDIA GPU is found) and the other libraries into your current environment, and
check that the engine starts. The Shap-E models (a few GB) download the first time
you generate. The **Engine setup...** button reopens that window at any time.

Where the Shap-E code is looked for, in order: the `SHAPE_E_HOME` environment
variable, an existing `~/shap-e`, then the app data folder
(`~/.local/share/ShapeForge/shap-e` on Linux, `%LOCALAPPDATA%\ShapeForge\shap-e` on
Windows, `~/Library/Application Support/ShapeForge/shap-e` on macOS).

Already have Shap-E working? Point ShapeForge at it and run with the Python that can
import Shap-E and PyTorch:

    SHAPE_E_HOME="$HOME/shap-e" python3 main.py

Qt platform plugin errors on Linux usually mean a missing system library:
`sudo apt install libxcb-cursor0` (Debian, Ubuntu) or
`sudo dnf install xcb-util-cursor` (Fedora).

Optional extras:

    python3 -m pip install fast-simplification   # enables Decimate

For AI prompt rewriting, install Ollama from https://ollama.com/download and make
sure it is running. ShapeForge's **Set up AI...** button walks you through the rest.

## Blender add-on

`blender/shapeforge_blender.py` adds a **ShapeForge** tab to the 3D Viewport
sidebar: type a prompt, press Generate, and the model is imported into your scene.

It is a thin client. Blender's bundled Python cannot host PyTorch, so the add-on
starts ShapeForge's own worker (`engine/server.py`) with your normal Python install,
sends it prompts, and imports the result. Shap-E stays loaded between generations.
The ShapeForge desktop app does not need to be open. It uses the Standard quality
settings.

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

When running from source, single generations are saved in `generated/` and batch
runs in `outputs/` inside the ShapeForge folder. Packaged builds save to
`~/ShapeForge/` instead. Batch runs get their own folders:

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

## Building the packages

GitHub Actions builds all three platforms from `.github/workflows/build.yml`:
run it by hand from the Actions tab (**Build > Run workflow**) for test builds, or
push a tag such as `v0.9.0` to also create a draft release.

To build on your own machine:

    python3 -m pip install -r requirements.txt pyinstaller pillow
    pyinstaller --noconfirm packaging/shapeforge.spec

The packaged app contains only the interface. PyTorch and Shap-E are installed on
first use into a private Python environment (created with
[uv](https://github.com/astral-sh/uv)). Set `SHAPEFORGE_PYTHON` to use an existing
Python instead.

## Project layout

| File | Role |
|---|---|
| `main.py` | Main window, single and batch generation, viewport |
| `engine/server.py` | Persistent Shap-E worker (JSON lines over stdin/stdout) |
| `engine/client.py` | Starts and talks to the worker; falls back to the one-shot generator |
| `engine/generate.py` | One-shot generator (fallback) |
| `engine/engine_setup.py`, `engine/runtime.py` | Engine setup assistant, quality presets, where files and the engine Python live |
| `engine/batchtools.py` | Prompt parsing and result thumbnails |
| `engine/enhance.py`, `engine/enhance_ui.py`, `engine/rules.py` | Prompt enhancement: rules mode and the optional Ollama integration |
| `engine/meshtools.py`, `engine/cleanup_ui.py` | Mesh cleanup and export |
| `engine/colors.py`, `engine/colorspace.py` | Vertex-color preview and color-space conversion |
| `engine/uxtools.py` | Hover help, startup window sizing |
| `blender/shapeforge_blender.py` | Blender add-on (client for `engine/server.py`) |
| `packaging/`, `assets/`, `.github/workflows/build.yml` | Packaged builds (AppImage, Windows, macOS) |

## Notes and limitations

- Shap-E is a research model. Meshes are often lumpy, may contain small stray
  pieces, and objects can miss parts (a car without wheels, for example). Generate a
  few variations with `xN` and keep the best; the cleanup tools handle the rest.
- After the first generation the worker keeps Shap-E in GPU memory (roughly 3 GB)
  until you close the app.
- Decimating to very low face counts can open holes in a mesh; ShapeForge warns
  when that happens.
- The AI rewrite only offers small chat models, so it fits in VRAM next to Shap-E.
- Negative prompts are not supported by Shap-E.

## Roadmap

- [x] v0.1 Basic text-to-3D
- [x] v0.2 Persistent engine and improved preview
- [x] v0.3 Batch Build
- [x] v0.4 Prompt enhancement
- [x] v0.5 Mesh cleanup and optimization
- [x] v0.6 Rules-based prompt cleanup, hover help, vertex-color preview
- [x] v0.7 Color-space fix, conservative AI rewrite, window sizing
- [x] v0.8 Engine setup assistant, quality presets, CPU support
- [x] v0.9 Packaged builds and GitHub Actions workflow
- [ ] Ideas: seed control, an experimental negative prompt, baking vertex colors
  into texture images, remembering settings between sessions, automated tests

## License

MIT. See [LICENSE](LICENSE). Shap-E is a separate project by OpenAI; see its
repository for its own license and terms.
