"""ShapeForge persistent engine.

Loads Shap-E once, then generates models for JSON-line jobs read from stdin.

Protocol (one JSON object per line):

  stdin  -> {"id": 1, "prompt": "a low-poly penguin", "output": "/path/to/dir"}
  stdin  -> {"cmd": "quit"}

  stdout <- {"event": "log",   "message": "..."}
  stdout <- {"event": "ready"}
  stdout <- {"event": "start", "id": 1}
  stdout <- {"event": "done",  "id": 1, "ok": true,  "seconds": 31.2,
             "obj": "...", "ply": "..."}
  stdout <- {"event": "done",  "id": 1, "ok": false, "error": "..."}

Only protocol JSON goes to stdout. Anything else (tqdm progress bars,
library prints) is sent to stderr so it can never corrupt the protocol.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Keep the real stdout for the protocol; send all stray prints to stderr.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


START_DIR = Path.cwd()
SHAPE_E_HOME = Path(
    os.environ.get("SHAPE_E_HOME", str(Path.home() / "shap-e"))
).expanduser()
if str(SHAPE_E_HOME) not in sys.path:
    sys.path.insert(0, str(SHAPE_E_HOME))


def emit(**event):
    _PROTOCOL.write(json.dumps(event) + "\n")
    _PROTOCOL.flush()


def log(message):
    emit(event="log", message=str(message))


def load_engine():
    import torch
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_config, load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    log("Loading Shap-E models (one time)...")
    started = time.time()
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    log(f"Models loaded in {time.time() - started:.1f}s")

    return torch, device, xm, model, diffusion


def generate(engine, prompt, output_dir):
    torch, device, xm, model, diffusion = engine

    from shap_e.diffusion.sample import sample_latents
    from shap_e.util.notebooks import decode_latent_mesh

    output_dir = Path(output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = START_DIR / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=15.0,
        model_kwargs={"texts": [prompt]},
        progress=True,
        clip_denoised=True,
        use_fp16=(device.type == "cuda"),
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()

    obj_path = output_dir / "generated.obj"
    ply_path = output_dir / "generated.ply"

    with open(obj_path, "w") as f:
        mesh.write_obj(f)
    with open(ply_path, "wb") as f:
        mesh.write_ply(f)

    return obj_path, ply_path


def main():
    if not SHAPE_E_HOME.is_dir():
        emit(event="fatal", error=f"SHAPE_E_HOME not found: {SHAPE_E_HOME}")
        return 1
    os.chdir(SHAPE_E_HOME)
    cache = SHAPE_E_HOME / "shap_e_model_cache"
    log(f"Model cache: {cache}")
    for name in ("transmitter.pt", "text_cond.pt", "ViT-L-14.pt"):
        if not (cache / name).exists():
            log(f"WARNING: {name} not in cache; it will be downloaded.")

    try:
        engine = load_engine()
    except Exception:
        emit(event="fatal", error=traceback.format_exc())
        return 1

    emit(event="ready")

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            job = json.loads(raw)
        except json.JSONDecodeError as error:
            log(f"Ignoring invalid job line: {error}")
            continue

        if job.get("cmd") == "quit":
            break

        job_id = job.get("id")
        prompt = job.get("prompt")
        output = job.get("output")

        if not prompt or not output:
            emit(event="done", id=job_id, ok=False,
                 error="Job needs 'prompt' and 'output'.")
            continue

        emit(event="start", id=job_id)
        started = time.time()

        try:
            obj_path, ply_path = generate(engine, prompt, output)
            emit(event="done", id=job_id, ok=True,
                 seconds=round(time.time() - started, 1),
                 obj=str(obj_path), ply=str(ply_path))
        except Exception as error:
            traceback.print_exc()
            emit(event="done", id=job_id, ok=False, error=str(error))
        finally:
            # Release cached GPU memory between jobs; models stay loaded.
            try:
                engine[0].cuda.empty_cache()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
