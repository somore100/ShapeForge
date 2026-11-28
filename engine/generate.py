import argparse
from pathlib import Path

import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="a stylized low-poly cartoon penguin, standing upright"
    )
    parser.add_argument("--output", default="generated")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading Shap-E models...")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    print("Generating:", args.prompt)

    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=15.0,
        model_kwargs={"texts": [args.prompt]},
        progress=True,
        clip_denoised=True,
        use_fp16=(device.type == "cuda"),
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    print("Converting generated shape into a mesh...")
    mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    try:  # Shap-E colours are linear; OBJ/PLY viewers expect sRGB
        import pathlib as _pl, sys as _sys
        _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
        from colorspace import mesh_colors_to_srgb
        mesh_colors_to_srgb(mesh)
    except Exception as error:  # never block a generation over colours
        print(f"Colour conversion skipped: {error}")

    obj_path = output_dir / "generated.obj"
    ply_path = output_dir / "generated.ply"

    with open(obj_path, "w") as f:
        mesh.write_obj(f)

    with open(ply_path, "wb") as f:
        mesh.write_ply(f)

    print("Finished!")
    print("OBJ:", obj_path)
    print("PLY:", ply_path)


if __name__ == "__main__":
    main()
