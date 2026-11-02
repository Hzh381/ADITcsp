"""Print how many DiT blocks the denoiser has and how many transformer layers the VAE has."""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.ldm_module import LatentDiffusionLitModule  # noqa: E402


def count_transformer_layers(module):
    """Number of transformer layers of an encoder / decoder module."""
    transformer = getattr(module, "transformer", None)
    if transformer is None:
        return 0
    num_layers = getattr(transformer, "num_layers", None)
    if num_layers is not None:
        return int(num_layers)
    return len(getattr(transformer, "layers", []))


def count_denoiser_blocks(denoiser):
    """Number of DiT blocks, split into encoder / SI / decoder when available."""
    blocks = getattr(denoiser, "blocks", None)
    total = len(blocks) if blocks is not None else 0
    groups = [
        (name, getattr(denoiser, name))
        for name in ("en_in_blocks", "en_out_blocks", "side_blocks", "de_blocks")
        if getattr(denoiser, name, None) is not None
    ]
    return total, groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="LDM checkpoint path")
    parser.add_argument(
        "--vae-ckpt",
        default=None,
        help="override the VAE checkpoint stored in the LDM checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    kwargs = {"map_location": "cpu"}
    if args.vae_ckpt:
        kwargs["autoencoder_ckpt"] = args.vae_ckpt
    model = LatentDiffusionLitModule.load_from_checkpoint(args.ckpt, **kwargs)

    denoiser = model.denoiser
    total, groups = count_denoiser_blocks(denoiser)
    print(f"denoiser: {type(denoiser).__name__}, DiT blocks = {total}")
    for name, blocks in groups:
        print(f"  {name:<15} {len(blocks)}")
    if hasattr(denoiser, "d_model"):
        print(f"  d_model={denoiser.d_model}, nhead={getattr(denoiser, 'nhead', '-')}")

    vae = model.autoencoder
    encoder_layers = count_transformer_layers(vae.encoder)
    decoder_layers = count_transformer_layers(vae.decoder)
    print(
        f"vae encoder: {type(vae.encoder).__name__}, "
        f"transformer layers = {encoder_layers}"
    )
    print(
        f"vae decoder: {type(vae.decoder).__name__}, "
        f"transformer layers = {decoder_layers}"
    )


if __name__ == "__main__":
    main()
