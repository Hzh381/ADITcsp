"""Evaluate composition-conditioned crystal structure prediction checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import rootutils
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.crystal_datamodule import (  # noqa: E402
    SUPPORTED_DATASETS,
    build_crystal_dataset,
)
from src.eval.crystal import structure_validity  # noqa: E402
from src.models.ldm_module import LatentDiffusionLitModule  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="carbon_24")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output = Path(args.output or ROOT / f"runs/{args.dataset}_test")
    cifs = output / "generated"
    cifs.mkdir(parents=True, exist_ok=True)

    kwargs = {"map_location": "cpu"}
    if args.vae_ckpt:
        kwargs["autoencoder_ckpt"] = args.vae_ckpt
    model = LatentDiffusionLitModule.load_from_checkpoint(args.ckpt, **kwargs)
    model.to(device)
    model.eval()
    model.autoencoder.eval()
    model._device = device
    model.interpolant.num_timesteps = 100

    dataset = build_crystal_dataset(
        ROOT / "data" / args.dataset,
        args.dataset,
        "test",
    )
    total = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    rows = []
    comp_hits = match_hits = 0
    comp_candidates = valid_candidates = overall_candidates = 0
    rms_values = []

    for index in tqdm(range(total), desc="CSP"):
        data = dataset[index]
        target_counts = data.composition.float().cpu()
        target = dataset.get_structure(index)
        comp = target_counts.repeat(args.candidates, 1)
        out, batch, _ = model.sample_and_decode(comp, cfg_scale=2.0)

        best_rms = float("inf")
        any_comp = any_match = False
        start = 0
        candidates = []
        for i, num_atom in enumerate(batch["num_atoms"].tolist()):
            atom_types = out["atom_types"][start : start + num_atom].argmax(-1).clone()
            atom_types[atom_types == 0] = 1
            frac = out["frac_coords"][start : start + num_atom]
            lengths = out["lengths"][i] * float(num_atom) ** (1 / 3)
            angles = torch.rad2deg(out["angles"][i])
            counts = torch.bincount(atom_types.detach().cpu(), minlength=100)
            comp_ok = bool(torch.equal(counts, target_counts[0].long()))
            lengths_np = lengths.detach().cpu().numpy()
            angles_np = angles.detach().cpu().numpy()
            frac_np = frac.detach().cpu().numpy()
            structure = None
            struct_ok = False
            if (
                np.isfinite(lengths_np).all()
                and np.isfinite(angles_np).all()
                and np.isfinite(frac_np).all()
                and (lengths_np > 0).all()
                and (angles_np > 0).all()
                and (angles_np < 180).all()
            ):
                try:
                    structure = Structure(
                        lattice=Lattice.from_parameters(*lengths_np.tolist(), *angles_np.tolist()),
                        species=atom_types.detach().cpu().numpy(),
                        coords=frac_np,
                        coords_are_cartesian=False,
                    )
                    struct_ok = bool(structure_validity(structure))
                except (ValueError, np.linalg.LinAlgError):
                    structure = None
            try:
                rms = (
                    matcher.get_rms_dist(structure, target)
                    if structure is not None
                    else None
                )
                rms = float("inf") if rms is None else float(rms[0])
            except Exception:
                rms = float("inf")
            matched = bool(np.isfinite(rms))
            any_comp |= comp_ok
            any_match |= matched
            best_rms = min(best_rms, rms)
            comp_candidates += int(comp_ok)
            valid_candidates += int(struct_ok)
            overall_candidates += int(comp_ok and struct_ok)
            cif_path = cifs / f"{data.id}_{i}.cif"
            if struct_ok:
                structure.to(str(cif_path), fmt="cif")
            candidates.append(
                {
                    "candidate": i,
                    "num_atoms": int(num_atom),
                    "composition_ok": comp_ok,
                    "struct_valid": struct_ok,
                    "rms": None if not np.isfinite(rms) else rms,
                    "cif": str(cif_path) if struct_ok else None,
                }
            )
            start += num_atom

        comp_hits += int(any_comp)
        match_hits += int(any_match)
        rms_values.append(best_rms)
        rows.append(
            {
                "id": data.id,
                "target_num_atoms": int(data.num_atoms.item()),
                "composition_success_at_k": any_comp,
                "structure_match_at_k": any_match,
                "best_rms": None if not np.isfinite(best_rms) else best_rms,
                "candidates": candidates,
            }
        )

    finite = [x for x in rms_values if np.isfinite(x)]
    summary = {
        "dataset": args.dataset,
        "num_targets": total,
        "candidates_per_target": args.candidates,
        "composition_match_rate": comp_candidates / max(total * args.candidates, 1),
        "composition_success_at_k": comp_hits / max(total, 1),
        "struct_valid_rate": valid_candidates / max(total * args.candidates, 1),
        "overall_valid_rate": overall_candidates / max(total * args.candidates, 1),
        "structure_match_at_k": match_hits / max(total, 1),
        "mean_best_rms": float(np.mean(finite)) if finite else None,
        "cfg": 2.0,
        "steps": 100,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "per_target.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
