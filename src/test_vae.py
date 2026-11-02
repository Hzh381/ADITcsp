"""Evaluate composition-conditioned VAE reconstruction on crystal data.

This entry point measures the reconstruction ceiling of a frozen VAE before
stage-two latent generation. Each ground-truth crystal is encoded once, then
decoded using its true composition in two ways:

1. deterministic posterior mean (one candidate per target);
2. repeated posterior sampling (``--candidates`` candidates per target).

The matching protocol is the same as ``test_csp.py``:
``StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10)``.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import rootutils
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch
from tqdm import tqdm

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.composition_utils import composition_from_batch  # noqa: E402
from src.data.crystal_datamodule import (  # noqa: E402
    SUPPORTED_DATASETS,
    build_crystal_dataset,
)
from src.eval.crystal import structure_validity  # noqa: E402
from src.models.vae_module import VariationalAutoencoderLitModule  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate posterior-mean and posterior-sampling VAE reconstruction."
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="carbon_24")
    parser.add_argument("--ckpt", required=True, help="VAE checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="posterior samples per target for Reconstruct@K",
    )
    parser.add_argument("--limit", type=int, default=50, help="0 evaluates the full split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--save-cifs",
        action="store_true",
        help="write every structurally valid reconstruction as a CIF",
    )
    args = parser.parse_args()
    if args.candidates < 1:
        parser.error("--candidates must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def _finite_mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _empty_candidate_stats():
    return {
        "candidates": 0,
        "composition_ok": 0,
        "struct_valid": 0,
        "overall_valid": 0,
        "matched_targets": 0,
        "best_rms": [],
    }


def _finalize_candidate_stats(stats, num_targets, samples_per_target):
    matched_targets = sum(math.isfinite(value) for value in stats["best_rms"])
    candidate_count = max(stats["candidates"], 1)
    return {
        "samples_per_target": samples_per_target,
        "candidate_count": stats["candidates"],
        "composition_match_rate": stats["composition_ok"] / candidate_count,
        "struct_valid_rate": stats["struct_valid"] / candidate_count,
        "overall_valid_rate": stats["overall_valid"] / candidate_count,
        "structure_match_at_k": matched_targets / max(num_targets, 1),
        "matched_targets": matched_targets,
        "unmatched_targets": num_targets - matched_targets,
        "mean_best_rms": _finite_mean(stats["best_rms"]),
    }


def _evaluate_candidate(
    *,
    out,
    start,
    end,
    graph_idx,
    target_counts,
    matcher,
    num_elements,
    data,
    material_id,
    index,
    label,
    save_cifs,
    cifs_dir,
):
    atom_types = out["atom_types"][start:end].argmax(-1).clone()
    atom_types[atom_types == 0] = 1
    frac = out["frac_coords"][start:end]
    lengths = out["lengths"][graph_idx] * float(end - start) ** (1 / 3)
    angles = torch.rad2deg(out["angles"][graph_idx])

    counts = torch.bincount(
        atom_types.detach().cpu(),
        minlength=num_elements,
    )
    composition_ok = bool(torch.equal(counts, target_counts.long()))

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
                lattice=Lattice.from_parameters(
                    *lengths_np.tolist(),
                    *angles_np.tolist(),
                ),
                species=atom_types.detach().cpu().numpy(),
                coords=frac_np,
                coords_are_cartesian=False,
            )
            struct_ok = bool(structure_validity(structure))
        except (ValueError, np.linalg.LinAlgError):
            structure = None

    try:
        rms = (
            matcher.get_rms_dist(structure, data)
            if structure is not None
            else None
        )
        rms = float("inf") if rms is None else float(rms[0])
    except Exception:
        rms = float("inf")

    cif_path = None
    if struct_ok and save_cifs:
        cif_path = cifs_dir / (
            f"{index:05d}_{_safe_filename(material_id)}_{label}.cif"
        )
        structure.to(str(cif_path), fmt="cif")

    return {
        "candidate": label,
        "num_atoms": int(end - start),
        "composition_ok": composition_ok,
        "struct_valid": struct_ok,
        "rms": None if not math.isfinite(rms) else rms,
        "cif": str(cif_path) if cif_path is not None else None,
    }


def _update_stats(stats, candidate):
    stats["candidates"] += 1
    stats["composition_ok"] += int(candidate["composition_ok"])
    stats["struct_valid"] += int(candidate["struct_valid"])
    stats["overall_valid"] += int(
        candidate["composition_ok"] and candidate["struct_valid"]
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output = Path(
        args.output
        or ROOT / f"runs/{args.dataset}_vae_{args.split}_reconstruction"
    )
    cifs_dir = output / "generated"
    if args.save_cifs:
        cifs_dir.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)

    model = VariationalAutoencoderLitModule.load_from_checkpoint(
        args.ckpt,
        map_location="cpu",
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if not model.composition_conditioning:
        raise ValueError(
            "This evaluation requires a composition-conditioned VAE checkpoint."
        )

    dataset = build_crystal_dataset(
        ROOT / "data" / args.dataset,
        args.dataset,
        args.split,
        num_elements=model.num_elements,
    )
    total = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    mode_stats = _empty_candidate_stats()
    sample_stats = _empty_candidate_stats()
    records = []

    with torch.inference_mode(), tqdm(total=total, desc="VAE reconstruction") as progress:
        for batch_start in range(0, total, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total)
            batch_indices = list(range(batch_start, batch_end))
            data_list = [dataset[index] for index in batch_indices]
            batch = Batch.from_data_list(data_list).to(device)
            num_graphs = len(data_list)

            encoded_base = model.encode(batch)
            composition = composition_from_batch(batch, model.num_elements)

            mode_encoded = dict(encoded_base)
            mode_encoded["x"] = encoded_base["posterior"].mode()
            mode_encoded["composition"] = composition
            mode_out = model.decode(mode_encoded)

            sample_outs = []
            for _ in range(args.candidates):
                sample_encoded = dict(encoded_base)
                sample_encoded["x"] = encoded_base["posterior"].sample()
                sample_encoded["composition"] = composition
                sample_outs.append(model.decode(sample_encoded))

            start = 0
            for graph_idx, (index, data) in enumerate(zip(batch_indices, data_list)):
                end = start + int(data.num_atoms.item())
                target_structure = dataset.get_structure(index)
                target_counts = composition[graph_idx].detach().cpu()
                record = {
                    "index": index,
                    "id": data.id,
                    "target_num_atoms": int(data.num_atoms.item()),
                }
                records.append(record)

                mode_candidate = _evaluate_candidate(
                    out=mode_out,
                    start=start,
                    end=end,
                    graph_idx=graph_idx,
                    target_counts=target_counts,
                    matcher=matcher,
                    num_elements=model.num_elements,
                    data=target_structure,
                    material_id=data.id,
                    index=index,
                    label="mode",
                    save_cifs=args.save_cifs,
                    cifs_dir=cifs_dir,
                )
                _update_stats(mode_stats, mode_candidate)
                mode_best_rms = (
                    mode_candidate["rms"]
                    if mode_candidate["rms"] is not None
                    else float("inf")
                )
                mode_stats["best_rms"].append(mode_best_rms)
                record["mode"] = {
                    **mode_candidate,
                    "structure_match": math.isfinite(mode_best_rms),
                }

                sample_candidates = []
                sample_best_rms = float("inf")
                for sample_index, sample_out in enumerate(sample_outs):
                    candidate = _evaluate_candidate(
                        out=sample_out,
                        start=start,
                        end=end,
                        graph_idx=graph_idx,
                        target_counts=target_counts,
                        matcher=matcher,
                        num_elements=model.num_elements,
                        data=target_structure,
                        material_id=data.id,
                        index=index,
                        label=f"sample_{sample_index}",
                        save_cifs=args.save_cifs,
                        cifs_dir=cifs_dir,
                    )
                    _update_stats(sample_stats, candidate)
                    sample_candidates.append(candidate)
                    candidate_rms = (
                        candidate["rms"]
                        if candidate["rms"] is not None
                        else float("inf")
                    )
                    sample_best_rms = min(sample_best_rms, candidate_rms)

                sample_stats["best_rms"].append(sample_best_rms)
                record["posterior"] = {
                    "structure_match_at_k": math.isfinite(sample_best_rms),
                    "best_rms": (
                        None
                        if not math.isfinite(sample_best_rms)
                        else sample_best_rms
                    ),
                    "candidates": sample_candidates,
                }
                start = end

            progress.update(num_graphs)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(args.ckpt),
        "num_targets": total,
        "seed": args.seed,
        "composition_conditioning": True,
        "structure_matcher": {
            "stol": 0.5,
            "ltol": 0.3,
            "angle_tol": 10,
        },
        "mode_reconstruction": _finalize_candidate_stats(
            mode_stats,
            total,
            1,
        ),
        "posterior_reconstruction_at_k": _finalize_candidate_stats(
            sample_stats,
            total,
            args.candidates,
        ),
        "rms_note": (
            "mean_best_rms is the mean of the best finite normalized RMSD only "
            "for targets with at least one match; unmatched targets are excluded."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output / "per_target.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
