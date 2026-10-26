"""Crystal reconstruction metrics used by VAE training and evaluation."""

import os
from typing import Dict

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher

from src.eval.crystal import Crystal


class CrystalReconstructionEvaluator:
    def __init__(self, stol=0.5, angle_tol=10, ltol=0.3, device="cpu"):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_crys_list = []
        self.gt_crys_list = []
        self.device = device

    def append_pred_array(self, pred: Dict[str, np.ndarray]):
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: Dict[str, np.ndarray]):
        self.gt_arrays_list.append(gt)

    def clear(self):
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_crys_list = []
        self.gt_crys_list = []

    def _arrays_to_crystals(self, save=False, save_dir=""):
        self.pred_crys_list = [
            array_dict_to_crystal(x, save=save, save_dir_name=save_dir)
            for x in self.pred_arrays_list
        ]
        self.gt_crys_list = [
            array_dict_to_crystal(x, save=False) for x in self.gt_arrays_list
        ]

    def _get_metrics(self, pred, gt):
        try:
            rms = self.matcher.get_rms_dist(pred.structure, gt.structure)
            return float("inf") if rms is None else float(rms[0])
        except Exception:
            return float("inf")

    def get_metrics(self, save=False, save_dir=""):
        assert len(self.pred_arrays_list) == len(self.gt_arrays_list)
        self._arrays_to_crystals(save, save_dir)
        self.rms_dists = torch.tensor(
            [self._get_metrics(p, g) for p, g in zip(self.pred_crys_list, self.gt_crys_list)],
            device=self.device,
        )
        match_rate = (~torch.isinf(self.rms_dists)).long()
        finite = self.rms_dists[~torch.isinf(self.rms_dists)]
        if finite.numel() == 0:
            return {
                "match_rate": match_rate,
                "rms_dist": torch.full_like(self.rms_dists, 10.0),
            }
        return {"match_rate": match_rate, "rms_dist": finite}


def array_dict_to_crystal(x, save=False, save_dir_name=""):
    if np.all(50 < x["angles"]) and np.all(x["angles"] < 130):
        crystal = Crystal(x)
        if crystal.valid and save:
            os.makedirs(save_dir_name, exist_ok=True)
            crystal.structure.to(
                os.path.join(save_dir_name, f"crystal_{x['sample_idx']}.cif")
            )
        return crystal
    return Crystal(
        {
            "frac_coords": np.zeros_like(x["frac_coords"]),
            "atom_types": np.zeros_like(x["atom_types"]),
            "lengths": 100 * np.ones_like(x["lengths"]),
            "angles": np.ones_like(x["angles"]) * 90,
            "sample_idx": x["sample_idx"],
        }
    )
