"""Lightweight crystal validity utilities used by CSP evaluation."""

import itertools
from collections import Counter

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


class Crystal:
    """Crystal structure plus structure/composition validity flags."""

    def __init__(self, crys_array_dict):
        self.frac_coords = np.asarray(crys_array_dict["frac_coords"])
        self.atom_types = np.asarray(crys_array_dict["atom_types"])
        self.sample_idx = crys_array_dict["sample_idx"]
        self.lengths = np.asarray(crys_array_dict["lengths"]).squeeze()
        self.angles = np.asarray(crys_array_dict["angles"]).squeeze()
        self.get_structure()
        if self.constructed:
            self.get_composition()
            self.get_validity()
        else:
            self.valid = self.comp_valid = self.struct_valid = False

    def get_structure(self):
        self.constructed = False
        self.invalid_reason = ""
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or np.isnan(self.frac_coords).any():
            self.invalid_reason = "nan_value"
            return
        if (self.lengths <= 0).any():
            self.invalid_reason = "non_positive_lattice"
            return
        if (self.atom_types < 1).any() or (self.atom_types > 104).any():
            self.invalid_reason = "atom_types_out_of_range"
            return
        try:
            self.structure = Structure(
                lattice=Lattice.from_parameters(
                    *(self.lengths.tolist() + self.angles.tolist())
                ),
                species=self.atom_types,
                coords=self.frac_coords,
                coords_are_cartesian=False,
            )
            self.constructed = bool(self.structure.volume >= 0.1)
            if not self.constructed:
                self.invalid_reason = "unrealistically_small_lattice"
        except Exception:
            self.invalid_reason = "construction_error"

    def get_composition(self):
        counter = Counter(self.atom_types.tolist())
        self.elems = tuple(sorted(counter))
        counts = np.array([counter[e] for e in self.elems])
        counts = counts // np.gcd.reduce(counts)
        self.comps = tuple(counts.astype(int).tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        self.struct_valid = structure_validity(self.structure)
        self.valid = self.comp_valid and self.struct_valid

    def __repr__(self):
        return f"Crystal(valid={self.valid}, atoms={self.atom_types.tolist()}, idx={self.sample_idx})"


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    symbols = tuple(chemical_symbols[elem] for elem in comp)
    if len(set(symbols)) == 1:
        return True

    # Carbon-24 never reaches this branch. Keep the general SMACT check for
    # the existing multicomponent HiLo experiments.
    import smact
    from smact.screening import pauling_test

    dictionary = smact.element_dictionary(symbols)
    elements = [dictionary[s] for s in symbols]
    if include_alloys and all(s in smact.metals for s in symbols):
        return True
    oxidation_states = [e.oxidation_states for e in elements]
    electronegativities = [e.pauling_eneg for e in elements]
    combinations = int(np.prod([len(states) for states in oxidation_states]))
    if combinations > 1_000_000:
        return False
    for states in itertools.product(*oxidation_states):
        neutral, _ = smact.neutral_ratios(
            states, stoichs=[(c,) for c in count], threshold=max(count)
        )
        if not neutral:
            continue
        if use_pauling_test:
            try:
                if pauling_test(states, electronegativities):
                    return True
            except TypeError:
                return True
        else:
            return True
    return False


def structure_validity(crystal, cutoff=0.5):
    matrix = np.asarray(crystal.lattice.matrix, dtype=float)
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) <= 0.0:
        return False
    if not np.isfinite(crystal.volume) or crystal.volume < 0.1:
        return False
    try:
        distances = crystal.distance_matrix
    except (ValueError, np.linalg.LinAlgError):
        return False
    distances = distances + np.diag(np.full(len(crystal), cutoff + 10.0))
    return bool(distances.min() >= cutoff and crystal.volume >= 0.1)


chemical_symbols = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
