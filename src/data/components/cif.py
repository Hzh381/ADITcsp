"""Minimal CIF parsing for Carbon-24; no pyxtal or graph construction."""

import numpy as np
from pymatgen.core import Lattice, Structure


def cif_to_arrays(cif: str) -> dict:
    structure = Structure.from_str(cif, fmt="cif").get_reduced_structure()
    lattice = Lattice.from_parameters(*structure.lattice.parameters)
    return {
        "atom_types": np.array([site.specie.Z for site in structure], dtype=np.int64),
        "frac_coords": np.asarray(structure.frac_coords, dtype=np.float64),
        "lengths": np.asarray(lattice.abc, dtype=np.float64),
        "angles": np.asarray(lattice.angles, dtype=np.float64),
        "cell": np.asarray(lattice.matrix, dtype=np.float64),
        "num_atoms": len(structure),
    }
