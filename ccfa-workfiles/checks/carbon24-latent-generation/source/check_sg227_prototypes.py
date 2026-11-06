"""Check whether same-space-group training structures cover SG227 test targets."""

from collections import Counter

import rootutils
from pymatgen.analysis.structure_matcher import StructureMatcher

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.carbon24_dataset import Carbon24Dataset


def main():
    train = Carbon24Dataset("data/carbon_24", "train")
    test = Carbon24Dataset("data/carbon_24", "test")
    train_items = [
        (index, record)
        for index, record in enumerate(train.records)
        if record["spacegroup"] == 227
    ]
    test_items = [
        (index, record)
        for index, record in enumerate(test.records)
        if record["spacegroup"] == 227
    ]
    by_n = {}
    for test_index, record in test_items:
        n = record["num_atoms"]
        candidates = [
            (index, item)
            for index, item in train_items
            if item["num_atoms"] == n
        ]
        by_n.setdefault(n, {"targets": 0, "matched": 0, "candidates": len(candidates)})
        by_n[n]["targets"] += 1
        target = test.get_structure(test_index)
        matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        hit = False
        for train_index, _ in candidates:
            try:
                if matcher.get_rms_dist(train.get_structure(train_index), target) is not None:
                    hit = True
                    break
            except Exception:
                continue
        by_n[n]["matched"] += int(hit)
    print(
        {
            "train_sg227": len(train_items),
            "test_sg227": len(test_items),
            "test_targets_with_same_N_train_prototype": sum(
                item["matched"] for item in by_n.values()
            ),
            "by_n": by_n,
            "train_n_counts": dict(
                sorted(Counter(item["num_atoms"] for _, item in train_items).items())
            ),
            "test_n_counts": dict(
                sorted(Counter(item["num_atoms"] for _, item in test_items).items())
            ),
        }
    )


if __name__ == "__main__":
    main()
