"""Search train-only structural prototypes for the 363 common misses."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import rootutils
from pymatgen.analysis.structure_matcher import StructureMatcher

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.carbon24_dataset import Carbon24Dataset  # noqa: E402


PER_TARGET = [
    Path("D:/HZH/DownLoad/per_target (1).json"),
    Path("D:/HZH/DownLoad/per_target (2).json"),
]


def main():
    records = [
        json.loads(path.read_text(encoding="utf-8-sig"))
        for path in PER_TARGET
    ]
    miss_sets = [
        {row["id"] for row in rows if not row["structure_match_at_k"]}
        for rows in records
    ]
    common_misses = miss_sets[0] & miss_sets[1]

    train = Carbon24Dataset("data/carbon_24", "train")
    test = Carbon24Dataset("data/carbon_24", "test")
    train_items = list(enumerate(train.records))
    test_by_id = {
        record["id"]: (index, record)
        for index, record in enumerate(test.records)
    }
    by_n = {}
    by_sg = {}
    unmatched = []

    for material_id in sorted(common_misses):
        test_index, target_record = test_by_id[material_id]
        num_atoms = target_record["num_atoms"]
        spacegroup = target_record["spacegroup"]
        target = test.get_structure(test_index)
        same_n = [
            (index, item)
            for index, item in train_items
            if item["num_atoms"] == num_atoms
        ]
        same_n_sg = [
            (index, item)
            for index, item in same_n
            if item["spacegroup"] == spacegroup
        ]
        matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        sg_match = False
        for index, _ in same_n_sg:
            try:
                if matcher.get_rms_dist(train.get_structure(index), target) is not None:
                    sg_match = True
                    break
            except Exception:
                continue
        item = by_n.setdefault(
            num_atoms,
            {
                "targets": 0,
                "same_n_candidates": 0,
                "same_n_sg_candidates": 0,
                "same_n_sg_match": 0,
            },
        )
        item["targets"] += 1
        item["same_n_candidates"] += len(same_n)
        item["same_n_sg_candidates"] += len(same_n_sg)
        item["same_n_sg_match"] += int(sg_match)
        sg_item = by_sg.setdefault(
            str(spacegroup),
            {
                "targets": 0,
                "same_n_candidates": 0,
                "same_n_sg_candidates": 0,
                "same_n_sg_match": 0,
            },
        )
        sg_item["targets"] += 1
        sg_item["same_n_candidates"] += len(same_n)
        sg_item["same_n_sg_candidates"] += len(same_n_sg)
        sg_item["same_n_sg_match"] += int(sg_match)
        if not sg_match:
            unmatched.append(material_id)

    result = {
        "common_misses": len(common_misses),
        "same_n_sg_train_match": sum(
            item["same_n_sg_match"] for item in by_n.values()
        ),
        "no_same_n_sg_train_match": len(unmatched),
        "by_n": {str(key): value for key, value in sorted(by_n.items())},
        "by_spacegroup": dict(
            sorted(by_sg.items(), key=lambda pair: int(pair[0]))
        ),
        "unmatched_ids": unmatched,
    }
    output = Path(__file__).resolve().parents[1] / "missing_prototypes.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
