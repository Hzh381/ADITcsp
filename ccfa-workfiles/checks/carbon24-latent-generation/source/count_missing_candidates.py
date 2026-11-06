"""Count train candidates with the same atom count and space group for common misses."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.carbon24_dataset import Carbon24Dataset  # noqa: E402


def main():
    paths = [
        Path("D:/HZH/DownLoad/per_target (1).json"),
        Path("D:/HZH/DownLoad/per_target (2).json"),
    ]
    runs = [json.loads(path.read_text(encoding="utf-8-sig")) for path in paths]
    misses = [
        {row["id"] for row in rows if not row["structure_match_at_k"]}
        for rows in runs
    ]
    common = misses[0] & misses[1]
    train = Carbon24Dataset("data/carbon_24", "train")
    test = Carbon24Dataset("data/carbon_24", "test")
    train_counts = Counter(
        (record["num_atoms"], record["spacegroup"]) for record in train.records
    )
    test_by_id = {record["id"]: record for record in test.records}
    no_candidate = []
    by_n = Counter()
    by_sg = Counter()
    for material_id in common:
        record = test_by_id[material_id]
        key = (record["num_atoms"], record["spacegroup"])
        if train_counts[key] == 0:
            no_candidate.append(material_id)
        else:
            by_n[record["num_atoms"]] += 1
            by_sg[record["spacegroup"]] += 1
    print(
        {
            "common_misses": len(common),
            "with_same_n_sg_train_candidate": len(common) - len(no_candidate),
            "without_same_n_sg_train_candidate": len(no_candidate),
            "missing_ids": no_candidate,
            "covered_by_n": dict(sorted(by_n.items())),
            "covered_by_sg": dict(sorted(by_sg.items())),
        }
    )


if __name__ == "__main__":
    main()
