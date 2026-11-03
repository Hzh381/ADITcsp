"""Summarize supplied experiments without training or modifying source data."""

import argparse
import csv
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def read_sections(path):
    chunks = re.split(
        r"^# ==== (.*?) ====\s*$",
        path.read_text(encoding="utf-8-sig"),
        flags=re.MULTILINE,
    )
    return {
        chunks[i]: list(csv.DictReader(io.StringIO(chunks[i + 1].strip())))
        for i in range(1, len(chunks), 2)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir
    summaries = json.loads((root / "summary_all.json").read_text(encoding="utf-8-sig"))
    raw = json.loads((root / "per_target_all.json").read_text(encoding="utf-8-sig"))
    tables = {
        key: read_sections(root / filename)
        for key, filename in (
            ("train", "train_metrics_all.csv"),
            ("validation", "validation_metrics_all.csv"),
            ("history", "loss_history_all.csv"),
        )
    }
    result = {"summaries": summaries, "training": {}, "targets": {}}
    for name, validation in tables["validation"].items():
        train = tables["train"][name]
        history = tables["history"][name]
        assert len(train) == len(history)
        for a, b in zip(train, history):
            assert a["epoch"] == b["epoch"]
            assert abs(float(a["loss"]) - float(b["train_loss"])) < 1e-5
        key = "plain_loss" if "plain_loss" in train[-1] else "x_loss" if "x_loss" in train[-1] else "loss"
        vkey = "x_loss" if "x_loss" in validation[-1] else "loss"
        best = min(validation, key=lambda row: float(row[vkey]))
        result["training"][name] = {
            "epochs": int(train[-1]["epoch"]),
            "steps": int(train[-1]["step"]),
            "last_train_primary_loss": float(train[-1][key]),
            "last_validation_primary_loss": float(validation[-1][vkey]),
            "best_validation_epoch": int(best["epoch"]),
            "best_validation_loss": float(best[vkey]),
            "last_train": train[-1],
            "last_validation": validation[-1],
        }

    repository = Path(__file__).resolve().parents[4]
    with (repository / "data/carbon_24/test.csv").open(encoding="utf-8") as stream:
        labels = {r["material_id"]: r["spacegroup.number"] for r in csv.DictReader(stream)}
    c6_sets = {}
    for name, rows in raw.items():
        reconstruct = "posterior" in rows[0]
        groups = defaultdict(list)
        for row in rows:
            groups[row["target_num_atoms"]].append(row)
        target_stats = {"count": len(rows), "by_n": {}}
        total_hits = 0
        for n, group in sorted(groups.items()):
            candidates = [r["posterior"]["candidates"] if reconstruct else r["candidates"] for r in group]
            hits = [r for r, cs in zip(group, candidates) if any(c["rms"] is not None for c in cs)]
            total_hits += len(hits)
            stat = {
                "count": len(group),
                "matched": len(hits),
                "match_rate": len(hits) / len(group),
                "matched_target_spacegroups": dict(Counter(labels[r["id"]] for r in hits)),
                "prefix_hits": {
                    str(k): sum(any(c["rms"] is not None for c in cs[:k]) for cs in candidates)
                    for k in (1, 5, 10, 20)
                },
                "matching_candidates_per_target_histogram": dict(sorted(Counter(sum(c["rms"] is not None for c in cs) for cs in candidates).items())),
            }
            target_stats["by_n"][str(n)] = stat
        expected = summaries[name]["posterior_reconstruction_at_k"]["structure_match_at_k"] if reconstruct else summaries[name]["structure_match_at_k"]
        assert abs(total_hits / len(rows) - expected) < 1e-12
        target_stats["matched"] = total_hits
        if not reconstruct:
            c6_sets[name] = {r["id"] for r in groups[6] if r["structure_match_at_k"]}
            target_stats["prefix_hits"] = {
                str(k): sum(any(c["rms"] is not None for c in r["candidates"][:k]) for r in rows)
                for k in (1, 5, 10, 20)
            }
            target_stats["targets_with_rms_below"] = {
                str(threshold): sum(any(c["rms"] is not None and c["rms"] < threshold for c in r["candidates"]) for r in rows)
                for threshold in (0.1, 0.2, 0.3)
            }
        result["targets"][name] = target_stats
    hilo_sets = [s for name, s in c6_sets.items() if "standard_vae_" not in name]
    result["hilo_c6_shared_targets"] = {
        "models": len(hilo_sets),
        "intersection": len(set.intersection(*hilo_sets)),
        "union": len(set.union(*hilo_sets)),
    }
    output = Path(__file__).resolve().parents[1] / "log_analysis.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "runs_checked": len(summaries),
        "training_runs_checked": len(tables["train"]),
        "target_rows_checked": sum(len(rows) for rows in raw.values()),
        "hilo_c6_shared_targets": result["hilo_c6_shared_targets"],
        "summary_reconciliation": "all match rates agree with per-target records",
    }, indent=2))


if __name__ == "__main__":
    main()
