"""Compare K8 standard-DiT and K16 CoDeMDT per-target CSP records."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
INPUTS = {
    "k8_standard": Path("D:/HZH/DownLoad/per_target (2).json"),
    "k16_codemdt": Path("D:/HZH/DownLoad/per_target (1).json"),
}


def finite(values):
    return [float(value) for value in values if value is not None]


def target_analysis(rows, labels):
    by_n = {}
    by_sg = {}
    hit_histogram = Counter()
    prefix_hits = {prefix: 0 for prefix in (1, 5, 10, 20)}
    candidate_rms = []
    candidate_valid = 0
    candidate_total = 0
    matched_rms = []
    hit_counts = []

    for row in rows:
        candidates = row["candidates"]
        hits = [c for c in candidates if c.get("rms") is not None]
        hit_count = len(hits)
        hit_histogram[hit_count] += 1
        hit_counts.append(hit_count)
        candidate_total += len(candidates)
        candidate_valid += sum(bool(c.get("struct_valid")) for c in candidates)
        candidate_rms.extend(float(c["rms"]) for c in hits)
        for prefix in prefix_hits:
            if any(c.get("rms") is not None for c in candidates[:prefix]):
                prefix_hits[prefix] += 1
        if hits:
            matched_rms.append(min(float(c["rms"]) for c in hits))

        n = int(row["target_num_atoms"])
        item = by_n.setdefault(
            n,
            {"targets": 0, "matches": 0, "best_rms": [], "hit_histogram": Counter()},
        )
        item["targets"] += 1
        item["matches"] += int(bool(hits))
        item["hit_histogram"][hit_count] += 1
        if hits:
            item["best_rms"].append(min(float(c["rms"]) for c in hits))

        sg = str(labels.get(row["id"], "unknown"))
        item = by_sg.setdefault(
            sg,
            {"targets": 0, "matches": 0, "best_rms": []},
        )
        item["targets"] += 1
        item["matches"] += int(bool(hits))
        if hits:
            item["best_rms"].append(min(float(c["rms"]) for c in hits))

    def finish(item):
        rms = item.pop("best_rms")
        return {
            **item,
            "match_rate": item["matches"] / max(item["targets"], 1),
            "mean_best_rms_matched": statistics.mean(rms) if rms else None,
        }

    candidate_rms.sort()
    matched_rms.sort()

    def q(values, fraction):
        if not values:
            return None
        return values[min(len(values) - 1, int((len(values) - 1) * fraction))]

    return {
        "targets": len(rows),
        "matches": sum(int(row["structure_match_at_k"]) for row in rows),
        "match_rate": sum(int(row["structure_match_at_k"]) for row in rows) / len(rows),
        "prefix_hits": prefix_hits,
        "hit_count_histogram": dict(sorted(hit_histogram.items())),
        "targets_all_20_candidates_match": hit_histogram[20],
        "targets_exactly_one_match": hit_histogram[1],
        "mean_hit_count": statistics.mean(hit_counts) if hit_counts else None,
        "candidate_total": candidate_total,
        "candidate_valid_rate": candidate_valid / max(candidate_total, 1),
        "candidate_rms_quantiles": {
            "p25": q(candidate_rms, 0.25),
            "median": q(candidate_rms, 0.5),
            "p75": q(candidate_rms, 0.75),
            "min": candidate_rms[0] if candidate_rms else None,
            "max": candidate_rms[-1] if candidate_rms else None,
        },
        "by_n": {str(k): finish(v) for k, v in sorted(by_n.items())},
        "by_spacegroup": {
            k: finish(v)
            for k, v in sorted(by_sg.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else 10**9)
        },
    }


def main():
    records = {
        name: json.loads(path.read_text(encoding="utf-8-sig"))
        for name, path in INPUTS.items()
    }
    with (ROOT / "data/carbon_24/test.csv").open(encoding="utf-8") as handle:
        labels = {
            row["material_id"]: row["spacegroup.number"]
            for row in csv.DictReader(handle)
        }
    result = {
        "source_files": {name: str(path) for name, path in INPUTS.items()},
        "runs": {},
    }
    for name, rows in records.items():
        result["runs"][name] = target_analysis(rows, labels)

    a = {row["id"]: row for row in records["k8_standard"]}
    b = {row["id"]: row for row in records["k16_codemdt"]}
    common_ids = set(a) & set(b)
    a_hits = {key for key in common_ids if a[key]["structure_match_at_k"]}
    b_hits = {key for key in common_ids if b[key]["structure_match_at_k"]}
    both = a_hits & b_hits

    def best_rms(rows, keys):
        values = []
        for key in keys:
            candidate_values = [
                float(c["rms"])
                for c in rows[key]["candidates"]
                if c.get("rms") is not None
            ]
            if candidate_values:
                values.append(min(candidate_values))
        return statistics.mean(values) if values else None

    result["comparison"] = {
        "common_ids": len(common_ids),
        "both_match": len(both),
        "k8_only_match": len(a_hits - b_hits),
        "k16_only_match": len(b_hits - a_hits),
        "neither_match": len(common_ids - a_hits - b_hits),
        "k8_match_rate_common": len(a_hits) / max(len(common_ids), 1),
        "k16_match_rate_common": len(b_hits) / max(len(common_ids), 1),
        "rms_on_common_hits": {
            "k8_mean": best_rms(a, both),
            "k16_mean": best_rms(b, both),
        },
    }
    result["comparison"]["by_n"] = {}
    all_n = set(result["runs"]["k8_standard"]["by_n"]) | set(
        result["runs"]["k16_codemdt"]["by_n"]
    )
    for n in sorted(all_n, key=int):
        left = result["runs"]["k8_standard"]["by_n"].get(n, {})
        right = result["runs"]["k16_codemdt"]["by_n"].get(n, {})
        result["comparison"]["by_n"][n] = {
            "k8_match_rate": left.get("match_rate"),
            "k16_match_rate": right.get("match_rate"),
            "delta": (
                right.get("match_rate") - left.get("match_rate")
                if right.get("match_rate") is not None
                and left.get("match_rate") is not None
                else None
            ),
        }
    result["comparison"]["by_spacegroup"] = {}
    all_sg = set(result["runs"]["k8_standard"]["by_spacegroup"]) | set(
        result["runs"]["k16_codemdt"]["by_spacegroup"]
    )
    for sg in sorted(all_sg, key=lambda value: int(value) if value.isdigit() else 10**9):
        left = result["runs"]["k8_standard"]["by_spacegroup"].get(sg, {})
        right = result["runs"]["k16_codemdt"]["by_spacegroup"].get(sg, {})
        result["comparison"]["by_spacegroup"][sg] = {
            "k8_match_rate": left.get("match_rate"),
            "k16_match_rate": right.get("match_rate"),
            "delta": (
                right.get("match_rate") - left.get("match_rate")
                if right.get("match_rate") is not None
                and left.get("match_rate") is not None
                else None
            ),
        }
    output = Path(__file__).resolve().parents[1] / "two_run_analysis.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": {
                    name: {
                        "targets": value["targets"],
                        "matches": value["matches"],
                        "match_rate": value["match_rate"],
                        "all20": value["targets_all_20_candidates_match"],
                        "exactly1": value["targets_exactly_one_match"],
                        "prefix_hits": value["prefix_hits"],
                    }
                    for name, value in result["runs"].items()
                },
                "comparison": result["comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
