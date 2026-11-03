"""Aggregate the supplied Carbon-24 logs and per-target evaluation records."""

import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


INPUT = Path(r"D:\HZH\DownLoad")
OUTPUT = Path(__file__).resolve().parents[1] / "supplied_results.json"


def split_sections(path):
    sections = defaultdict(list)
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# ===="):
            current = line.removeprefix("# ====").removesuffix("====").strip()
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def parse_csv_sections(path):
    parsed = {}
    for name, text in split_sections(path).items():
        if not text:
            parsed[name] = []
            continue
        rows = list(csv.DictReader(io.StringIO(text)))
        clean = []
        for row in rows:
            converted = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value) if value not in (None, "") else None
                except ValueError:
                    converted[key] = value
            clean.append(converted)
        parsed[name] = clean
    return parsed


def finite(value):
    return value is not None and math.isfinite(value)


def metric_summary(rows):
    if not rows:
        return {}
    numeric = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and finite(value):
                numeric[key].append((row.get("epoch"), value))

    def best(key, mode):
        candidates = numeric.get(key, [])
        if not candidates:
            return None
        chooser = min if mode == "min" else max
        return chooser(candidates, key=lambda item: item[1])

    final = rows[-1]
    result = {
        "rows": len(rows),
        "first_epoch": rows[0].get("epoch"),
        "last_epoch": final.get("epoch"),
        "columns": list(rows[0]),
        "final": final,
    }
    for key in (
        "loss",
        "val_loss",
        "x_loss",
        "match_rate",
        "rms_dist",
        "loss_atom_types",
        "loss_lengths",
        "loss_angles",
        "loss_frac_coords",
        "loss_kl",
        "lr_pg1",
    ):
        if key in ("match_rate",):
            item = best(key, "max")
        else:
            item = best(key, "min")
        if item is not None:
            result[f"best_{key}"] = {"epoch": item[0], "value": item[1]}
    return result


def candidate_rows(record):
    if "candidates" in record:
        return record["candidates"]
    posterior = record.get("posterior", {})
    return posterior.get("candidates", [])


def target_match(record):
    if "structure_match_at_k" in record:
        return bool(record["structure_match_at_k"])
    if "structure_match" in record:
        return bool(record["structure_match"])
    return bool(record.get("posterior", {}).get("structure_match_at_k"))


def target_rms(record):
    value = record.get("best_rms")
    if value is None:
        value = record.get("posterior", {}).get("best_rms")
    if value is None:
        value = record.get("rms")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def per_target_summary(records):
    total = len(records)
    counts = Counter(int(record["target_num_atoms"]) for record in records)
    matches_by_n = Counter()
    for record in records:
        if target_match(record):
            matches_by_n[int(record["target_num_atoms"])] += 1
    finite_best = [target_rms(r) for r in records if target_rms(r) is not None]
    candidate_count = comp_ok = valid = candidate_matches = 0
    candidate_hits_per_target = Counter()
    candidate_rms = []
    samples_per_target = max((len(candidate_rows(record)) for record in records), default=0)
    for record in records:
        target_candidate_matches = 0
        for candidate in candidate_rows(record):
            candidate_count += 1
            comp_ok += int(bool(candidate.get("composition_ok")))
            valid += int(bool(candidate.get("struct_valid")))
            rms = candidate.get("rms")
            try:
                is_match = rms is not None and math.isfinite(float(rms))
                candidate_matches += int(is_match)
                target_candidate_matches += int(is_match)
                if is_match:
                    candidate_rms.append(float(rms))
            except (TypeError, ValueError):
                pass
        candidate_hits_per_target[target_candidate_matches] += 1

    sorted_candidate_rms = sorted(candidate_rms)

    def quantile(fraction):
        if not sorted_candidate_rms:
            return None
        index = int((len(sorted_candidate_rms) - 1) * fraction)
        return sorted_candidate_rms[index]

    return {
        "targets": total,
        "matches": sum(target_match(record) for record in records),
        "match_rate": sum(target_match(record) for record in records) / max(total, 1),
        "mean_finite_best_rms": sum(finite_best) / max(len(finite_best), 1),
        "finite_best_rms_targets": len(finite_best),
        "candidates": candidate_count,
        "candidate_composition_rate": comp_ok / max(candidate_count, 1),
        "candidate_struct_valid_rate": valid / max(candidate_count, 1),
        "candidate_match_rate": candidate_matches / max(candidate_count, 1),
        "samples_per_target": samples_per_target,
        "targets_with_zero_candidate_matches": candidate_hits_per_target[0],
        "targets_with_all_candidates_matching": candidate_hits_per_target[samples_per_target],
        "candidate_match_count_distribution": {
            str(count): number
            for count, number in sorted(candidate_hits_per_target.items())
        },
        "matched_candidate_rms_quantiles": {
            "p10": quantile(0.1),
            "p50": quantile(0.5),
            "p90": quantile(0.9),
        },
        "by_num_atoms": {
            str(n): {
                "targets": counts[n],
                "matches": matches_by_n[n],
                "match_rate": matches_by_n[n] / counts[n],
            }
            for n in sorted(counts)
        },
    }


def match_identity(records):
    return [
        str(record.get("id"))
        for record in records
        if target_match(record)
    ]


def main():
    train = parse_csv_sections(INPUT / "train_metrics_all.csv")
    validation = parse_csv_sections(INPUT / "validation_metrics_all.csv")
    history = parse_csv_sections(INPUT / "loss_history_all.csv")
    targets = json.loads((INPUT / "per_target_all.json").read_text(encoding="utf-8"))

    result = {
        "sources": {
            "train_metrics": metric_summary_summary(train),
            "validation_metrics": metric_summary_summary(validation),
            "loss_history": metric_summary_summary(history),
            "per_target": {name: per_target_summary(records) for name, records in targets.items()},
        },
        "match_set_overlap": match_set_overlap(targets),
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def metric_summary_summary(sections):
    return {name: metric_summary(rows) for name, rows in sections.items()}


def match_set_overlap(targets):
    sets = {name: set(match_identity(records)) for name, records in targets.items()}
    names = list(sets)
    result = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            intersection = len(sets[left] & sets[right])
            union = len(sets[left] | sets[right])
            if intersection:
                result[f"{left}__{right}"] = {
                    "left_matches": len(sets[left]),
                    "right_matches": len(sets[right]),
                    "intersection": intersection,
                    "jaccard": intersection / max(union, 1),
                }
    return result


if __name__ == "__main__":
    main()
