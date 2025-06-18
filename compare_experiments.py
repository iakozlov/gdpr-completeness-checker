#!/usr/bin/env python3
import os
import json
import re
import argparse
from typing import Dict, Tuple, List, Set

# Mapping between internal requirement numbers and R labels used in evaluation
REQ_NUM_TO_R = {
    1: "10", 2: "11", 3: "12", 4: "13", 5: "15", 6: "16", 7: "17", 8: "18",
    9: "19", 10: "20", 11: "22", 12: "21", 13: "23", 14: "24", 15: "25",
    16: "26", 17: "27", 18: "28", 19: "29"
}
R_TO_REQ_NUM = {v: str(k) for k, v in REQ_NUM_TO_R.items()}


def load_baseline_results(baseline_dir: str) -> Tuple[Set[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
    """Load baseline predictions and ground truth from aggregated results.

    Returns two sets of keys (dpa, requirement_label, segment_id): one for
    baseline predictions marked as satisfied and one for ground truth satisfied
    segments.
    """

    agg_file = os.path.join(baseline_dir, "aggregated_evaluation_results.json")
    if os.path.exists(agg_file):
        with open(agg_file, "r") as f:
            data = json.load(f)
        preds: Set[Tuple[str, str, str]] = set()
        truths: Set[Tuple[str, str, str]] = set()
        for dpa, dpa_data in data.get("per_dpa_results", {}).items():
            for r_label, details in dpa_data.get("requirements_details", {}).items():
                for seg in details.get("satisfied_segments", []):
                    preds.add((dpa, r_label, str(seg)))
                for seg in details.get("ground_truth_segments", []):
                    truths.add((dpa, r_label, str(seg)))
        return preds, truths

    # Fallback to legacy baseline_results files if aggregated file not found
    results = []
    for root, _, files in os.walk(baseline_dir):
        for name in files:
            if name.startswith("baseline_results") and name.endswith(".json"):
                with open(os.path.join(root, name), "r") as f:
                    try:
                        data = json.load(f)
                        results.extend(data)
                    except Exception:
                        continue
    if not results:
        raise FileNotFoundError(
            f"No aggregated_evaluation_results.json or baseline_results*.json files found in {baseline_dir}"
        )

    preds: Set[Tuple[str, str, str]] = set()
    truths: Set[Tuple[str, str, str]] = set()
    for entry in results:
        dpa = entry.get("dpa")
        req_num = entry.get("requirement_id")
        seg_id = str(entry.get("segment_id"))
        r_label = REQ_NUM_TO_R.get(int(req_num), str(req_num))
        if entry.get("predicted") == "satisfied":
            preds.add((dpa, r_label, seg_id))
        if entry.get("ground_truth") == "satisfied":
            truths.add((dpa, r_label, seg_id))
    return preds, truths


def parse_deolingo_results(file_path: str) -> Dict[Tuple[str, str, str], str]:
    """Parse Deolingo solver results into a dictionary."""
    pattern = re.compile(r"Processing DPA (.+?), Requirement (\d+), Segment (\d+)")
    status_pattern = re.compile(r"status\((\w+)\)")
    results = {}

    if not os.path.exists(file_path):
        return results

    with open(file_path, "r") as f:
        content = f.read()

    sections = content.split("-" * 50)
    for sec in sections:
        if not sec.strip():
            continue
        lines = [l.strip() for l in sec.strip().splitlines() if l.strip()]
        if not lines:
            continue
        m = pattern.search(lines[0])
        if not m:
            continue
        dpa, req_num, seg_id = m.group(1), m.group(2), m.group(3)
        r_label = REQ_NUM_TO_R.get(int(req_num), req_num)
        status = "not_mentioned"
        for line in lines:
            sm = status_pattern.search(line)
            if sm:
                status = sm.group(1)
                break
        results[(dpa, r_label, seg_id)] = status
    return results


def extract_facts(lp_file: str) -> List[str]:
    """Extract facts from an LP file."""
    if not os.path.exists(lp_file):
        return []
    facts = []
    with open(lp_file, "r") as f:
        lines = f.readlines()
    in_facts = False
    for line in lines:
        if line.strip().startswith("% 2. Facts"):
            in_facts = True
            continue
        if in_facts:
            if line.startswith("%") or line.strip() == "":
                break
            facts.append(line.strip().rstrip("."))
    return facts


def build_comparisons(baseline_preds: Set[Tuple[str, str, str]],
                      ground_truth: Set[Tuple[str, str, str]],
                      symbolic_results: Dict[Tuple[str, str, str], str],
                      lp_root: str) -> Dict:
    """Build comparison records between baseline and symbolic approaches."""

    comparisons_yes_no = []
    comparisons_no_yes = []
    total_pairs = 0

    symbolic_satisfied_keys = {k for k, v in symbolic_results.items() if v == "satisfied"}
    all_keys = baseline_preds.union(symbolic_satisfied_keys)

    for key in sorted(all_keys):
        dpa, r_label, seg_id = key
        baseline_pred = "satisfied" if key in baseline_preds else "not_satisfied"
        sym_pred = symbolic_results.get(key, "not_satisfied")
        ground = key in ground_truth

        lp_file = os.path.join(
            lp_root,
            f"dpa_{dpa.replace(' ', '_')}",
            f"req_{R_TO_REQ_NUM.get(r_label, r_label)}",
            f"segment_{seg_id}.lp",
        )
        facts = extract_facts(lp_file)

        record = {
            "dpa": dpa,
            "segment_id": seg_id,
            "requirement": r_label,
            "ground_truth": ground,
            "baseline_prediction": baseline_pred,
            "symbolic_prediction": sym_pred,
            "symbolic_facts": facts,
        }

        total_pairs += 1
        if baseline_pred == "satisfied" and sym_pred != "satisfied":
            comparisons_yes_no.append(record)
        elif baseline_pred != "satisfied" and sym_pred == "satisfied":
            comparisons_no_yes.append(record)

    return {
        "total_pairs": total_pairs,
        "baseline_satisfied_symbolic_not": comparisons_yes_no,
        "symbolic_satisfied_baseline_not": comparisons_no_yes,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and symbolic approach results")
    parser.add_argument("--baseline_dir", required=True, help="Directory with baseline results")
    parser.add_argument("--deolingo_results", required=True, help="Deolingo results from symbolic approach")
    parser.add_argument("--lp_root", required=True, help="Root directory containing generated LP files")
    parser.add_argument("--output", required=True, help="Output JSON file for comparison")
    args = parser.parse_args()

    baseline_preds, ground_truth = load_baseline_results(args.baseline_dir)
    symbolic_results = parse_deolingo_results(args.deolingo_results)

    comparison = build_comparisons(baseline_preds, ground_truth, symbolic_results, args.lp_root)

    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
