import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def parse_classification(json_string: str) -> Dict[str, List[str]]:
    if not isinstance(json_string, str) or json_string.strip() == "":
        return {}
    try:
        return json.loads(json_string)
    except Exception:
        return {}


def should_ignore_primary_label(label: str) -> bool:
    """Return True if the primary label should be ignored (e.g., 其他/其它/other)."""
    if not isinstance(label, str):
        return True
    normalized = label.strip().lower()
    # Cover common variants
    return normalized in {
        "其他",
        "其它",
        "其他类",
        "其它类",
        "其他类别",
        "其它类别",
        "other",
        "others",
    }


def iterate_label_pairs(rows: Iterable[Dict[str, str]]) -> Iterable[Tuple[str, str]]:
    for row in rows:
        class_obj = parse_classification(row.get("classification", ""))
        for primary_label, secondary_labels in class_obj.items():
            if should_ignore_primary_label(primary_label):
                continue
            if not isinstance(secondary_labels, list):
                continue
            for secondary_label in secondary_labels:
                if isinstance(secondary_label, str) and secondary_label.strip() != "":
                    yield primary_label, secondary_label.strip()


def aggregate_counts(df: pd.DataFrame) -> Tuple[Counter, Dict[str, Counter]]:
    total_by_primary: Counter = Counter()
    counts_by_primary_secondary: Dict[str, Counter] = defaultdict(Counter)

    for primary, secondary in iterate_label_pairs(df.to_dict("records")):
        total_by_primary[primary] += 1
        counts_by_primary_secondary[primary][secondary] += 1

    return total_by_primary, counts_by_primary_secondary


def build_flat_rows(
    total_by_primary: Counter, counts_by_primary_secondary: Dict[str, Counter]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_all = sum(total_by_primary.values())
    for primary in sorted(counts_by_primary_secondary.keys()):
        total_primary = int(total_by_primary.get(primary, 0))
        if total_primary == 0:
            continue
        # Collect all secondary labels for this primary, ordered by frequency desc
        secondaries_sorted = [
            s for s, _ in counts_by_primary_secondary[primary].most_common()
        ]
        keywords = "、".join(secondaries_sorted)
        ratio_primary = (total_primary / total_all) if total_all > 0 else 0.0
        rows.append(
            {
                "类别 (一级标签)": primary,
                "关键词 (二级标签)": keywords,
                "一级标签占比": round(ratio_primary, 4),
                "一级标签数量": total_primary,
            }
        )
    return rows


def write_output(df: pd.DataFrame, output_path: str) -> None:
    ext = Path(output_path).suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        # Default to CSV
        df.to_csv(output_path, index=False, encoding="utf-8-sig")


def analyze(input_csv: str, output_path: str) -> None:
    df = pd.read_csv(input_csv)
    total_by_primary, counts_by_primary_secondary = aggregate_counts(df)
    rows = build_flat_rows(total_by_primary, counts_by_primary_secondary)
    out_df = pd.DataFrame(
        rows,
        columns=[
            "类别 (一级标签)",
            "关键词 (二级标签)",
            "一级标签占比",
            "一级标签数量",
        ],
    )
    write_output(out_df, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate classification statistics into a flat table."
    )
    parser.add_argument(
        "--input",
        default="test2_classified.csv",
        help="Path to input CSV produced by topic_classification.py",
    )
    parser.add_argument(
        "--output",
        default="classified_stats.csv",
        help="Output file path (.csv or .xlsx). Defaults to classified_stats.csv",
    )
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()