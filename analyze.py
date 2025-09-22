import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd


def parse_classification(json_string: str) -> Dict[str, List[str]]:
    if not isinstance(json_string, str) or json_string.strip() == "":
        return {}
    try:
        return json.loads(json_string)
    except Exception:
        return {}


def sanitize_secondary_label(label: Union[str, object]) -> str:
    if not isinstance(label, str):
        return ""
    cleaned = label.strip()
    if cleaned == "":
        return ""
    for delimiter in ("：", ":"):
        if delimiter in cleaned:
            cleaned = cleaned.split(delimiter, 1)[0].strip()
            break
    return cleaned


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
                cleaned_secondary = sanitize_secondary_label(secondary_label)
                if cleaned_secondary:
                    yield primary_label, cleaned_secondary


def collect_examples(df: pd.DataFrame, text_field: str = "ocr") -> Dict[str, List[str]]:
    """Collect example texts per primary label from the dataframe."""
    examples_by_primary: Dict[str, List[str]] = defaultdict(list)
    for row in df.to_dict("records"):
        class_obj = parse_classification(row.get("classification", ""))
        text_value = row.get(text_field, "")
        if not isinstance(text_value, str) or text_value.strip() == "":
            continue
        for primary_label in class_obj.keys():
            if should_ignore_primary_label(primary_label):
                continue
            examples_by_primary[primary_label].append(text_value.strip())
    return examples_by_primary


def collect_examples_by_pair(
    df: pd.DataFrame, text_field: str = "ocr"
) -> Dict[Tuple[str, str], List[str]]:
    """Collect example texts per (primary, secondary) pair from the dataframe."""
    examples_by_pair: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in df.to_dict("records"):
        class_obj = parse_classification(row.get("classification", ""))
        text_value = row.get(text_field, "")
        if not isinstance(text_value, str) or text_value.strip() == "":
            continue
        for primary_label, secondary_labels in class_obj.items():
            if should_ignore_primary_label(primary_label) or not isinstance(secondary_labels, list):
                continue
            for secondary_label in secondary_labels:
                cleaned_secondary = sanitize_secondary_label(secondary_label)
                if cleaned_secondary:
                    examples_by_pair[(primary_label, cleaned_secondary)].append(
                        text_value.strip()
                    )
    return examples_by_pair


def aggregate_counts(df: pd.DataFrame) -> Tuple[Counter, Dict[str, Counter]]:
    total_by_primary: Counter = Counter()
    counts_by_primary_secondary: Dict[str, Counter] = defaultdict(Counter)

    for primary, secondary in iterate_label_pairs(df.to_dict("records")):
        total_by_primary[primary] += 1
        counts_by_primary_secondary[primary][secondary] += 1

    return total_by_primary, counts_by_primary_secondary


def build_flat_rows(
    total_by_primary: Counter,
    counts_by_primary_secondary: Dict[str, Counter],
    examples_by_primary: Dict[str, List[str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_all = sum(total_by_primary.values())
    # Sort by frequency desc, then name asc
    primaries_sorted = sorted(
        counts_by_primary_secondary.keys(),
        key=lambda p: (-int(total_by_primary.get(p, 0)), str(p)),
    )
    for primary in primaries_sorted:
        total_primary = int(total_by_primary.get(primary, 0))
        if total_primary == 0:
            continue
        # Collect top-10 secondary labels for this primary, ordered by frequency desc
        secondaries_sorted = [
            s for s, _ in counts_by_primary_secondary[primary].most_common(10)
        ]
        cleaned_keywords: List[str] = []
        for secondary_label in secondaries_sorted:
            cleaned_secondary = sanitize_secondary_label(secondary_label)
            if cleaned_secondary and cleaned_secondary not in cleaned_keywords:
                cleaned_keywords.append(cleaned_secondary)
        keywords = "、".join(cleaned_keywords)
        ratio_primary = (total_primary / total_all) if total_all > 0 else 0.0
        # Pick up to 3 random example texts for this primary
        example_pool = examples_by_primary.get(primary, [])
        if len(example_pool) <= 3:
            sampled_examples = example_pool
        else:
            sampled_examples = random.sample(example_pool, 3)
        examples_text = "\n".join(sampled_examples)
        rows.append(
            {
                "类别 (一级标签)": primary,
                "关键词 (二级标签)": keywords,
                "一级标签占比": round(ratio_primary, 4),
                "一级标签数量": total_primary,
                "示例文本": examples_text,
            }
        )
    return rows


def build_secondary_rows(
    total_by_primary: Counter,
    counts_by_primary_secondary: Dict[str, Counter],
    examples_by_pair: Dict[Tuple[str, str], List[str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    # Sort primaries by frequency desc, then name asc
    primaries_sorted = sorted(
        counts_by_primary_secondary.keys(),
        key=lambda p: (-int(total_by_primary.get(p, 0)), str(p)),
    )
    for primary in primaries_sorted:
        total_primary = int(total_by_primary.get(primary, 0))
        if total_primary == 0:
            continue
        for secondary, count in counts_by_primary_secondary[primary].most_common():
            # Skip low-frequency secondary labels (< 5)
            if int(count) < 5:
                continue
            ratio = (count / total_primary) if total_primary > 0 else 0.0
            example_pool = examples_by_pair.get((primary, secondary), [])
            if len(example_pool) <= 3:
                sampled_examples = example_pool
            else:
                sampled_examples = random.sample(example_pool, 3)
            examples_text = "\n".join(sampled_examples)
            rows.append(
                {
                    "一级标签": primary,
                    "二级标签": secondary,
                    "二级标签数量": int(count),
                    "二级标签占比": round(ratio, 4),
                    "文本示例": examples_text,
                }
            )
    return rows


def write_output(
    df_or_sheets: Union[pd.DataFrame, Dict[str, pd.DataFrame]], output_path: str
) -> None:
    ext = Path(output_path).suffix.lower()
    if ext in {".xlsx", ".xls"}:
        if isinstance(df_or_sheets, dict):
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                for sheet_name, sheet_df in df_or_sheets.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            df_or_sheets.to_excel(output_path, index=False)
    else:
        # Default to CSV (single sheet only)
        if isinstance(df_or_sheets, dict):
            first_df = next(iter(df_or_sheets.values()))
            first_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            df_or_sheets.to_csv(output_path, index=False, encoding="utf-8-sig")


def analyze(input_csv: str, output_path: str) -> None:
    df = pd.read_csv(input_csv)
    total_by_primary, counts_by_primary_secondary = aggregate_counts(df)
    examples_by_primary = collect_examples(df, text_field="ocr")
    examples_by_pair = collect_examples_by_pair(df, text_field="ocr")
    primary_rows = build_flat_rows(
        total_by_primary, counts_by_primary_secondary, examples_by_primary
    )
    secondary_rows = build_secondary_rows(
        total_by_primary, counts_by_primary_secondary, examples_by_pair
    )
    out_primary = pd.DataFrame(
        primary_rows,
        columns=[
            "类别 (一级标签)",
            "关键词 (二级标签)",
            "一级标签占比",
            "一级标签数量",
            "示例文本",
        ],
    )
    out_secondary = pd.DataFrame(
        secondary_rows,
        columns=[
            "一级标签",
            "二级标签",
            "二级标签数量",
            "二级标签占比",
            "文本示例",
        ],
    )
    # Ensure sorted
    out_primary = out_primary.sort_values(
        by=["一级标签数量", "类别 (一级标签)"], ascending=[False, True]
    )
    out_secondary = out_secondary.sort_values(
        by=["二级标签数量", "一级标签", "二级标签"], ascending=[False, True, True]
    )
    sheets = {"一级统计": out_primary, "二级统计": out_secondary}
    write_output(sheets, output_path)


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
        default="classified_stats.xlsx",
        help="Output file path (.xlsx recommended). Defaults to classified_stats.xlsx",
    )
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
