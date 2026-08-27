#!/usr/bin/env python3
"""
"""

import csv
import sys

# Se True, tiene solo le righe con process==0 e status=="processed"
FILTER_ONLY_PROCESSED = True


def extract_case_id(slide_id: str) -> str:
    """Estrae il case_id (prime 3 parti) da uno slide_id TCGA."""
    parts = slide_id.split("-")
    return "-".join(parts[:3])


def load_process_list(path: str):
    slide_ids = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row["slide_id"]
            # rimuove estensione .svs se presente
            if slide_id.lower().endswith(".svs"):
                slide_id = slide_id[:-4]

            if FILTER_ONLY_PROCESSED:
                proc = row.get("process", "").strip()
                status = row.get("status", "").strip().lower()
                if proc != "0" or status != "processed":
                    continue

            slide_ids.append(slide_id)
    return slide_ids


def load_her2_results(path: str):
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["cases.submitter_id"].strip()
            label = row["her2_result"].strip()
            mapping[case_id] = label
    return mapping


def main():
    if len(sys.argv) != 4:
        print("Uso: python3 build_labels.py process_list.csv her2_results.csv output.csv")
        sys.exit(1)

    process_list_path, her2_path, output_path = sys.argv[1:4]

    slide_ids = load_process_list(process_list_path)
    her2_map = load_her2_results(her2_path)

    rows = []
    missing = []
    for slide_id in slide_ids:
        case_id = extract_case_id(slide_id)
        label = her2_map.get(case_id)
        if label is None:
            missing.append((slide_id, case_id))
            continue
        rows.append((case_id, slide_id, label))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "slide_id", "label"])
        writer.writerows(rows)

    print(f"Scritte {len(rows)} righe in {output_path}")
    if missing:
        print(f"\nATTENZIONE: {len(missing)} slide della process list non hanno un case_id corrispondente nel file HER2:")
        for slide_id, case_id in missing:
            print(f"  - {slide_id} (case_id dedotto: {case_id})")


if __name__ == "__main__":
    main()