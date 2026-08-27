"""
Filtra una process_list (CLAM) mantenendo solo i vetrini utilizzabili.

Rimuove:
  1. i vetrini che NON sono tessuto tumorale primario (sample type code != 01),
     quindi elimina 11A (normale), 06 (metastasi), 02, ecc.
  2. i vetrini il cui caso ha her2_result = "Equivocal" nel file dei casi
     (di default vengono tenuti solo Positive/Negative; con --keep-unknown
      si tengono anche i casi non presenti nel file dei casi).

Uso:
    python filter_process_list.py process_list.csv cases.csv -o process_list_filtered.csv

Nome slide atteso:
    TCGA-3C-AAAU-01A-01-TS1.<uuid>.svs
     |---- case id ----|^^^  <- sample type code
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# codici "sample type" TCGA da mantenere: 01 = Primary Solid Tumor
DEFAULT_KEEP_CODES = {"01"}
# risultati her2 da escludere
DEFAULT_EXCLUDE_HER2 = {"equivocal"}

SLIDE_RE = re.compile(
    r"^(?P<case>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(?P<sample>\d{2})(?P<vial>[A-Z]?)",
    re.IGNORECASE,
)


def parse_slide_id(slide_id: str):
    """Ritorna (case_id, sample_code) oppure (None, None) se il nome non e' TCGA."""
    m = SLIDE_RE.match(slide_id.strip())
    if not m:
        return None, None
    return m.group("case").upper(), m.group("sample")


def load_cases(path: Path):
    """case_id -> her2_result (lowercase)."""
    mapping = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            sys.exit(f"[ERRORE] {path} sembra vuoto.")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        id_col = cols.get("cases.submitter_id") or cols.get("submitter_id") or cols.get("case_id")
        her2_col = cols.get("her2_result") or cols.get("her2")
        if id_col is None or her2_col is None:
            sys.exit(f"[ERRORE] {path}: servono le colonne 'cases.submitter_id' e 'her2_result'.")
        for row in reader:
            case = (row[id_col] or "").strip().upper()
            if not case:
                continue
            mapping[case] = (row[her2_col] or "").strip().lower()
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Filtra la process_list CLAM.")
    ap.add_argument("process_list", type=Path, help="CSV con slide_id, process, status, ...")
    ap.add_argument("cases", type=Path, help="CSV con cases.submitter_id, ihc_score, her2_result")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="CSV filtrato (default: <input>_filtered.csv)")
    ap.add_argument("--report", type=Path, default=None,
                    help="CSV opzionale con le righe scartate e il motivo")
    ap.add_argument("--keep-codes", default=",".join(sorted(DEFAULT_KEEP_CODES)),
                    help="sample type codes da tenere, separati da virgola (default: 01)")
    ap.add_argument("--exclude-her2", default=",".join(sorted(DEFAULT_EXCLUDE_HER2)),
                    help="valori her2_result da escludere (default: equivocal)")
    ap.add_argument("--keep-unknown", action="store_true",
                    help="tieni i vetrini il cui caso non e' presente nel file dei casi")
    args = ap.parse_args()

    keep_codes = {c.strip() for c in args.keep_codes.split(",") if c.strip()}
    exclude_her2 = {v.strip().lower() for v in args.exclude_her2.split(",") if v.strip()}
    out_path = args.output or args.process_list.with_name(
        args.process_list.stem + "_filtered.csv")

    cases = load_cases(args.cases)

    with open(args.process_list, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            sys.exit(f"[ERRORE] {args.process_list} sembra vuoto.")
        slide_col = reader.fieldnames[0]  # tipicamente 'slide_id'
        rows = list(reader)
        fieldnames = reader.fieldnames

    kept, dropped = [], []
    for row in rows:
        slide = (row.get(slide_col) or "").strip()
        case, sample = parse_slide_id(slide)

        if case is None:
            dropped.append((row, "nome slide non riconosciuto"))
            continue
        if sample not in keep_codes:
            dropped.append((row, f"sample type {sample} non ammesso"))
            continue

        her2 = cases.get(case)
        if her2 is None:
            if args.keep_unknown:
                kept.append(row)
            else:
                dropped.append((row, "caso assente nel file dei casi"))
            continue
        if her2 in exclude_her2:
            dropped.append((row, f"her2_result = {her2}"))
            continue

        kept.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames + ["motivo_scarto"])
            writer.writeheader()
            for row, reason in dropped:
                out = dict(row)
                out["motivo_scarto"] = reason
                writer.writerow(out)

    print(f"Righe in ingresso : {len(rows)}")
    print(f"Mantenute         : {len(kept)}  -> {out_path}")
    print(f"Scartate          : {len(dropped)}")
    motivi = {}
    for _, reason in dropped:
        motivi[reason] = motivi.get(reason, 0) + 1
    for reason, n in sorted(motivi.items(), key=lambda x: -x[1]):
        print(f"  - {reason}: {n}")


if __name__ == "__main__":
    main()