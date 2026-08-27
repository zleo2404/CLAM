import argparse
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("process_list", type=Path)
    args = parser.parse_args()

    total_slides = 0
    total_patches = 0
    min_patches = float('inf')
    max_patches = 0

    with open(args.process_list, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        if reader.fieldnames is None or 'n_patches' not in reader.fieldnames:
            print("[ERRORE] Il file non contiene la colonna 'n_patches'.")
            return

        for row in reader:
            status = row.get("status", "").strip().lower()
            if status and status != "processed":
                continue

            try:
                n = int(row["n_patches"])
                total_patches += n
                total_slides += 1
                if n < min_patches:
                    min_patches = n
                if n > max_patches:
                    max_patches = n
            except ValueError:
                continue

    if total_slides == 0:
        print("[ATTENZIONE] Nessuna slide valida trovata.")
        return

    mean_patches = total_patches / total_slides

    print("\n--- STATISTICHE PATCH ESTRATTE ---")
    print(f"File analizzato       : {args.process_list}")
    print(f"Vetrini processati    : {total_slides}")
    print(f"Totale patch estratte : {total_patches:,}")
    print(f"Media patch per vetrino: {mean_patches:.2f}")
    print(f"Minimo patch in un vetrino: {min_patches}")
    print(f"Massimo patch in un vetrino: {max_patches}")
    print("----------------------------------\n")

if __name__ == "__main__":
    main()  