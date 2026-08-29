#!/usr/bin/env python3
"""
Incrocia il file CSV clinico (cases.submitter_id, ihc_score, her2_result)
con i vetrini .svs presenti in una cartella e stampa le statistiche del
dataset prima e dopo i filtri.

Uso:
    python analisi_dataset.py clinical.csv /percorso/cartella/vetrini
    python analisi_dataset.py clinical.csv /percorso/vetrini --csv-out riepilogo.csv
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict


# --------------------------------------------------------------------
# Parsing dei nomi file
# --------------------------------------------------------------------

def parse_barcode(filename):
    """
    Estrae le componenti dal nome file di un vetrino TCGA.

    Esempio:
        TCGA-3C-AAAU-01A-01-TS1.2F52DD63-...-E06092DB6CC1.svs
        -> patient_id  = TCGA-3C-AAAU
           sample_vial = 01A
           sample_type = 01
           vial        = A
           slide_code  = TS1
           slide_type  = TS
    """
    base = os.path.basename(filename)
    barcode = base.split('.')[0]          # scarta UUID ed estensione
    parts = barcode.split('-')

    if len(parts) < 6 or parts[0] != 'TCGA':
        return None

    return {
        'filename': base,
        'barcode': barcode,
        'patient_id': '-'.join(parts[:3]),
        'sample_vial': parts[3],
        'sample_type': parts[3][:2],
        'vial': parts[3][2:] if len(parts[3]) > 2 else '',
        'portion': parts[4],
        'slide_code': parts[5],
        'slide_type': parts[5][:2],
    }


def scan_slides(folder):
    """Percorre ricorsivamente la cartella e restituisce i vetrini validi."""
    slides, skipped = [], []

    for root, _dirs, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith('.svs'):
                continue
            info = parse_barcode(f)
            if info is None:
                skipped.append(f)
            else:
                info['path'] = os.path.join(root, f)
                slides.append(info)

    return slides, skipped


# --------------------------------------------------------------------
# Lettura del CSV clinico
# --------------------------------------------------------------------

def normalize_result(value):
    """Uniforma i valori di her2_result (rimuove spazi e ':' finali)."""
    return value.strip().rstrip(':').strip().capitalize()


def load_clinical(csv_path):
    """
    Legge il CSV clinico. Attende le colonne:
        cases.submitter_id, ihc_score, her2_result
    Restituisce un dizionario {patient_id: {'ihc': ..., 'her2': ...}}.
    """
    clinical = {}
    duplicati = []

    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)

        campi = {c.strip(): c for c in (reader.fieldnames or [])}
        col_id = campi.get('cases.submitter_id') or campi.get('case_id')
        col_ihc = campi.get('ihc_score')
        col_her2 = campi.get('her2_result')

        if not (col_id and col_ihc and col_her2):
            sys.exit(
                'ERRORE: il CSV deve contenere le colonne '
                'cases.submitter_id, ihc_score, her2_result.\n'
                f'Colonne trovate: {reader.fieldnames}'
            )

        for riga in reader:
            pid = riga[col_id].strip()
            if not pid:
                continue
            if pid in clinical:
                duplicati.append(pid)
                continue
            clinical[pid] = {
                'ihc': riga[col_ihc].strip(),
                'her2': normalize_result(riga[col_her2]),
            }

    return clinical, duplicati


# --------------------------------------------------------------------
# Stampa
# --------------------------------------------------------------------

def titolo(testo):
    print()
    print('=' * 62)
    print(testo)
    print('=' * 62)


def tabella_distribuzione(counter, ordine=None):
    """Stampa una tabella classe / numero / percentuale."""
    totale = sum(counter.values())
    if totale == 0:
        print('  (nessun caso)')
        return

    chiavi = ordine or sorted(counter)
    print(f"  {'Classe':<20}{'N':>8}{'%':>10}")
    print('  ' + '-' * 38)
    for k in chiavi:
        n = counter.get(k, 0)
        if n == 0 and ordine:
            continue
        print(f'  {k:<20}{n:>8}{100 * n / totale:>9.1f}%')
    print('  ' + '-' * 38)
    print(f"  {'Totale':<20}{totale:>8}{100.0:>9.1f}%")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Incrocia il CSV clinico con i vetrini presenti su disco.'
    )
    ap.add_argument('csv_clinico', help='CSV con cases.submitter_id, ihc_score, her2_result')
    ap.add_argument('cartella_vetrini', help='Cartella contenente i file .svs')
    ap.add_argument('--sample-type', default='01',
                    help="Codice del tipo di campione da conservare (default: 01, tumore solido primario)")
    ap.add_argument('--csv-out', default=None,
                    help='Se specificato, salva in questo file il dataset finale')
    args = ap.parse_args()

    clinical, dup_csv = load_clinical(args.csv_clinico)
    slides, skipped = scan_slides(args.cartella_vetrini)

    titolo('1. FILE LETTI')
    print(f'  Righe nel CSV clinico (pazienti univoci): {len(clinical)}')
    if dup_csv:
        print(f'  ATTENZIONE: {len(dup_csv)} righe duplicate nel CSV, ignorate')
    print(f'  Vetrini .svs trovati su disco:            {len(slides)}')
    if skipped:
        print(f'  ATTENZIONE: {len(skipped)} file con nome non riconosciuto:')
        for f in skipped[:5]:
            print(f'      {f}')
        if len(skipped) > 5:
            print(f'      ... e altri {len(skipped) - 5}')

    # ---------------- corrispondenze ----------------
    abbinati, senza_clinica = [], []
    for s in slides:
        if s['patient_id'] in clinical:
            s.update(clinical[s['patient_id']])
            abbinati.append(s)
        else:
            senza_clinica.append(s)

    pazienti_con_vetrino = {s['patient_id'] for s in abbinati}
    senza_vetrino = sorted(set(clinical) - pazienti_con_vetrino)

    titolo('2. CORRISPONDENZE CSV <-> VETRINI')
    print(f'  Vetrini abbinati a un record clinico:     {len(abbinati)}')
    print(f'  Vetrini SENZA record clinico:             {len(senza_clinica)}')
    print(f'  Pazienti nel CSV SENZA vetrino su disco:  {len(senza_vetrino)}')
    if senza_clinica:
        print('  Esempi di vetrini senza record clinico:')
        for s in senza_clinica[:5]:
            print(f"      {s['patient_id']}")
    if senza_vetrino:
        print('  Esempi di pazienti senza vetrino:')
        for p in senza_vetrino[:5]:
            print(f'      {p}')

    # ---------------- composizione dei vetrini ----------------
    titolo('3. COMPOSIZIONE DEI VETRINI ABBINATI')
    print('  Tipo di vetrino (DX = diagnostico FFPE; TS/MS/BS = tessuto congelato):')
    for tipo, n in sorted(Counter(s['slide_type'] for s in abbinati).items()):
        print(f'      {tipo}: {n}')
    print('  Tipo di campione (01 = tumore primario, 06 = metastasi, 11 = normale):')
    for tipo, n in sorted(Counter(s['sample_type'] for s in abbinati).items()):
        print(f'      {tipo}: {n}')

    # ---------------- distribuzione iniziale ----------------
    ORDINE = ['Positive', 'Negative', 'Equivocal']

    titolo('4. DISTRIBUZIONE INIZIALE (tutti i vetrini abbinati)')
    tabella_distribuzione(Counter(s['her2'] for s in abbinati), ORDINE)

    # ---------------- deduplicazione ----------------
    per_paziente = defaultdict(list)
    for s in abbinati:
        per_paziente[s['patient_id']].append(s)

    dedup = [sorted(v, key=lambda x: x['filename'])[0]
             for v in per_paziente.values()]
    rimossi_dup = len(abbinati) - len(dedup)

    titolo('5. DOPO DEDUPLICAZIONE (un vetrino per paziente)')
    print(f'  Vetrini rimossi come duplicati: {rimossi_dup}')
    print()
    tabella_distribuzione(Counter(s['her2'] for s in dedup), ORDINE)

    # ---------------- filtri finali ----------------
    dopo_tipo = [s for s in dedup if s['sample_type'] == args.sample_type]
    rimossi_tipo = len(dedup) - len(dopo_tipo)

    finale = [s for s in dopo_tipo if s['her2'] in ('Positive', 'Negative')]
    rimossi_eq = len(dopo_tipo) - len(finale)

    titolo('6. DATASET FINALE')
    print(f"  Esclusi per tipo campione diverso da '{args.sample_type}': {rimossi_tipo}")
    print(f'  Esclusi in quanto equivoci:                        {rimossi_eq}')
    print()
    tabella_distribuzione(Counter(s['her2'] for s in finale),
                          ['Positive', 'Negative'])

    # ---------------- riepilogo ----------------
    titolo('7. RIEPILOGO')
    print(f'  Vetrini di partenza:      {len(abbinati)}')
    print(f'  - duplicati:              {rimossi_dup}')
    print(f'  - tipo campione errato:   {rimossi_tipo}')
    print(f'  - casi equivoci:          {rimossi_eq}')
    print(f'  Dataset finale:           {len(finale)}')

    # ---------------- salvataggio ----------------
    if args.csv_out:
        with open(args.csv_out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['patient_id', 'slide_id', 'ihc_score', 'her2_result'])
            for s in sorted(finale, key=lambda x: x['patient_id']):
                w.writerow([s['patient_id'], s['barcode'], s['ihc'], s['her2']])
        print(f'\n  Dataset finale salvato in: {args.csv_out}')


if __name__ == '__main__':
    main()