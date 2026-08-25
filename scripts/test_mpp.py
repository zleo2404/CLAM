import os
import argparse
import openslide
from tqdm import tqdm

def check_mpp_distribution(wsi_dir, target_mpp=0.25, tolerance=0.05):
    # Definisco le estensioni valide per le Whole Slide Images
    valid_extensions = ('.svs', '.ndpi', '.tif', '.tiff', '.mrxs')
    all_files = [f for f in os.listdir(wsi_dir) if f.lower().endswith(valid_extensions)]
    
    if not all_files:
        print(f"ERRORE: Nessuna WSI trovata nella cartella {wsi_dir}")
        return
        
    print(f"Trovate {len(all_files)} WSI. Inizio controllo dei metadati MPP...")
    
    # Liste per raggruppare i file in base al loro stato
    in_scale = []
    out_of_scale = []
    missing_mpp = []
    errors = []
    
    # tqdm mostra una comoda barra di avanzamento a schermo
    for slide_name in tqdm(all_files, desc="Analisi WSI"):
        wsi_path = os.path.join(wsi_dir, slide_name)
        try:
            wsi = openslide.OpenSlide(wsi_path)
            mpp_str = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X)
            
            if mpp_str is None:
                missing_mpp.append(slide_name)
            else:
                mpp = float(mpp_str)
                # Controllo se il valore rientra nel range di tolleranza
                if abs(mpp - target_mpp) <= tolerance:
                    in_scale.append((slide_name, mpp))
                else:
                    out_of_scale.append((slide_name, mpp))
                    
        except Exception as e:
            errors.append((slide_name, str(e)))
            
    # Generazione del report testuale finale
    print("\n" + "="*50)
    print("REPORT RISOLUZIONE WSI (MPP)")
    print("="*50)
    print(f"Target MPP impostato : {target_mpp} (+/- {tolerance})")
    print(f"Totale slide analizzate: {len(all_files)}")
    print(f"  - In scala (conformi): {len(in_scale)}")
    print(f"  - Fuori scala (anomale): {len(out_of_scale)}")
    print(f"  - MPP mancante nei metadati: {len(missing_mpp)}")
    print(f"  - Errori di lettura file : {len(errors)}")
    
    if len(out_of_scale) > 0:
        print("\n--- DETTAGLIO SLIDE FUORI SCALA ---")
        # Ordino la lista per valore di MPP cosi e piu facile da leggere
        out_of_scale.sort(key=lambda x: x[1])
        for name, val in out_of_scale:
            print(f"MPP: {val:.4f} -> {name}")
            
    if len(missing_mpp) > 0:
        print("\n--- DETTAGLIO SLIDE SENZA METADATI MPP ---")
        for name in missing_mpp:
            print(name)
            
    if len(errors) > 0:
        print("\n--- DETTAGLIO ERRORI DI LETTURA ---")
        for name, err in errors:
            print(f"{name} -> {err}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str, required=True, help="Cartella contenente le slide")
    parser.add_argument('--target_mpp', type=float, default=0.25, help="MPP di riferimento (default 0.25 per scansioni a 40x)")
    parser.add_argument('--tolerance', type=float, default=0.05, help="Tolleranza rispetto al target (default 0.05)")
    args = parser.parse_args()
    
    check_mpp_distribution(args.wsi_dir, args.target_mpp, args.tolerance)