import os
import argparse
import openslide
from tqdm import tqdm

def check_level1_mpp(wsi_dir):
    valid_extensions = ('.svs', '.ndpi', '.tif', '.tiff', '.mrxs')
    all_files = [f for f in os.listdir(wsi_dir) if f.lower().endswith(valid_extensions)]
    
    if not all_files:
        print(f"ERRORE: Nessuna WSI trovata nella cartella {wsi_dir}")
        return
        
    print(f"Trovate {len(all_files)} WSI. Inizio controllo del Livello 1...")
    
    no_level_1 = []
    missing_mpp = []
    errors = []
    
    # Dizionario per raggruppare i file in base al loro MPP effettivo al Livello 1
    level_1_stats = {
        "~0.50 (20x)": [],
        "~1.00 (10x)": [],
        "~0.25 (40x)": [],
        "Altro (Scala anomala)": []
    }
    
    for slide_name in tqdm(all_files, desc="Analisi Livello 1"):
        wsi_path = os.path.join(wsi_dir, slide_name)
        try:
            wsi = openslide.OpenSlide(wsi_path)
            
            # 1. Controllo se il Livello 1 esiste
            if wsi.level_count < 2:
                no_level_1.append(slide_name)
                continue
                
            # 2. Leggo MPP del Livello 0
            mpp_str = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X)
            if mpp_str is None:
                missing_mpp.append(slide_name)
                continue
                
            mpp_0 = float(mpp_str)
            
            # 3. Calcolo MPP effettivo del Livello 1
            downsample_1 = wsi.level_downsamples[1]
            mpp_1 = mpp_0 * downsample_1
            
            # 4. Categorizzo il risultato
            if 0.45 <= mpp_1 <= 0.55:
                level_1_stats["~0.50 (20x)"].append((slide_name, mpp_1, downsample_1))
            elif 0.90 <= mpp_1 <= 1.10:
                level_1_stats["~1.00 (10x)"].append((slide_name, mpp_1, downsample_1))
            elif 0.20 <= mpp_1 <= 0.30:
                level_1_stats["~0.25 (40x)"].append((slide_name, mpp_1, downsample_1))
            else:
                level_1_stats["Altro (Scala anomala)"].append((slide_name, mpp_1, downsample_1))
                
        except Exception as e:
            errors.append((slide_name, str(e)))
            
    # Report Finale
    print("\n" + "="*50)
    print("REPORT RISOLUZIONE LIVELLO 1")
    print("="*50)
    print(f"Totale slide analizzate: {len(all_files)}")
    print(f"  - Slide SENZA Livello 1 (solo Livello 0): {len(no_level_1)}")
    print(f"  - Slide con MPP Livello 0 mancante: {len(missing_mpp)}")
    print(f"  - Errori di lettura file: {len(errors)}")
    
    print("\nDistribuzione MPP effettivo del Livello 1:")
    for category, items in level_1_stats.items():
        print(f"  - {category}: {len(items)} slide")
        
    # Stampiamo i dettagli dei casi strani per investigare
    if len(level_1_stats["~1.00 (10x)"]) > 0:
        print("\n--- ESEMPI DI SLIDE DOVE IL LIVELLO 1 E' A 10x ---")
        for name, mpp_1, ds in level_1_stats["~1.00 (10x)"][:5]:
            print(f"MPP L1: {mpp_1:.4f} (Downsample dal L0: {ds:.2f}x) -> {name}")

    if len(level_1_stats["Altro (Scala anomala)"]) > 0:
        print("\n--- ESEMPI DI SLIDE CON LIVELLO 1 ANOMALO ---")
        for name, mpp_1, ds in level_1_stats["Altro (Scala anomala)"][:5]:
            print(f"MPP L1: {mpp_1:.4f} (Downsample dal L0: {ds:.2f}x) -> {name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str, required=True, help="Cartella contenente le slide")
    args = parser.parse_args()
    check_level1_mpp(args.wsi_dir)