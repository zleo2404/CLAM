import openslide
import cv2
import numpy as np
from PIL import Image
import os
import argparse
import random

# =====================================================================
# FUNZIONI DA TESTARE
# =====================================================================

def isBlurryPatch(patch_PIL, blur_thresh=150.0):
    patch_np = np.array(patch_PIL)
    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return True if variance < blur_thresh else False

def isInkPatch(patch_PIL, saturation_thresh=120, non_pink_ratio=0.1):
    patch_np = np.array(patch_PIL)
    hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    lower_tissue = np.array([120, 10, 50])  
    upper_tissue = np.array([170, 255, 255])
    tissue_mask = cv2.inRange(hsv, lower_tissue, upper_tissue)
    high_saturation_mask = hsv[:,:,1] > saturation_thresh
    artifact_mask = cv2.bitwise_and(cv2.bitwise_not(tissue_mask), high_saturation_mask.astype(np.uint8)*255)
    artifact_ratio = np.count_nonzero(artifact_mask) / (patch_np.shape[0] * patch_np.shape[1])
    return True if artifact_ratio > non_pink_ratio else False

def macenko_normalization(img_array, Io=240, alpha=1, beta=0.15):
    HERA_REFERENCE = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
    MAX_CONC_REF = np.array([1.9705, 1.0308])
    h, w, c = img_array.shape
    img_array = img_array.reshape((-1, 3))
    OD = -np.log((img_array.astype(float) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    if len(ODhat) == 0:
        return img_array.reshape((h, w, c))
    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    eigvecs = eigvecs[:, [1, 2]]
    T_hat = np.dot(ODhat, eigvecs)
    phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])
    minPhi, maxPhi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    vMin = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    HE = np.array((vMin, vMax)).T if vMin[0] > vMax[0] else np.array((vMax, vMin)).T
    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    C = C / maxC[:, None] * MAX_CONC_REF[:, None]
    Inorm = Io * np.exp(-np.dot(HERA_REFERENCE, C))
    Inorm = Inorm.T.reshape((h, w, 3))
    return np.clip(Inorm, 0, 255).astype(np.uint8)

# =====================================================================
# MOTORE DI TEST MULTIPLO
# =====================================================================

def test_random_wsis(wsi_dir, output_dir, num_samples=3):
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova le slide
    valid_extensions = ('.svs', '.ndpi', '.tif', '.tiff', '.mrxs')
    all_files = [f for f in os.listdir(wsi_dir) if f.lower().endswith(valid_extensions)]
    
    if not all_files:
        print(f"ERRORE: Nessuna WSI trovata nella cartella {wsi_dir}")
        return
        
    num_to_sample = min(num_samples, len(all_files))
    selected_slides = random.sample(all_files, num_to_sample)
    
    print(f"Trovate {len(all_files)} WSI in totale.")
    print(f"Selezionate {num_to_sample} slide random per il test.\n")
    
    for slide_name in selected_slides:
        wsi_path = os.path.join(wsi_dir, slide_name)
        base_name = os.path.splitext(slide_name)[0]
        print("=" * 50)
        print(f"SLIDE: {slide_name}")
        
        try:
            wsi = openslide.OpenSlide(wsi_path)
        except Exception as e:
            print(f"  [!] Errore lettura slide: {e}")
            continue
            
        # --- TEST 1: MPP ---
        native_mpp = wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, "NON TROVATO")
        print(f"  [MPP] Nativo: {native_mpp}")
        if native_mpp != "NON TROVATO":
            print(f"  [MPP] Fattore Scala (Target 0.5, 20x): {0.5 / float(native_mpp):.4f}")
        
        # --- TEST 2: CERCA PATCH CON TESSUTO ---
        w, h = wsi.dimensions
        patch_size = 512
        patch_PIL = None
        
        # Spirale di ricerca dal centro per trovare tessuto (evitare lo sfondo bianco)
        offsets = [0, -1500, 1500, -3000, 3000]
        for offset_x in offsets:
            for offset_y in offsets:
                cx, cy = (w // 2) + offset_x, (h // 2) + offset_y
                if cx < 0 or cy < 0 or cx >= w - patch_size or cy >= h - patch_size:
                    continue
                    
                temp_patch = wsi.read_region((cx, cy), 0, (patch_size, patch_size)).convert('RGB')
                
                # Se la media pixel e < 225, non e sfondo bianco puro, la teniamo!
                if np.mean(np.array(temp_patch)) < 225:
                    patch_PIL = temp_patch
                    print(f"  [PATCH] Tessuto trovato alle coordinate: ({cx}, {cy})")
                    break
            if patch_PIL: break
                
        if not patch_PIL:
            print("  [PATCH] Nessun tessuto trovato al centro. Salto ai prossimi test.")
            continue

        # Salva Originale
        orig_path = os.path.join(output_dir, f"{base_name}_originale.png")
        patch_PIL.save(orig_path)
        
        # --- TEST 3: ARTEFATTI ---
        patch_np = np.array(patch_PIL)
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blur = isBlurryPatch(patch_PIL, blur_thresh=150.0)
        is_ink = isInkPatch(patch_PIL)
        print(f"  [FILTRI] Varianza Laplaciano: {variance:.1f} | Sfocata? {is_blur} | Inchiostro? {is_ink}")

        # --- TEST 4: MACENKO ---
        try:
            norm_np = macenko_normalization(patch_np)
            norm_PIL = Image.fromarray(norm_np)
            norm_path = os.path.join(output_dir, f"{base_name}_macenko.png")
            norm_PIL.save(norm_path)
            print(f"  [MACENKO] Patch normalizzata salvata con successo.")
        except Exception as e:
            print(f"  [MACENKO] Fallito (forse matrice singolare o poco tessuto): {e}")
            
    print("\n" + "=" * 50)
    print("TEST COMPLETATI. Scarica la cartella di output per la validazione visiva.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str, required=True, help="Cartella contenente le slide HER2")
    parser.add_argument('--out', type=str, default="./test_multiplo_out", help="Cartella di output")
    parser.add_argument('--num', type=int, default=3, help="Numero di slide random da testare")
    args = parser.parse_args()
    
    test_random_wsis(args.wsi_dir, args.out, num_samples=args.num)