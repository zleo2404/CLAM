import os
import random
import cv2
import numpy as np
import openslide
from PIL import Image

def extract_candidate_targets(wsi_path, output_dir, num_candidates=5, patch_size=1024):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        slide = openslide.open_slide(wsi_path)
    except Exception as e:
        print(f"Error opening slide: {e}")
        return
        
    w, h = slide.level_dimensions[0]
    print(f"Slide: {os.path.basename(wsi_path)}")
    print(f"Level 0 dimensions: {w} x {h}")
    print(f"Searching for {num_candidates} candidate patches ({patch_size}x{patch_size})...\n")
    
    candidates_found = 0
    attempts = 0
    max_attempts = 1500
    
    while candidates_found < num_candidates and attempts < max_attempts:
        attempts += 1
        
        # Sample random coordinates
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        
        # Read patch at level 0
        patch_pil = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
        patch_np = np.array(patch_pil)
        
        # HSV analysis for tissue content and contrast
        hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        mean_s = np.mean(s_channel)
        mean_v = np.mean(v_channel)
        std_v = np.std(v_channel)
        
        # Filter background, overexposed areas, and low-contrast regions
        if mean_s > 45 and 100 < mean_v < 220 and std_v > 25:
            out_name = f"target_cand_{candidates_found+1}_x{x}_y{y}.png"
            out_path = os.path.join(output_dir, out_name)
            patch_pil.save(out_path)
            
            print(f"[+] Candidate {candidates_found+1} saved at (X:{x}, Y:{y})")
            print(f"    Saturation: {mean_s:.1f}, Value: {mean_v:.1f}, Texture/Std: {std_v:.1f}")
            
            candidates_found += 1
            
    if candidates_found == 0:
        print("\n[-] No patches met the criteria.")
    else:
        print(f"\n[!] Completed. Saved {candidates_found} candidate images to '{output_dir}'.")

if __name__ == "__main__":
    slide_path = "/scratch.hpc/sabrina.tassinari/ProgettoTesi/wsi_organizzate/TCGA-A8-A0A7-01A-01-TS1.dfc11237-aff0-442f-8a39-ce81f0d4aeb1.svs"
    output_dir = "/scratch.hpc/leonardo.meloni/CLAM/target_patches"
    
    extract_candidate_targets(
        wsi_path=slide_path, 
        output_dir=output_dir, 
        num_candidates=6, 
        patch_size=1024
    )