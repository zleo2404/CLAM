import os
import torch
import timm

local_dir = "/scratch.hpc/leonardo.meloni/conda_home/UNI2-modello"

print("1. Building model...")
timm_kwargs = {
    'model_name': 'vit_giant_patch14_224',
    'img_size': 224, 
    'patch_size': 14, 
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5, 
    'embed_dim': 1536,
    'mlp_ratio': 2.66667*2,
    'num_classes': 0, 
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 
    'act_layer': torch.nn.SiLU, 
    'reg_tokens': 8, 
    'dynamic_img_size': True
}

model = timm.create_model(pretrained=False, **timm_kwargs)

print("2. Loading weights...")
bin_path = os.path.join(local_dir, "pytorch_model.bin")
safe_path = os.path.join(local_dir, "model.safetensors")

try:
    if os.path.exists(bin_path):
        model.load_state_dict(torch.load(bin_path, map_location="cpu"), strict=True)
    elif os.path.exists(safe_path):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(safe_path), strict=True)
    else:
        print("Error: No weights found.")
        exit()
except Exception as e:
    print(f"Error: {e}")
    exit()

model.eval()
print("Weights loaded.")

print("3. Testing model...")
dummy_image = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    features = model(dummy_image)

print(f"Output shape: {features.shape}")

if features.shape == (1, 1536):
    print("TEST PASSED")
else:
    print("TEST FAILED")