import numpy as np
import torch
import torchstain

class MacenkoNormalizeWrapper:
	def __init__(self, target_path, device='cpu'):
		from PIL import Image
		self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
		target = np.array(Image.open(target_path).convert('RGB'))
		self.normalizer.fit(torch.from_numpy(target).permute(2, 0, 1))
		self.device = device

	def __call__(self, img):
		arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(self.device)
		try:
			norm, _, _ = self.normalizer.normalize(I=arr, stains=False)
			return Image.fromarray(norm.cpu().numpy().astype(np.uint8))
		except Exception:
			return img  # fall back to the raw patch rather than crashing extraction