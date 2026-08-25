import numpy as np
import torch
import torchstain
from PIL import Image

class MacenkoNormalizeWrapper:
	def __init__(self, target_path, device='cpu'):
		self.device = torch.device(device)
		self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
		target = np.array(Image.open(target_path).convert('RGB'))
		self.normalizer.fit(torch.from_numpy(target).permute(2, 0, 1).to(self.device))
		self._fail_count = 0

	def __call__(self, img):
		arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(self.device)
		try:
			norm, _, _ = self.normalizer.normalize(I=arr, stains=False)
			return Image.fromarray(norm.cpu().numpy().astype(np.uint8))
		except Exception:
			self._fail_count += 1
			if self._fail_count % 500 == 1:  # avoid flooding stdout on bad slides
				print(f'[stain_normalizer] normalization failed on {self._fail_count} patches so far, falling back to raw patch')
			return img