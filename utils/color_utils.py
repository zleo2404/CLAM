import numpy as np
import torch
import torchstain
from PIL import Image

class MacenkoNormalizeWrapper:
	def __init__(self, target_path, device='cpu'):
		self.device = torch.device(device)
		self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

		if target_path.endswith('.npz'):
			# reference averaged over many patches of many slides by
			# scripts/fit_stain_reference.py; fit() is bypassed and its two outputs
			# (HERef, maxCRef) are injected directly
			ref = np.load(target_path)
			self.normalizer.HERef = torch.from_numpy(ref['HERef']).float().to(self.device)
			self.normalizer.maxCRef = torch.from_numpy(ref['maxCRef']).float().to(self.device)
			print('[stain_normalizer] reference loaded from {} ({} patches, {} slides, {})'.format(
				target_path, ref['n_patches'], ref['n_slides'], ref['reduce']))
		else:
			# single reference patch: HERef and maxCRef come from that one image
			target = np.array(Image.open(target_path).convert('RGB'))
			self.normalizer.fit(torch.from_numpy(target).permute(2, 0, 1).to(self.device))
			print('[stain_normalizer] reference fitted on the single patch {}'.format(target_path))

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