from torchvision import transforms

def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []
	
	if target_img_size > 0:
		# (h, w), not a scalar: a scalar resizes the shorter side and preserves the
		# aspect ratio, which would leave a non-square patch (e.g. a padded edge
		# patch) non-square instead of at the size the encoder expects
		trsforms.append(transforms.Resize((target_img_size, target_img_size)))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms