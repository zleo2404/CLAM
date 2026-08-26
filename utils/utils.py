import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_MIL_coords(batch):
	img = torch.cat([item[0] for item in batch], dim=0)
	label = torch.LongTensor([item[1] for item in batch])
	coords = [item[2] for item in batch]
	if isinstance(coords[0], np.ndarray):
		coords = np.vstack(coords)
	elif isinstance(coords[0], torch.Tensor):
		coords = torch.cat(coords, dim=0)
	return [img, label, coords]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1, return_coords=False):
	collate = collate_MIL_coords if return_coords else collate_MIL
	kwargs = {'num_workers': num_workers, 'pin_memory': False} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn=collate, **kwargs)
	return loader 

def get_split_loader(split_dataset, training=False, testing=False, weighted=False, return_coords=False):
	"""
		return either the validation loader or training loader 
	"""
	collate = collate_MIL_coords if return_coords else collate_MIL
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)), collate_fn=collate, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset), collate_fn=collate, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset), collate_fn=collate, **kwargs)
	else:
		ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset)*0.1), replace=False)
		loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate, **kwargs )

	return loader

class FocalLoss(nn.Module):
	"""
	Multi-class focal loss (Lin et al., 2017).

	Scales cross entropy by (1 - p_t) ** gamma, so easy slides contribute less and
	the hard minority ones dominate the gradient. gamma=0 reduces to plain CE.
	alpha, if given, is an additional per-class weight (a list or tensor of length
	n_classes) applied on top.
	"""
	def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
		super().__init__()
		self.gamma = gamma
		self.reduction = reduction
		if alpha is not None and not isinstance(alpha, torch.Tensor):
			alpha = torch.tensor(alpha, dtype=torch.float)
		self.register_buffer('alpha', alpha)

	def forward(self, logits, target):
		target = target.view(-1)
		log_pt = F.log_softmax(logits, dim=-1).gather(1, target.view(-1, 1)).squeeze(1)
		pt = log_pt.exp()
		loss = -((1 - pt) ** self.gamma) * log_pt

		if self.alpha is not None:
			loss = loss * self.alpha.to(logits.device)[target]

		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()
		return loss

def compute_inverse_frequency_alpha(split_dataset, n_classes):
	"""
	Per-class focal weights as inverse class frequency, using the sklearn 'balanced'
	convention: N / (n_classes * count_c). Classes absent from the split get 0.

	Computed on the training split only -- deriving it from val/test would leak.
	"""
	counts = np.array([len(split_dataset.slide_cls_ids[c]) for c in range(n_classes)], dtype=float)
	total = counts.sum()
	alpha = np.divide(total, n_classes * counts, out=np.zeros(n_classes), where=counts > 0)
	return alpha.tolist()

def parse_focal_alpha(spec, split_dataset, n_classes):
	"""
	Resolve --focal_alpha: 'auto' (inverse class frequency), 'none', or an explicit
	comma-separated list of one weight per class.
	"""
	if spec is None or spec == 'none':
		return None
	if spec == 'auto':
		return compute_inverse_frequency_alpha(split_dataset, n_classes)

	values = [float(v) for v in spec.split(',')]
	assert len(values) == n_classes, \
		'--focal_alpha expects {} values for {} classes, got {}'.format(n_classes, n_classes, len(values))
	return values

def get_optim(model, args):
	params = filter(lambda p: p.requires_grad, model.parameters())
	if args.opt == "adam":
		optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.reg)
	elif args.opt == "adamw":
		optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def get_scheduler(optimizer, args):
	"""
	Optional learning-rate schedule. Returns None when --scheduler is 'none', which
	keeps the original constant-LR behaviour.

	'plateau' is stepped with the validation loss, the others once per epoch; the
	training loop handles that difference.
	"""
	scheduler_name = getattr(args, 'scheduler', 'none')

	if scheduler_name == 'none':
		return None
	elif scheduler_name == 'cosine':
		return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.scheduler_min_lr)
	elif scheduler_name == 'step':
		return optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
	elif scheduler_name == 'plateau':
		return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_gamma,
													patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
	else:
		raise NotImplementedError

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

