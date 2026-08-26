import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
        
def has_UNI2():
    HAS_UNI2 = False
    UNI2_CKPT_PATH = ''
    # check if UNI2_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI2_CKPT_PATH is set
        if 'UNI2_CKPT_PATH' not in os.environ:
            raise ValueError('UNI2_CKPT_PATH not set')
        HAS_UNI2 = True
        UNI2_CKPT_PATH = os.environ['UNI2_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI2, UNI2_CKPT_PATH

def resolve_ckpt_path(path):
    """Accept either the checkpoint file itself or the directory that holds it."""
    if os.path.isdir(path):
        for filename in ('pytorch_model.bin', 'model.safetensors'):
            candidate = os.path.join(path, filename)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            'no pytorch_model.bin or model.safetensors found in {}'.format(path))
    if not os.path.isfile(path):
        raise FileNotFoundError('checkpoint not found: {}'.format(path))
    return path

def load_ckpt(path):
    """Load a state dict from either a .bin or a .safetensors checkpoint."""
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")

def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'uni_v2':
        HAS_UNI2, UNI2_CKPT_PATH = has_UNI2()
        assert HAS_UNI2, 'UNI2 is not available'
        # architecture per the MahmoodLab/UNI2-h model card. Built inside the try:
        # timm.layers.SwiGLUPacked is resolved while the dict is constructed, so an
        # old timm fails here rather than in create_model.
        try:
            timm_kwargs = {'model_name': 'vit_giant_patch14_224',
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
                           'dynamic_img_size': True}
            model = timm.create_model(pretrained=False, **timm_kwargs)
        except (TypeError, AttributeError) as e:
            raise RuntimeError(
                'could not build UNI2-h ({}). It needs timm >= 1.0.0 for reg_tokens and '
                'SwiGLUPacked, while env.yml pins timm==0.9.8.'.format(e))
        model.load_state_dict(load_ckpt(resolve_ckpt_path(UNI2_CKPT_PATH)), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms