import numpy as np
import torch
from model import patchify, unpatchify
from config import PATCH_SIZE, IMG_SIZE

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalise(t):
    img = (t.cpu().float() * STD + MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def make_masked_image(image, mask, patch_size=PATCH_SIZE, img_size=IMG_SIZE):
    w   = img_size // patch_size
    out = denormalise(image).copy().astype(float)
    for idx in range(mask.size(0)):
        if mask[idx] == 1:
            r = (idx // w) * patch_size
            c = (idx %  w) * patch_size
            out[r:r+patch_size, c:c+patch_size] = 127
    return out.astype(np.uint8)


def reconstruct_image(pred_patches, original, mask):
    orig_p = patchify(original.unsqueeze(0)).squeeze(0)
    mean   = orig_p.mean(dim=-1, keepdim=True)
    var    = orig_p.var(dim=-1, keepdim=True)
    pred_d = pred_patches * (var + 1e-6).sqrt() + mean
    comp   = torch.where(mask.unsqueeze(-1).bool(), pred_d, orig_p)
    return denormalise(unpatchify(comp.unsqueeze(0)).squeeze(0))