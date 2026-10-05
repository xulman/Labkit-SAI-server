import os
import numpy as np
import torch
from instanseg import inference_class as IS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Will use: {device}")


def create_official_model():
    """
    creates (gets from somewhere, in fact) a pretrained latest model for fluo cell and nuclei images
    """
    return IS.InstanSeg("fluorescence_nuclei_and_cells", image_reader= "skimage.io", verbosity=1, device=device)


def apply_model(model, img):
    """
    img must be np.array and normalized, and grayscale (single channel)
    """
    # check if not gray image
    if len(img.shape) != 2:
        print(f"The input image must be grayscale (single channel), but was provided {img.shape}.")
        return
    # NB: 'img' should be turned to torch tensor, but the eval() methods do "tensorify" themselves too...

    # explicitly don't re-scale the input image
    img_pixel_size = 1.0

    # code below taken (and slightly adopted) from the original repo's file inference_class.py
    # NB: the provided 'model' is already associated with a particular device
    num_pixels = img.size  #np.cumprod(img.shape)[-1]
    if num_pixels < model.small_image_threshold:
        masks = model.eval_small_image(image = img,
                                       normalise = False,
                                       pixel_size = img_pixel_size,
                                       target = "cells", # only useful for the official models
                                       return_image_tensor = False)
    else:
        masks = model.eval_medium_image(image = img,
                                        normalise = False,
                                        pixel_size = img_pixel_size,
                                        target = "cells", # only useful for the official models
                                        return_image_tensor = False)
    return masks.detach().numpy()[0,0,:,:]
    # NB: always the first image on the list (of length 1 = batch size) of created images
    # NB: always the one-and-only channel


def load_model(model_name, subfolder = '', model_name_suffix = ".pt"):
    instanseg_script = torch.jit.load(os.path.join('.', subfolder, model_name + model_name_suffix))
    return IS.InstanSeg(model_type = instanseg_script, image_reader= "skimage.io", verbosity=1, device=device)

