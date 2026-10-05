import os
import numpy as np
import numpy.typing as npt


def basic_inplace_normalization(float_img: npt.NDArray) -> npt.NDArray:
    m = float_img.min()
    r = float_img.max() - m
    float_img = (float_img - m) / r
    return float_img


def set_one_everywhere(i):
    j = np.ones(i.shape, dtype='uint16')
    return j


def list_models_files(models_root_folder: str, env_code: str):
    res_ = os.listdir(models_root_folder)
    results = []

    for r in res_:
        if r.find('.model') > 0 \
                and os.path.isfile(os.path.join(models_root_folder,r)):
            _,_,env = filename_to_atoms(r)
            results.append(r) if env == env_code

    return results


def filename_to_atoms(filename: str):
    '''
    Assumes file names, e.g., like original.cellpose.torch.model, which encodes
    data_type, network, environment_nickname, and a mandatory suffix 'model'.
    For the example here, the function should return original,cellpose,torch.
    '''
    atoms = filename.split('.')
    if len(atoms) != 4:
        raise Exception(f"Model filename {filename} is not a dot-separated chain of four elements.")

    model = atoms[0]
    net = atoms[1]
    env = atoms[2]

    return (model,net,env)

