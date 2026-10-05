import os
import sys

sys.path.append("../server-pytorch")
import utils

from stardist.models import StarDist2D as S
from csbdeep.data import PercentileNormalizer
sd_norm_fun = PercentileNormalizer()


class SegmentationMethods:
    def __init__(self, methods_folder: str = '.'):
        # a map between method name and its file
        self.available_methods = {}
        self.rescan_methods(methods_folder)

        self.last_used_method = "__intentionally_invalid_method_name__"
        self.last_used_fun = None


    def rescan_methods(self, methods_folder: str = '.'):
        '''
        Search for model.arch.*.model files in the 'methods_folder',
        and return them as a map (dict) between method name and its file.
        '''

        methods = utils.list_models_files(methods_folder, 'tensorflow')
        self.available_methods = {}

        for m in methods:
            model,net,_ = utils.filename_to_atoms(m)
            self.available_methods[f"{net}.{model}"] = os.path.join(methods_folder, m)

        self.available_methods["fakenet.square"] = "invalid_path"

        print("Discovery of available networks is finished.")


    def list_avail_methods(self):
        return self.available_methods.keys()


    def get_segmentation_fun(self, wanted_method:str):
        # re-use the currently activated method
        if wanted_method == self.last_used_method:
            return self.last_used_fun

        # sanity check
        if wanted_method not in self.available_methods.keys():
            print("REQUESTED MODEL NOT AVAILABLE!")
            return None

        wanted_file = self.available_methods[wanted_method]
        wanted_net,wanted_model = wanted_method.split('.')

        if wanted_net == "stardist":
            print(">>>>> DIDN'T YOU FORGET TO USE: export CUDA_VISIBLE_DEVICES=0   <<<<<")
            print(f"LOADING STARDIST model: {wanted_file}")
            model = S.from_pretrained('2D_versatile_fluo') if wanted_model == "original" else S.from_pretrained(wanted_file)
            #self.last_used_fun = lambda i : model.predict_instances(utils.basic_inplace_normalization(i))[0]
            self.last_used_fun = lambda i : model.predict_instances(i, normalizer=sd_norm_fun)[0].astype('uint16')
            self.last_used_method = wanted_method

        elif wanted_net == "fakenet":
            print(f"LOADING FAKENET square model...")
            self.last_used_fun = utils.set_one_everywhere
            self.last_used_method = wanted_method

        else:
            print("NOT LOADING ANY MODEL!?")
            return None ## this should never happen

        return self.last_used_fun

