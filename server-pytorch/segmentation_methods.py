import os
import sys

import utils

sys.path.append("../segmentation-wrappers")
import cellpose_wrapper as C
import instanseg_wrapper as I
import maskrcnn_wrapper as M
import sam2_demo as S


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

        methods = utils.list_models_files(methods_folder, 'torch')
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

        if wanted_net == "cellpose":
            print(f"LOADING CELLPOSE model: {wanted_file}")
            model = C.load_model(wanted_file) if wanted_model != "original" else C.create_official_model()
            self.last_used_fun = lambda i : C.apply_model(model,i)
            self.last_used_method = wanted_method
            # NB: keep the 'last_used_method' only if switching to this method went well,
            #     otherwise by not memorizing it, the request to the same method will trigger
            #     _again_ this code path

        elif wanted_net == "maskrcnnv1COCO":
            print(f"LOADING MASKRCNN_v1COCO  model: {wanted_file}")
            model = M.create_official_model()
            if wanted_model != "original":
                M.load_model(model, wanted_file)
            self.last_used_fun = lambda i : M.apply_model(model,utils.basic_inplace_normalization(i))[0]
            self.last_used_method = wanted_method

        elif wanted_net == "instanseg":
            print(f"LOADING INSTANSEG model: {wanted_file}")
            model = I.create_official_model()
            if wanted_model != "original":
                model = I.load_model(wanted_file, subfolder='.', model_name_suffix='')
            self.last_used_fun = lambda i : I.apply_model(model,utils.basic_inplace_normalization(i).copy()).astype('uint16')
            self.last_used_method = wanted_method

        elif wanted_net == "fakenet":
            print(f"LOADING FAKENET square model...")
            self.last_used_fun = utils.set_one_everywhere
            self.last_used_method = wanted_method

        elif wanted_net == "sam2":
            print(f"LOADING SAM2 model: {wanted_file}")
            model = S.Sam2BoxSegmenter(f"facebook/sam2.1-hiera-{wanted_model[9:]}")
            self.last_used_fun = lambda i : model.set_image(i).segment_box(20,20, 100,100)
            self.last_used_method = wanted_method

        else:
            print("NOT LOADING ANY MODEL!?")
            return None ## this should never happen

        return self.last_used_fun


