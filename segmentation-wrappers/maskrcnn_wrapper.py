import torchvision.models.detection as D
import torch
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Will use: {device}")


def create_official_model():
    """
    creates (gets from somewhere, in fact) a pretrained COCO_V1 model
    """
    weights = D.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    # alternatively:
    # model = torchvision.models.get_model('maskrcnn_resnet50_fpn', weights=DEFAULT)
    model = D.maskrcnn_resnet50_fpn(weights=weights).to(device)
    return model


def load_model(model, filepath):
    """
    Start with 'model = create_official_model()'.
    """
    with open(filepath,"rb") as f:
        w = pickle.load(f) #NB: reads-in always CPU versions
        if device.type == 'cuda':
            for k in w.keys():
                w[k] = w[k].to(device)
        model.load_state_dict(w)


def apply_model(model, img, res_confidence_threshold: float = 0.5):
    _ = model.eval()

    imgAsTensor = torch.reshape(torch.Tensor(img), [1,1,*img.shape]).to(device)
    outTensor = torch.empty(imgAsTensor.shape)

    complete_model_response = model(imgAsTensor)
    outTensor = fill_masks_from_retval_struct(imgAsTensor, outTensor, complete_model_response)

    return outTensor[0,0].to(torch.uint16).cpu().numpy(), complete_model_response[0]



def fill_masks_from_retval_struct(reference_size_tensor, fill_this_tensor, model_complete_ret_val,
                                  do_binary_output = False, mask_confidence_threshold = 0.5):
    """
    Possibly resizes the 'fill_this_tensor' to the size of 'reference_size_tensor', and returns
    the filled-in tensor -- be it the original or the resized one.

    It works only with 4-dimensional [B,1,H,W] tensors. It is further assuming
    that len(model_complete_ret_val) == B. It then visits all mask images in the
    'model_complete_ret_val[b]['masks'], and virtually (nothing is saved) thresholds
    them against 'mask_confidence_threshold' and the result is max'ed into the
    output filled-in tensor.
    """

    # make sure the output tensor is of the same size as its reference...
    if not fill_this_tensor.shape == reference_size_tensor.shape:
        fill_this_tensor = torch.empty_like(reference_size_tensor, device=device) # dtype=troch.uint16

    #... and initiated
    fill_this_tensor[:] = 0

    # for each input/result in the batch:
    for idx_in_batch in range(reference_size_tensor.shape[0]):
        out_mask = fill_this_tensor[idx_in_batch]
        in_masks = model_complete_ret_val[idx_in_batch]['masks']

        for idx_in_found_labels in range(in_masks.shape[0]):
            in_mask = in_masks[idx_in_found_labels]

            val = 1 if do_binary_output else idx_in_found_labels+1
            out_mask[ in_mask > mask_confidence_threshold ] = val
            # TODO: maybe some sanity checks on bbox sizes, overwriting pixels or alike

    return fill_this_tensor

