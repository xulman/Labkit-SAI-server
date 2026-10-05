import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Will use: {device}")

def create_official_model(pretrained_model = 'facebook/sam2.1-hiera-small'):
    return SAM2ImagePredictor.from_pretrained(pretrained_model, device=device)


def set_image_and_segment(model,
        img2d: np.ndarray,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
        res_confidence_threshold: float = 0.5):
    set_image(model, img2d)
    return segment_box(x_min,y_min,x_max,y_max, res_confidence_threshold)


def set_image(model, img2d: np.ndarray, intensity_range=None) -> None:
    if intensity_range is None:
        intensity_range = (img2d.min(), img2d.max())
    w = intensity_range[1] - intensity_range[0]
    gray = np.empty(img2d.shape, dtype='uint8')
    gray[:] = (256 * (img2d[:]-intensity_range[0])) // w
    rgb = np.array([gray,gray,gray], dtype='uint8')
    model.set_image( rgb )


def segment_box(model,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
        res_confidence_threshold: float = 0.5):

    x0, x1 = sorted((x_min, x_max))
    y0, y1 = sorted((y_min, y_max))
    if x0 < 0 or y0 < 0:
        raise RuntimeError("Prompt starts at a negative coordinate.")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Prompt is a degenerate box.")

    box = np.array([x0, y0, x1, y1], dtype=np.float32)
    with torch.inference_mode():
        masks, scores, logits = model.predict(box=box, multimask_output=False)

    mask = masks[0] > res_confidence_threshold
    return mask

