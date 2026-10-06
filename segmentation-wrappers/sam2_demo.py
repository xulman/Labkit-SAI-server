import contextlib
import numpy as np
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Will use: {device}")


def to_rgb_uint8(img: np.ndarray, intensity_range=None, percentiles=(1.0, 99.8)) -> np.ndarray:
    """2D array of any dtype -> HxWx3 uint8 (SAM was trained on 8-bit RGB).

    intensity_range=(lo, hi) fixes the mapping, e.g. to keep a time-lapse
    consistent; otherwise per-image percentiles are used.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {img.shape}")
    if img.dtype == np.uint8 and intensity_range is None:
        g = img
    else:
        lo, hi = intensity_range if intensity_range is not None else np.percentile(img, percentiles)
        g = np.clip((img.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        g = (g * 255.0 + 0.5).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


class Sam2BoxSegmenter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small"):
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        self.shape: tuple[int, int] | None = None

    # -- helpers -------------------------------------------------------------
    def _ctx(self):
        if "cuda" in device.type:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # -- entry point 1 -------------------------------------------------------
    def set_image(self, img2d: np.ndarray, intensity_range=None) -> None:
        """Encode the image. Call once per image; all boxes then reuse the embedding."""
        rgb = to_rgb_uint8(img2d, intensity_range)
        with torch.inference_mode(), self._ctx():
            self.predictor.set_image(rgb)
        self.shape = img2d.shape
        return self

    # -- entry point 2 -------------------------------------------------------
    def segment_box(self, x_min: float, y_min: float, x_max: float, y_max: float, res_confidence_threshold: float = 0.5) -> np.ndarray:
        """Segment the single object inside the box.

        Pixel coordinates, x = column, y = row (i.e. img[y, x]).
        """
        if self.shape is None:
            raise RuntimeError("call set_image() first")
        h, w = self.shape
        x0, x1 = sorted((float(x_min), float(x_max)))
        y0, y1 = sorted((float(y_min), float(y_max)))
        x0, x1 = np.clip([x0, x1], 0, w - 1)
        y0, y1 = np.clip([y0, y1], 0, h - 1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("degenerate box after clipping to the image")

        box = np.array([x0, y0, x1, y1], dtype=np.float32)
        with torch.inference_mode(), self._ctx():
            masks, scores, logits = self.predictor.predict(box=box, multimask_output=False)

        # predict() returns float 0/1 masks
        # this turns <res_confidence_threshold to 0.0, else to 1.0
        # (floor() is perhaps not needed and would happen as part of astype())
        mask = np.floor(masks[0] + (1.0-res_confidence_threshold)).astype('uint16')
        return mask

