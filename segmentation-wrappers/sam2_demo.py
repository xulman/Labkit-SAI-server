"""
SAM 2.1 (hiera-small) box-prompted segmentation of a 2D numpy image.

Two entry points:
    seg = Sam2BoxSegmenter()                 # loads the model once
    seg.set_image(img2d)                     # 1) bootstrap on an image (runs the encoder)
    res = seg.segment_box(x0, y0, x1, y1)    # 2) one box -> one mask (cheap, repeatable)

Install (Apache-2.0 code + weights):
    pip install torch torchvision          # pick the wheel matching your CUDA
    pip install "git+https://github.com/facebookresearch/sam2.git" huggingface_hub numpy
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class BoxResult:
    mask: np.ndarray        # (H, W) bool, same shape as the input image
    score: float            # model's own IoU estimate for the mask, ~[0, 1]
    bbox: tuple | None      # tight (x_min, y_min, x_max, y_max) of the mask, None if empty
    logits: np.ndarray      # (256, 256) float32 low-res logits, reusable for refinement


class Sam2BoxSegmenter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=self.device)

    # -- helpers -------------------------------------------------------------
    def _ctx(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @staticmethod
    def _to_rgb_uint8(img: np.ndarray, intensity_range=None, percentiles=(1.0, 99.8)) -> np.ndarray:
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

    # -- entry point 1 -------------------------------------------------------
    def set_image(self, img2d: np.ndarray, intensity_range=None) -> None:
        """Encode the image. Call once per image; all boxes then reuse the embedding."""
        rgb = self._to_rgb_uint8(img2d, intensity_range)
        with torch.inference_mode(), self._ctx():
            self.predictor.set_image(rgb)
        self.shape = img2d.shape

    # -- entry point 2 -------------------------------------------------------
    def segment_box(self, x_min: float, y_min: float, x_max: float, y_max: float) -> BoxResult:
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

        mask = masks[0] > 0.5            # predict() returns float 0/1 masks
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size else None
        return BoxResult(mask=mask, score=float(scores[0]), bbox=bbox, logits=logits[0])


if __name__ == "__main__":
    # synthetic test: two bright blobs on a noisy 16-bit background
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:512, 0:512]
    img = rng.normal(1000, 50, (512, 512))
    img += 3000 * (((xx - 150) ** 2 + (yy - 200) ** 2) < 60 ** 2)
    img += 3000 * (((xx - 380) ** 2 + (yy - 330) ** 2) < 40 ** 2)
    img = img.astype(np.uint16)

    seg = Sam2BoxSegmenter()
    seg.set_image(img)
    for box in [(80, 130, 220, 270), (330, 280, 430, 380)]:
        r = seg.segment_box(*box)
        print(f"box={box} score={r.score:.3f} area={r.mask.sum()} bbox={r.bbox}")
