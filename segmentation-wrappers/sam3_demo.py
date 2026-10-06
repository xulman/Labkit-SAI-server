"""
SAM 3 box-prompted segmentation of a 2D numpy image, mirroring sam2_demo.py.

Same two-step contract:
    seg = Sam3BoxSegmenter().set_image(img2d)    # 1) bootstrap (runs the encoder once)
    mask = seg.segment_box(x0, y0, x1, y1)       # 2a) SAM2-style: this one object
    labels = seg.segment_similar(x0, y0, x1, y1) # 2b) SAM3-only: all objects like it

Install (Python 3.12, recent PyTorch):
    git clone https://github.com/facebookresearch/sam3.git && pip install -e ./sam3
    hf auth login        # weights facebook/sam3 are gated; Meta SAM License
"""
import contextlib

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Will use: {device}")


def to_rgb_uint8(img: np.ndarray, intensity_range=None, percentiles=(1.0, 99.8)) -> np.ndarray:
    """2D array of any dtype -> HxWx3 uint8. Same as in sam2_demo.py."""
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


class Sam3BoxSegmenter:
    def __init__(self, checkpoint_path: str | None = None):
        # checkpoint_path: a local sam3.pt to skip the Hugging Face download
        self.model = build_sam3_image_model(
            device=device.type,
            enable_inst_interactivity=True,      # needed for SAM1/SAM2-style predict_inst()
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
        )
        self.processor = Sam3Processor(self.model)
        self.state = None
        self.shape: tuple[int, int] | None = None

    # -- helpers -------------------------------------------------------------
    def _ctx(self):
        if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _clip_box(self, x_min, y_min, x_max, y_max):
        if self.shape is None:
            raise RuntimeError("call set_image() first")
        h, w = self.shape
        x0, x1 = np.clip(sorted((float(x_min), float(x_max))), 0, w - 1)
        y0, y1 = np.clip(sorted((float(y_min), float(y_max))), 0, h - 1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("degenerate box after clipping to the image")
        return x0, y0, x1, y1

    # -- entry point 1 -------------------------------------------------------
    def set_image(self, img2d: np.ndarray, intensity_range=None) -> "Sam3BoxSegmenter":
        """Encode the image. Call once per image; all prompts then reuse the embedding."""
        rgb = Image.fromarray(to_rgb_uint8(img2d, intensity_range))
        with torch.inference_mode(), self._ctx():
            self.state = self.processor.set_image(rgb)
        self.shape = img2d.shape
        return self

    # -- entry point 2a: one box -> one object (drop-in for the SAM2 version) --
    def segment_box(self, x_min, y_min, x_max, y_max, min_score: float = 0.0) -> np.ndarray:
        """Pixel coordinates, x = column, y = row (i.e. img[y, x]).
        Returns a uint16 0/1 mask; all zeros if SAM3's IoU estimate < min_score."""
        box = np.array(self._clip_box(x_min, y_min, x_max, y_max), dtype=np.float32)
        with torch.inference_mode(), self._ctx():
            masks, scores, _ = self.model.predict_inst(
                self.state, box=box[None, :], multimask_output=False
            )
        masks, scores = np.asarray(masks), np.asarray(scores)
        if float(scores.reshape(-1)[0]) < min_score:
            return np.zeros(self.shape, dtype=np.uint16)
        return (masks.reshape(-1, *self.shape)[0] > 0.5).astype(np.uint16)

    # -- entry point 2b: box as an exemplar -> all similar objects --------------
    def segment_similar(self, x_min, y_min, x_max, y_max, min_score: float = 0.5) -> np.ndarray:
        """Box is a visual example of a concept; returns a uint16 instance label image
        (0 = background, 1..N = objects), largest objects painted first."""
        x0, y0, x1, y1 = self._clip_box(x_min, y_min, x_max, y_max)
        h, w = self.shape
        # this API wants normalised [cx, cy, w, h]
        norm_box = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
        with torch.inference_mode(), self._ctx():
            self.processor.reset_all_prompts(self.state)   # forget previous exemplars/text
            out = self.processor.add_geometric_prompt(state=self.state, box=norm_box, label=True)

        masks = out["masks"].reshape(-1, h, w).cpu().numpy()
        scores = out["scores"].float().cpu().numpy().reshape(-1)
        keep = np.flatnonzero(scores >= min_score)
        keep = keep[np.argsort([-masks[i].sum() for i in keep])]

        labels = np.zeros((h, w), dtype=np.uint16)
        for lbl, i in enumerate(keep, start=1):
            labels[masks[i] > 0] = lbl
        return labels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:512, 0:512]
    img = rng.normal(1000, 50, (512, 512))
    for cx, cy, r in [(150, 200, 50), (380, 330, 45), (300, 100, 48), (100, 420, 52)]:
        img += 3000 * (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2)
    img = img.astype(np.uint16)

    seg = Sam3BoxSegmenter().set_image(img)
    m = seg.segment_box(90, 140, 210, 260)
    print(f"segment_box:     area={int(m.sum())}")
    lab = seg.segment_similar(90, 140, 210, 260)
    print(f"segment_similar: {int(lab.max())} objects found")
