from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def to_pil_image(data: Image.Image | np.ndarray | dict | Any) -> Image.Image:
    """Convert common Gradio Sketchpad outputs to PIL.Image.

    Handles:
    - PIL.Image.Image as-is
    - numpy arrays (H, W, 3) uint8
    - dict with key 'composite' (numpy array) or 'image'
    """
    if isinstance(data, Image.Image):
        return ensure_rgb(data)
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        return Image.fromarray(data.astype(np.uint8), mode="RGB")
    if isinstance(data, dict):
        # Gradio may return {'composite': np.ndarray, ...}
        if "composite" in data and isinstance(data["composite"], np.ndarray):
            return to_pil_image(data["composite"])
        if "image" in data and isinstance(data["image"], np.ndarray | Image.Image):
            return to_pil_image(data["image"])
    # Fallback: try to construct via PIL
    return ensure_rgb(Image.fromarray(np.array(data)))


def resize_to_square(img: Image.Image, size: int = 512) -> Image.Image:
    return ensure_rgb(img).resize((size, size), Image.BICUBIC)


def scribble_preprocess(img: Image.Image | np.ndarray | dict | Any, size: int = 512) -> Image.Image:
    """Convert an input sketch into ControlNet-friendly scribble.

    Produces a white background with black line art at the target size.
    """
    pil_img = to_pil_image(img)
    img_resized = resize_to_square(pil_img, size)
    np_img = np.array(img_resized)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    # Invert edges so lines are black (0) on white (255)
    edges_uint8 = edges.astype(np.uint8)
    scribble = 255 - edges_uint8
    scribble_rgb = cv2.cvtColor(scribble, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(scribble_rgb)
