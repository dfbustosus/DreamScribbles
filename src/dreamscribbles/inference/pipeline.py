from __future__ import annotations

import importlib
import os

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from ..config import Settings


class DreamScribblesPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.pick_device()
        self.dtype = settings.pick_dtype(self.device)

        # Propagate HF token to environment if provided
        if settings.hf_token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)
            os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        controlnet = ControlNetModel.from_pretrained(
            settings.controlnet_id, torch_dtype=self.dtype, use_safetensors=True
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            settings.model_id,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        # Disable safety checker only if requested
        if settings.disable_safety_checker:
            self.pipe.safety_checker = None

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Memory/perf optimizations
        self.pipe.enable_attention_slicing()
        if self.device.type == "cuda":
            try:
                importlib.import_module("xformers")
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    @torch.inference_mode()
    def generate(
        self,
        control_image,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        controlnet_conditioning_scale: float = 1.0,
    ):
        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=str(self.device)).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        return result.images[0]
