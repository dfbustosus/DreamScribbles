from __future__ import annotations

import os
from typing import Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DevicePref = Literal["cuda", "mps", "cpu"] | None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="dreamscribbles_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    hf_token: str | None = Field(default=None)
    model_id: str = Field(default="runwayml/stable-diffusion-v1-5")
    controlnet_id: str = Field(default="lllyasviel/sd-controlnet-scribble")

    device_preference: DevicePref = Field(default=None)
    port: int = Field(default=int(os.getenv("PORT", 7860)))
    share: bool = Field(default=False)

    disable_safety_checker: bool = Field(default=False)

    @field_validator("device_preference", mode="before")
    @classmethod
    def _normalize_device_preference(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip().lower()
            if v == "":
                return None
            if v in {"cuda", "mps", "cpu"}:
                return v
        return None

    def pick_device(self) -> torch.device:
        # Honor explicit preference if available and supported
        pref = (self.device_preference or "").lower() if self.device_preference else ""
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
        # Auto-detect best device
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def pick_dtype(self, device: torch.device) -> torch.dtype:
        if device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32


def get_settings() -> Settings:
    return Settings()
