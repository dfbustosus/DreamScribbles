from __future__ import annotations

STYLE_PRESETS: dict[str, str] = {
    "Pixar": "whimsical 3D animated film aesthetic, soft lighting, cinematic framing, high detail",
    "Watercolor": "watercolor painting, soft brush strokes, paper texture, gentle gradients",
    "Photorealistic": (
        "ultra-detailed, photorealistic, natural lighting, 35mm lens, high dynamic range"
    ),
    "Anime": "cel-shaded animated aesthetic, clean lines, vibrant colors, studio-quality finish",
    "Oil Painting": "oil painting on canvas, impasto texture, rich colors, baroque lighting",
    "Cyberpunk": "cyberpunk aesthetic, neon lights, rain, reflective surfaces, futuristic city",
    "Fantasy": "high fantasy art, ethereal, magical atmosphere, epic composition",
    "Neon": "neon-glow aesthetic, vibrant rim lights, dark background, electric colors",
    "Sketch": "pencil sketch style, cross-hatching, paper texture, monochrome",
    "Low Poly": "low poly 3D render, simplistic facets, geometric style, minimalistic shading",
}


NEGATIVE_PROMPT_DEFAULT = (
    "low quality, worst quality, artifacts, blurry, distorted, deformed, jpeg artifacts"
)


def list_styles() -> list[str]:
    return list(STYLE_PRESETS.keys())


def build_prompt(subject: str, style: str) -> str:
    style_suffix = STYLE_PRESETS.get(style, "")
    subject_clean = subject.strip() if subject else "a beautiful scene"
    if style_suffix:
        return f"{subject_clean}, {style_suffix}"
    return subject_clean
