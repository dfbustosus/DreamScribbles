import gradio as gr  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

from ..config import get_settings
from ..inference.pipeline import DreamScribblesPipeline
from ..styles import NEGATIVE_PROMPT_DEFAULT, build_prompt, list_styles
from ..utils.image_ops import scribble_preprocess


def launch_minimal(settings=None):
    settings = settings or get_settings()
    pipe = DreamScribblesPipeline(settings)

    def generate_image(
        img, subject_text, style_choice, negative_text, steps_v, guidance_v, strength_v, seed_v
    ):
        if img is None:
            return None, None

        # Handle different image formats
        if isinstance(img, str):
            # Base64 or file path
            if img.startswith("data:image"):
                import base64
                import io

                img_data = img.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
            else:
                img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype("uint8"), "RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")

        control = scribble_preprocess(img)
        prompt = build_prompt(subject_text, style_choice)

        out = pipe.generate(
            control_image=control,
            prompt=prompt,
            negative_prompt=negative_text or None,
            num_inference_steps=int(steps_v),
            guidance_scale=float(guidance_v),
            seed=int(seed_v) if seed_v is not None else -1,
            controlnet_conditioning_scale=float(strength_v),
        )
        return control, out

    # Create interface
    iface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Image(tool="sketch", label="Scribble"),
            gr.Textbox(label="What are you drawing?", value="a cute house with a garden"),
            gr.Dropdown(choices=list_styles(), value=list_styles()[0], label="Style"),
            gr.Textbox(label="Negative prompt", value=NEGATIVE_PROMPT_DEFAULT),
            gr.Slider(5, 60, value=30, step=1, label="Steps"),
            gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance scale"),
            gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Control strength"),
            gr.Number(value=-1, precision=0, label="Seed (-1 for random)"),
        ],
        outputs=[gr.Image(label="Preprocessed Scribble"), gr.Image(label="Generated Image")],
        title="DreamScribbles: The AI Art Transformer 🎨",
        description=(
            "Draw a scribble, pick a style, and generate a stunning image "
            "with ControlNet + Stable Diffusion."
        ),
    )

    iface.launch(server_port=settings.port, share=False)
