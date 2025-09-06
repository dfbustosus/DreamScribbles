from PIL import Image, ImageDraw

from dreamscribbles.utils.image_ops import scribble_preprocess


def test_scribble_preprocess_returns_image():
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((10, 10, 200, 200), fill=(0, 0, 0), width=5)
    out = scribble_preprocess(img)
    assert isinstance(out, Image.Image)
    assert out.size == (512, 512)
    assert out.mode == "RGB"
