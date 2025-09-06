from dreamscribbles.styles import STYLE_PRESETS, build_prompt, list_styles


def test_style_presets_non_empty():
    assert isinstance(STYLE_PRESETS, dict)
    assert len(STYLE_PRESETS) >= 5


def test_list_styles_contains_known():
    styles = list_styles()
    assert "Pixar" in styles


def test_build_prompt_uses_style():
    p = build_prompt("a castle", "Pixar")
    assert "castle" in p
    assert "Pixar" not in p  # style text is descriptive, not literal name
    assert "," in p
