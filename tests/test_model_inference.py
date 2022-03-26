def test_draft_model(draft_model, draft_batch):
    line, hint, color = draft_batch
    input_dict = {"line": line, "hint": hint, "training": True}
    x = draft_model(input_dict)
    assert list(x.shape) == list(color.shape)


def test_disc_model(draft_model, disc, draft_batch):
    line, hint, color = draft_batch
    input_dict = {"line": line, "hint": hint, "training": True}
    x = draft_model(input_dict)
    _logits = disc(x)
    logits = disc(color)
    assert list(_logits.shape) == list(logits.shape)


def test_colorization_model(colorization_model, colorization_batch):
    line, line_draft, hint, color = colorization_batch
    draft = color
    input_dict = {"line": line, "hint": draft, "training": True}
    _color = colorization_model(input_dict)
    assert list(_color.shape) == list(color.shape)
