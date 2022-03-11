def test_draft_model(draft_model, draft_batch):
    line, hint, color = draft_batch
    x = draft_model(line, hint)
    assert list(x.shape) == list(color.shape)


def test_disc_model(draft_model, disc, draft_batch):
    line, hint, color = draft_batch
    x = draft_model(line, hint)
    _logits = disc(x)
    logits = disc(color)
    assert list(_logits.shape) == list(logits.shape)


def test_colorization_model(colorization_model, colorization_batch):
    line, line_draft, hint, color = colorization_batch
    draft = color
    _color = colorization_model(line, draft)
    assert list(_color.shape) == list(color.shape)
