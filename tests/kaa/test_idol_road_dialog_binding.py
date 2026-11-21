from kaa.idol_road.dialog_binding import FormationDialogBinding


def test_residual_formation_keeps_pending_open_identity():
    binding = FormationDialogBinding()
    binding.open('hski', 13)
    binding.on_formation('hski', 13)
    assert binding.identity == ('hski', 13)


def test_close_keeps_identity_for_residual_dialog_then_invalidates_on_formation():
    binding = FormationDialogBinding()
    binding.open('hski', 13)
    binding.close()
    assert binding.identity == ('hski', 13)
    binding.on_formation('hski', 13)
    assert binding.identity is None


def test_new_formation_or_explicit_clear_invalidates_binding():
    binding = FormationDialogBinding()
    binding.open('hski', 13)
    binding.on_formation('ttmr', 16)
    assert binding.identity is None
    binding.open('ttmr', 16)
    binding.clear()
    assert binding.identity is None
