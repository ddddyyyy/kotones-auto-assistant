from types import SimpleNamespace
from typing import cast

from kaa.tasks.produce.new.strategies import standard
from kaa.tasks.produce.new.controller import ProduceController
from kaa.tasks.produce.new.page import ExamContext


def test_failed_final_exam_keeps_live_cover_flow(monkeypatch):
    settled: list[bool] = []
    aborted: list[bool] = []
    button = SimpleNamespace(click=lambda: None)
    resources = SimpleNamespace(
        Common=SimpleNamespace(ButtonNext=SimpleNamespace(wait=lambda: button)),
        InProduce=SimpleNamespace(
            TextRechallengeEndProduce=SimpleNamespace(
                try_wait=lambda **_kwargs: object(),
            ),
        ),
    )
    monkeypatch.setattr(standard, 'R', resources)
    monkeypatch.setattr(standard, '_build_battle_strategy', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(standard, 'do_cards', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(standard, 'sleep', lambda *_args: None)
    monkeypatch.setattr(standard, 'Loop', lambda *_args, **_kwargs: ())
    monkeypatch.setattr(standard, 'device', SimpleNamespace(click=lambda *_args: None))
    monkeypatch.setattr(standard, 'produce_end', lambda *, has_live: settled.append(has_live))
    controller = SimpleNamespace(page=None, abort=lambda: aborted.append(True))
    strategy = standard.StandardStrategy(cast(ProduceController, controller))
    ctx = SimpleNamespace(is_final_exam=lambda: True)

    strategy.on_exam_entered(cast(ExamContext, ctx))

    assert settled == [True]
    assert aborted == [True]
