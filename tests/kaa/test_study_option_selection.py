from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

from kaa.tasks.produce.new.strategies.standard import StandardStrategy, _study_gain


def _choose(*descriptions: str) -> int:
    context = SimpleNamespace(
        is_self_study=lambda: False,
        fetch_options=lambda: [SimpleNamespace(description=text) for text in descriptions],
        commit=Mock(),
    )
    strategy = StandardStrategy.__new__(StandardStrategy)

    strategy.on_study(cast(Any, context))

    context.commit.assert_called_once()
    return context.commit.call_args.args[0]


def test_study_gain_accepts_full_width_symbols_and_digits():
    assert _study_gain('ダンス上昇＋５０カードを獲得') == 50
    assert _study_gain('何もせず戻る') is None


def test_study_keeps_existing_plus_thirty_preference():
    assert _choose('ダンス上昇+50', 'ボーカル上昇+30', 'ビジュアル上昇+25') == 1


def test_study_uses_largest_readable_gain_when_plus_thirty_is_absent():
    assert _choose('ダンス上昇+25', 'ボーカル上昇+40', 'ビジュアル上昇+50') == 2


def test_study_uses_safe_index_when_ocr_cannot_read_gains():
    assert _choose('候補A') == 0
    assert _choose('候補A', '候補B', '候補C') == 1
