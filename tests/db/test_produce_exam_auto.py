import sqlite3
from unittest.mock import patch

from kaa.db.produce_exam_auto import (
    AUTO_PLAY_TYPE,
    get_auto_effect_evaluation,
    get_auto_play_card_evaluation,
    get_auto_trigger_evaluation,
    normalize_remaining_term,
)


def setup_function():
    get_auto_play_card_evaluation.cache_clear()
    get_auto_effect_evaluation.cache_clear()
    get_auto_trigger_evaluation.cache_clear()


def test_normalize_remaining_term():
    assert normalize_remaining_term(None) == 7
    assert normalize_remaining_term(0) == 1
    assert normalize_remaining_term(4) == 4
    assert normalize_remaining_term(20) == 7


def test_override_evaluation_takes_precedence():
    override = {
        'type': AUTO_PLAY_TYPE,
        'produceCardId': 'card-1',
        'remainingTerm': 3,
        'evaluation': 900,
    }
    with patch('kaa.db.produce_exam_auto.select', return_value=override) as query:
        result = get_auto_play_card_evaluation('card-1', 3)

    assert result is not None
    assert result.evaluation == 900
    assert result.is_override
    assert query.call_count == 1


def test_generic_evaluation_is_fallback():
    generic = {
        'produceCardId': 'card-2',
        'remainingTerm': 7,
        'evaluation': 120,
    }
    with patch('kaa.db.produce_exam_auto.select', side_effect=[None, generic]):
        result = get_auto_play_card_evaluation('card-2', 99)

    assert result is not None
    assert result.remaining_term == 7
    assert result.evaluation == 120
    assert not result.is_override


def test_missing_evaluation_returns_none():
    with patch('kaa.db.produce_exam_auto.select', side_effect=[None, None]):
        assert get_auto_play_card_evaluation('missing', 2) is None


def test_old_database_without_auto_tables_falls_back(caplog):
    with patch(
        'kaa.db.produce_exam_auto.select',
        side_effect=sqlite3.OperationalError('no such table'),
    ):
        assert get_auto_play_card_evaluation('card-1', 2) is None
    assert 'unavailable' in caplog.text


def test_effect_evaluation_reads_archetype_and_turn_weight():
    row = {
        'examEffectType': 'ProduceExamEffectType_ExamFullPower',
        'remainingTerm': 7,
        'evaluationType': (
            'ProduceExamAutoEvaluationType_ExamFullPowerPointAdditive'
        ),
        'evaluation': 16486,
        'examStatusEnchantCoefficientPermil': 1000,
    }
    with patch('kaa.db.produce_exam_auto.select', return_value=row):
        result = get_auto_effect_evaluation(
            row['examEffectType'],
            7,
            row['evaluationType'],
        )

    assert result is not None
    assert result.evaluation == 16486
    assert result.status_enchant_coefficient_permil == 1000


def test_trigger_evaluation_reads_expected_value_coefficient():
    row = {
        'examStatusEnchantProduceExamTriggerId': 'trigger-1',
        'coefficientPermil': 650,
        'count': 2,
    }
    with patch('kaa.db.produce_exam_auto.select', return_value=row):
        result = get_auto_trigger_evaluation('trigger-1')

    assert result is not None
    assert result.coefficient_permil == 650
    assert result.count == 2
