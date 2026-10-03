import cv2
import numpy as np
from types import SimpleNamespace
from typing import Any, cast

from kotonebot.primitives import Rect

from kaa.db.constants import ProduceExamEffectType, ProducePlanType
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.shared.cards import _order_move_options
from kaa.tasks.produce.new.play_cards.move_selection import (
    find_move_scroll_thumb,
    recognize_move_cards,
)
from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState, BattleState
from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, EvaluationConfidence


def _reference_options():
    screen = cv2.imread(
        'kotonebot-resource/sprites/jp/in_produce/'
        'screenshot_skill_card_move_multiple.png'
    )
    assert screen is not None
    letters = [
        Rect(x, y, 17, 15)
        for x, y in (
            (132, 498), (278, 498), (425, 498), (571, 498),
            (132, 645), (278, 645),
        )
    ]
    return recognize_move_cards(screen, letters + [Rect(132, 498, 17, 15)])


def test_move_dialog_recognizes_distinct_cards_and_upgrade():
    options = _reference_options()

    assert len(options) == 6
    assert [option.card._id if option.card else None for option in options] == [
        'p_card-00-act-0_001',
        'p_card-03-act-0_021',
        'p_card-03-act-1_042',
        'p_card-03-men-0_013',
        'p_card-03-men-0_013',
        'p_card-03-men-0_015',
    ]
    assert options[2].card is not None
    assert options[2].card.upgrade_count == 1


def test_full_power_holds_payoff_cards_before_basic_cards():
    options = _reference_options()
    planner = PlannerStrategy()
    planner._last_move_state = BattleState(
        remaining_turns=4,
        hp=20,
        genki=10,
        archetype=ProduceExamEffectType.ExamFullPower,
        plan_type=ProducePlanType.Plan3,
        anomaly=AnomalyState(full_power_points=8),
    )

    ranked = planner.rank_move_cards([option.card for option in options])

    assert ranked is not None
    assert ranked[:2] == [2, 1]
    ordered = _order_move_options(options, planner)
    assert ordered[:2] == [options[2], options[1]]


def test_hold_ranking_waits_for_observed_battle_state():
    assert PlannerStrategy().rank_move_cards([None]) is None


def test_full_power_hold_prioritizes_payoff_over_high_setup_prior():
    anomaly = AnomalyState(full_power_points=8, cumulative_full_power_points=8)
    planner = PlannerStrategy()
    planner._memory.anomaly = anomaly
    planner._last_move_state = BattleState(
        remaining_turns=5,
        hp=20,
        genki=15,
        archetype=ProduceExamEffectType.ExamFullPower,
        plan_type=ProducePlanType.Plan3,
        anomaly=anomaly,
    )
    setting = SkillCard.from_asset_id('img_general_skillcard_men-1_051', 1)
    spur = SkillCard.from_asset_id('img_general_skillcard_act-0_021', 0)
    assert setting is not None and spur is not None

    assert planner.rank_move_cards([setting, spur]) == [1, 0]


def test_move_order_falls_back_when_no_strategy_state_is_available():
    options = _reference_options()
    assert _order_move_options(options, PlannerStrategy())[0] is options[-1]


def test_move_scroll_thumb_is_detected_without_confusing_the_track():
    screen = np.full((1280, 720, 3), 240, dtype=np.uint8)
    screen[510:1095, 683] = (146, 143, 134)
    screen[1000:1090, 683] = (129, 128, 132)

    assert find_move_scroll_thumb(screen) == 1045
    assert find_move_scroll_thumb(np.full_like(screen, 240)) is None


def test_hold_does_not_prioritize_card_unaffordable_next_turn():
    class Evaluator:
        def evaluate(self, card, state):
            return CardEvaluation(
                card_id=card._id,
                score=100 if card._id == 'expensive' else 10,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test',
            )

    planner = PlannerStrategy(cast(Any, Evaluator()))
    planner._last_move_state = BattleState(remaining_turns=3, hp=5, genki=0)
    expensive = SimpleNamespace(_id='expensive', name='expensive', stamina=10, force_stamina=0)
    affordable = SimpleNamespace(_id='affordable', name='affordable', stamina=2, force_stamina=0)

    assert planner.rank_move_cards(cast(Any, [expensive, affordable])) == [1, 0]
