from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kaa.db.constants import ProduceExamEffectType, ProducePlanType
from kaa.db.skill_card import SkillCard
from kaa.tasks.produce.new.play_cards.planner import evaluator as evaluator_module
from kaa.tasks.produce.new.play_cards.planner import memory as memory_module
from kaa.tasks.produce.new.play_cards.planner.evaluator import (
    CardEvaluation,
    CompositeEvaluator,
    EffectHeuristicEvaluator,
    EvaluationConfidence,
    conservative_target_output,
)
from kaa.tasks.produce.new.play_cards.planner.memory import BattleMemory
from kaa.tasks.produce.new.play_cards.planner.profiles import PROFILES, profile_for
from kaa.tasks.produce.new.play_cards.planner.state import (
    AnomalyState,
    AnomalyStance,
    BattleState,
    ExamBuffState,
)
from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict


def _effect(
    kind: ProduceExamEffectType,
    value: int,
    turn: int = 0,
    value2: int = 0,
    count: int = 1,
):
    return SimpleNamespace(
        produce_exam_effect=SimpleNamespace(
            effect_type=kind,
            effect_value1=value,
            effect_value2=value2,
            effect_count=count,
            effect_turn=turn,
        )
    )


def _card(
    card_id: str,
    *effects,
    cost: int = 0,
    move: str = 'Grave',
    plan: ProducePlanType = ProducePlanType.Common,
    cost_type: str | None = None,
    cost_value: int = 0,
):
    return SimpleNamespace(
        _id=card_id,
        stamina=cost,
        force_stamina=0,
        plan_type=plan.value,
        cost_type=cost_type,
        cost_value=cost_value,
        evaluation=0,
        play_effects=list(effects),
        play_move_position_type=f'ProduceCardMovePositionType_{move}',
    )


def test_immediate_output_wins_late_but_setup_scales_with_remaining_turns():
    evaluator = EffectHeuristicEvaluator()
    output = _card('output', _effect(ProduceExamEffectType.ExamLesson, 9), cost=4)
    setup = _card('setup', _effect(ProduceExamEffectType.ExamLessonBuff, 4), cost=4)

    late = BattleState(remaining_turns=1, hp=20, genki=0)
    early = BattleState(remaining_turns=7, hp=20, genki=0)

    assert evaluator.evaluate(cast(Any, output), late).score > evaluator.evaluate(cast(Any, setup), late).score
    assert evaluator.evaluate(cast(Any, setup), early).score > evaluator.evaluate(cast(Any, setup), late).score


@pytest.mark.parametrize('stance', [AnomalyStance.NORMAL, AnomalyStance.PRESERVATION_2])
def test_observed_strong_stance_corrects_memory_without_inventing_transition_rewards(stance):
    memory = BattleMemory(anomaly=AnomalyState(stance=stance, enthusiasm=3))
    memory.reconcile_effect_values({'concentration': None})
    assert memory.anomaly.stance == AnomalyStance.AGGRESSIVE
    assert memory.anomaly.enthusiasm == 3
    assert memory.anomaly.stance_change_count == 0
    assert memory.anomaly.concentration_change_count == 0
    memory.reconcile_effect_values({})
    memory.reconcile_effect_values({})
    assert memory.anomaly.stance == AnomalyStance.AGGRESSIVE


def test_observed_strong_stance_does_not_downgrade_inferred_super_strong():
    memory = BattleMemory(anomaly=AnomalyState(stance=AnomalyStance.AGGRESSIVE_2))
    memory.reconcile_effect_values({'concentration': None})
    assert memory.anomaly.stance == AnomalyStance.AGGRESSIVE_2


def test_hud_stance_correction_changes_output_and_affordability():
    from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play

    memory = BattleMemory()
    memory.reconcile_effect_values({'concentration': None})
    card = cast(Any, _card('output', _effect(ProduceExamEffectType.ExamLesson, 9), cost=4))
    before = BattleState(remaining_turns=3, hp=5, genki=1)
    after = BattleState(remaining_turns=3, hp=5, genki=1, anomaly=memory.anomaly)
    evaluator = EffectHeuristicEvaluator()
    assert evaluator.evaluate(card, before).confirmed_output == 9
    assert evaluator.evaluate(card, after).confirmed_output == 18
    assert can_afford_play(before, card) is True
    assert can_afford_play(after, card) is False


@pytest.mark.parametrize('points', [0, 8, 13])
def test_point_observation_replaces_current_but_not_cumulative_or_activation(points):
    memory = BattleMemory(anomaly=AnomalyState(
        full_power_points=17, cumulative_full_power_points=42, full_power_pending=True,
    ))
    memory.reconcile_effect_values({'full_power_point': points})
    assert memory.anomaly.full_power_points == points
    assert memory.anomaly.full_power_pending is (points >= 10)
    assert memory.anomaly.cumulative_full_power_points == 42
    assert not memory.anomaly.full_power_active
    assert memory.anomaly.extra_play_count == 0
    memory.reconcile_effect_values({'full_power_point': None})
    memory.reconcile_effect_values({})
    assert memory.anomaly.full_power_points == points


@pytest.mark.parametrize('stance', list(AnomalyStance))
def test_full_power_hud_replaces_stale_stance_without_transition_rewards(stance):
    memory = BattleMemory(anomaly=AnomalyState(stance=stance, enthusiasm=7))
    memory.reconcile_effect_values({'full_power': None, 'concentration': None})
    assert memory.anomaly.full_power_active
    assert memory.anomaly.stance is AnomalyStance.NORMAL
    assert memory.anomaly.enthusiasm == 7
    assert memory.anomaly.stance_change_count == 0
    assert memory.anomaly.extra_play_count == 0
    memory.reconcile_effect_values({'concentration': None})
    assert memory.anomaly.stance is AnomalyStance.NORMAL


@pytest.mark.parametrize('kind', [
    ProduceExamEffectType.ExamPreservation,
    ProduceExamEffectType.ExamConcentration,
    ProduceExamEffectType.ExamStanceReset,
])
def test_full_power_blocks_card_stance_changes_but_keeps_other_effects(kind):
    memory = BattleMemory(anomaly=AnomalyState(full_power_active=True, full_power_turns=1))
    card = _card('stance', _effect(kind, 2), _effect(ProduceExamEffectType.ExamFullPowerPoint, 3))
    memory.record_play(cast(Any, card))
    assert memory.anomaly.stance is AnomalyStance.NORMAL
    assert memory.anomaly.full_power_points == 3
    assert memory.anomaly.stance_change_count == 0
    assert memory.anomaly.enthusiasm == 0
    memory.observe(2, [])
    memory.observe(1, [])
    memory.record_play(cast(Any, _card('strong', _effect(ProduceExamEffectType.ExamConcentration, 1))))
    assert memory.anomaly.stance is AnomalyStance.AGGRESSIVE


@pytest.mark.parametrize('stance', list(AnomalyStance))
def test_full_power_output_and_cost_do_not_stack_with_stale_stance(stance):
    from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play

    card = cast(Any, _card('output', _effect(ProduceExamEffectType.ExamLesson, 9), cost=4))
    anomaly = AnomalyState(full_power_active=True, stance=stance)
    state = BattleState(remaining_turns=2, hp=3, genki=2, anomaly=anomaly)
    assert EffectHeuristicEvaluator().evaluate(card, state).confirmed_output == 27
    assert can_afford_play(state, card)
    poor = BattleState(remaining_turns=2, hp=1, genki=1, anomaly=anomaly)
    assert not can_afford_play(poor, card)


def test_direct_full_power_card_replaces_previous_stance():
    memory = BattleMemory(anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION))
    memory.record_play(cast(Any, _card('power', _effect(ProduceExamEffectType.ExamFullPower, 1))))
    assert memory.anomaly.full_power_active
    assert memory.anomaly.stance is AnomalyStance.NORMAL


@pytest.mark.parametrize('elapsed', [1, 2, 3, 5])
def test_memory_turn_jump_matches_repeated_turn_transitions(elapsed):
    from copy import deepcopy

    hand = cast(Any, _card('visible'))
    memory = BattleMemory(
        anomaly=AnomalyState(full_power_points=23, full_power_pending=True),
        buffs=ExamBuffState(parameter_buff_turns=4, parameter_buff_multiple_turns=2),
    )
    memory.observe(9, [hand])
    expected = deepcopy(memory)
    for turn in range(8, 8 - elapsed, -1):
        expected.observe(turn, [])
    memory.observe(9 - elapsed, [])
    assert memory.anomaly == expected.anomaly
    assert memory.buffs == expected.buffs
    assert memory.discard == expected.discard == ['visible']
    assert memory.plays_this_turn == 0


def test_memory_turn_jump_expires_previous_full_power_and_temporary_buffs():
    memory = BattleMemory(
        anomaly=AnomalyState(full_power_points=10, full_power_pending=True),
        buffs=ExamBuffState(parameter_buff_turns=2, parameter_buff_multiple_turns=1),
    )
    memory.observe(9, [])
    memory.observe(7, [])
    assert not memory.anomaly.full_power_active
    assert memory.anomaly.full_power_points == 0
    assert memory.anomaly.full_power_change_count == 1
    assert memory.buffs.parameter_buff_turns == 0
    assert memory.buffs.parameter_buff_multiple_turns == 0


@pytest.mark.parametrize('points,pending', [(10, False), (13, True), (7, True)])
def test_full_power_already_pending_does_not_repeat_threshold_bonus(points, pending):
    evaluator = EffectHeuristicEvaluator()
    card = cast(Any, _card('charge', _effect(ProduceExamEffectType.ExamFullPowerPoint, 3)))
    crossing = BattleState(
        remaining_turns=2, hp=25, genki=30,
        anomaly=AnomalyState(full_power_points=7),
    )
    already_pending = BattleState(
        remaining_turns=2, hp=25, genki=30,
        anomaly=AnomalyState(full_power_points=points, full_power_pending=pending),
    )
    new = evaluator.evaluate(card, crossing)
    redundant = evaluator.evaluate(card, already_pending)
    assert new.score == 53.0
    assert redundant.score == pytest.approx(6.3)
    assert '不重复计启动收益' in redundant.reason


def test_full_power_threshold_bonus_requires_a_following_turn():
    evaluator = EffectHeuristicEvaluator()
    card = cast(Any, _card('charge', _effect(ProduceExamEffectType.ExamFullPowerPoint, 3)))
    state = BattleState(
        remaining_turns=1, hp=25, genki=30,
        anomaly=AnomalyState(full_power_points=7),
    )
    assert evaluator.evaluate(card, state).score == pytest.approx(0.45)


@pytest.mark.parametrize('kind', [
    ProduceExamEffectType.ExamPreservation,
    ProduceExamEffectType.ExamOverPreservation,
    ProduceExamEffectType.ExamAddGrowEffect,
])
def test_deferred_option_value_decays_and_has_no_terminal_reward(kind):
    evaluator = EffectHeuristicEvaluator()
    card = cast(Any, _card('deferred', _effect(kind, 1)))
    scores = [
        evaluator.evaluate(card, BattleState(remaining_turns=t, hp=20, genki=10)).score
        for t in (7, 3, 2, 1)
    ]
    assert scores[0] > scores[1] > scores[2] > scores[3]
    assert scores[3] == 0


@pytest.mark.parametrize('kind', [
    ProduceExamEffectType.ExamPreservation,
    ProduceExamEffectType.ExamAddGrowEffect,
])
@pytest.mark.parametrize('opportunity', ['followup', 'extra_turn'])
def test_terminal_deferred_value_retains_actual_extra_opportunities(kind, opportunity):
    evaluator = EffectHeuristicEvaluator()
    effects = [_effect(kind, 1)]
    if opportunity == 'extra_turn':
        effects.append(_effect(ProduceExamEffectType.ExamExtraTurn, 1))
    card = cast(Any, _card('deferred', *effects))
    state = BattleState(
        remaining_turns=1, hp=20, genki=10,
        playable_count=2 if opportunity == 'followup' else 1,
    )
    baseline = cast(Any, _card('baseline', *effects[1:]))
    assert evaluator.evaluate(card, state).score > evaluator.evaluate(baseline, state).score


def test_positive_preservation_prior_cannot_restore_flat_late_bonus(monkeypatch):
    monkeypatch.setattr(
        EffectHeuristicEvaluator, '_official_effect_prior',
        classmethod(lambda cls, kind, state, **kwargs: 12.0 if kind == ProduceExamEffectType.ExamPreservation else 0.0),
    )
    evaluator = EffectHeuristicEvaluator()
    # Captures the failed lesson's decision shape, not a full screenshot replay:
    # healthy HP, only three turns, a small genki/stance card vs seven output.
    setup = cast(Any, _card(
        'setup', _effect(ProduceExamEffectType.ExamPreservation, 1),
        _effect(ProduceExamEffectType.ExamBlock, 2),
    ))
    output = cast(Any, _card('output', _effect(ProduceExamEffectType.ExamLesson, 7)))
    state = BattleState(
        remaining_turns=3, hp=16, genki=2, score_gap=45,
        score_target_is_final=False, archetype=ProduceExamEffectType.ExamFullPower,
    )
    assert evaluator.evaluate(output, state).score > evaluator.evaluate(setup, state).score
    assert '后续兑现机会有限' in evaluator.evaluate(setup, state).reason


@pytest.mark.parametrize('is_exam', [False, True])
def test_unknown_hp_does_not_receive_critical_defence_bonus(is_exam: bool):
    evaluator = EffectHeuristicEvaluator()
    defence = _card('defence', _effect(ProduceExamEffectType.ExamBlock, 10))
    low = BattleState(remaining_turns=4, hp=3, genki=0, is_exam=is_exam)
    unknown = BattleState(
        remaining_turns=4, hp=0, genki=0, hp_known=False, is_exam=is_exam,
    )
    healthy = BattleState(remaining_turns=4, hp=20, genki=0, is_exam=is_exam)

    low_score = evaluator.evaluate(cast(Any, defence), low).score
    unknown_score = evaluator.evaluate(cast(Any, defence), unknown).score
    healthy_score = evaluator.evaluate(cast(Any, defence), healthy).score

    assert low_score > unknown_score > healthy_score


def test_unknown_hp_gets_intermediate_cost_penalty():
    evaluator = EffectHeuristicEvaluator()
    green = _card('green', cost=4)
    red = _card('red')
    red.force_stamina = 4
    low = BattleState(remaining_turns=4, hp=3, genki=0)
    unknown = BattleState(remaining_turns=4, hp=0, genki=0, hp_known=False)
    healthy = BattleState(remaining_turns=4, hp=20, genki=0)

    for card in (green, red):
        low_score = evaluator.evaluate(cast(Any, card), low).score
        unknown_score = evaluator.evaluate(cast(Any, card), unknown).score
        healthy_score = evaluator.evaluate(cast(Any, card), healthy).score
        assert low_score < unknown_score < healthy_score


def test_unknown_genki_does_not_count_as_sufficient_resource():
    evaluator = EffectHeuristicEvaluator()
    card = _card('green', cost=4)
    known = BattleState(remaining_turns=4, hp=20, genki=4)
    unknown = BattleState(
        remaining_turns=4, hp=20, genki=0, genki_known=False,
    )

    assert evaluator.evaluate(cast(Any, card), known).score > evaluator.evaluate(cast(Any, card), unknown).score


def test_last_play_does_not_value_setup_without_a_followup_card_play():
    evaluator = EffectHeuristicEvaluator()
    setup = _card('setup', _effect(ProduceExamEffectType.ExamLessonBuff, 20))
    final_play = BattleState(remaining_turns=1, hp=20, genki=10, playable_count=1)
    extra_play = BattleState(remaining_turns=1, hp=20, genki=10, playable_count=2)

    assert evaluator.evaluate(cast(Any, setup), final_play).score == 0
    assert evaluator.evaluate(cast(Any, setup), extra_play).score > 0


def test_last_play_still_values_end_of_turn_review_and_immediate_output():
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=1, hp=20, genki=10, playable_count=1)
    review = _card('review', _effect(ProduceExamEffectType.ExamReview, 5))
    mixed = _card(
        'mixed',
        _effect(ProduceExamEffectType.ExamLessonBuff, 20),
        _effect(ProduceExamEffectType.ExamLesson, 8),
    )

    assert evaluator.evaluate(cast(Any, review), state).score > 0
    assert evaluator.evaluate(cast(Any, mixed), state).score >= 8


def test_setup_still_has_value_when_card_grants_an_extra_turn():
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=1, hp=20, genki=10, playable_count=1)
    extra_turn = _effect(ProduceExamEffectType.ExamExtraTurn, 1)
    turn_only = _card('turn', extra_turn)
    setup_and_turn = _card(
        'setup-and-turn',
        _effect(ProduceExamEffectType.ExamLessonBuff, 20),
        extra_turn,
    )

    assert evaluator.evaluate(cast(Any, setup_and_turn), state).score > evaluator.evaluate(cast(Any, turn_only), state).score


@pytest.mark.parametrize(
    ('effect_type', 'draw_amount', 'expected_count'),
    [
        (ProduceExamEffectType.ExamCardDraw, 2, 2),
        (ProduceExamEffectType.ExamHandGraveCountCardDraw, 0, 1),
    ],
)
def test_draw_value_distinguishes_unknown_deck_from_confirmed_empty_deck(
    effect_type: ProduceExamEffectType,
    draw_amount: int,
    expected_count: int,
):
    evaluator = EffectHeuristicEvaluator()
    draw = _card('draw', _effect(effect_type, draw_amount, count=0))

    unknown = evaluator.evaluate(
        cast(Any, draw),
        BattleState(remaining_turns=5, hp=20, genki=0, playable_count=2),
    )
    empty = evaluator.evaluate(
        cast(Any, draw),
        BattleState(
            remaining_turns=5, hp=20, genki=0,
            playable_count=2, draw_pile_empty=True,
        ),
    )
    nonempty = evaluator.evaluate(
        cast(Any, draw),
        BattleState(
            remaining_turns=5,
            hp=20,
            genki=0,
            playable_count=2,
            draw_pile_card_ids=('seen-card',),
        ),
    )

    assert empty.score == 8.0 * expected_count
    assert unknown.score == 13.0 * expected_count
    assert nonempty.score == 18.0 * expected_count
    assert '过牌' in unknown.reason


@pytest.mark.parametrize(
    ('effect_type', 'draw_amount', 'expected_count'),
    [
        (ProduceExamEffectType.ExamCardDraw, 2, 2),
        (ProduceExamEffectType.ExamHandGraveCountCardDraw, 0, 1),
    ],
)
def test_draw_value_requires_another_play_after_card(
    effect_type: ProduceExamEffectType,
    draw_amount: int,
    expected_count: int,
):
    evaluator = EffectHeuristicEvaluator()
    draw = _card('draw', _effect(effect_type, draw_amount, count=0))
    draw_with_extra_play = _card(
        'draw-and-play',
        _effect(effect_type, draw_amount, count=0),
        _effect(ProduceExamEffectType.ExamPlayableValueAdd, 1),
    )
    last_play = BattleState(remaining_turns=1, hp=20, genki=0)
    two_plays = BattleState(
        remaining_turns=1, hp=20, genki=0, playable_count=2,
    )

    stranded = evaluator.evaluate(cast(Any, draw), last_play)
    playable = evaluator.evaluate(cast(Any, draw), two_plays)
    self_enabled = evaluator.evaluate(cast(Any, draw_with_extra_play), last_play)

    assert stranded.score == pytest.approx(1.3 * expected_count)
    assert playable.score == 13.0 * expected_count
    assert self_enabled.score == 12.0 + 13.0 * expected_count
    assert '过牌后无出牌次数' in stranded.reason
    assert '过牌后无出牌次数' not in self_enabled.reason


def test_inactive_extra_play_does_not_make_draw_playable(monkeypatch):
    draw_effect = _effect(ProduceExamEffectType.ExamCardDraw, 0)
    extra_play = _effect(ProduceExamEffectType.ExamPlayableValueAdd, 1)
    extra_play._produce_exam_trigger_id = 'inactive-extra-play'
    card = _card('conditional-draw', draw_effect, extra_play)
    state = BattleState(remaining_turns=1, hp=20, genki=0)

    def trigger_verdict(trigger_id, _state):
        return (
            TriggerVerdict.INACTIVE
            if trigger_id == 'inactive-extra-play'
            else TriggerVerdict.ACTIVE
        )

    monkeypatch.setattr(evaluator_module, 'evaluate_trigger_id', trigger_verdict)
    monkeypatch.setattr(memory_module, 'evaluate_trigger_id', trigger_verdict)

    evaluation = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)

    assert evaluation.score == pytest.approx(1.3)
    assert '过牌后无出牌次数' in evaluation.reason
    assert '条件未满足' in evaluation.reason


@pytest.mark.parametrize('draw_amount', [1, 2, 3, 4, 5])
def test_fixed_draw_uses_effect_value_not_effect_count(draw_amount: int):
    card = _card(
        'fixed-draw',
        _effect(ProduceExamEffectType.ExamCardDraw, draw_amount, count=0),
    )
    state = BattleState(
        remaining_turns=5, hp=20, genki=0, playable_count=2,
    )

    evaluation = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)

    assert evaluation.score == 13.0 * draw_amount


@pytest.mark.parametrize(
    ('hand_size', 'expected_draws'),
    [(1, 0), (2, 1), (4, 3)],
)
def test_hand_exchange_draws_one_card_per_remaining_hand_card(
    hand_size: int,
    expected_draws: int,
):
    card = _card(
        'exchange',
        _effect(ProduceExamEffectType.ExamHandGraveCountCardDraw, 0, count=0),
    )
    state = BattleState(
        remaining_turns=5,
        hp=20,
        genki=0,
        playable_count=2,
        hand_size=hand_size,
    )

    evaluation = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)

    assert evaluation.score == 13.0 * expected_draws


def test_official_effect_weight_is_a_bounded_archetype_prior(monkeypatch):
    evaluator = EffectHeuristicEvaluator()
    card = _card(
        'full-power-point',
        _effect(ProduceExamEffectType.ExamFullPowerPoint, 3),
    )
    state = BattleState(
        remaining_turns=7,
        hp=20,
        genki=0,
        archetype=ProduceExamEffectType.ExamFullPower,
    )
    monkeypatch.setattr(
        evaluator_module,
        'get_auto_effect_evaluation',
        lambda *_args: SimpleNamespace(evaluation=10000),
    )

    weighted = evaluator.evaluate(cast(Any, card), state)

    assert '官方体系权重' in weighted.reason
    assert 0 < evaluator._official_effect_prior(
        ProduceExamEffectType.ExamFullPowerPoint,
        state,
    ) <= 12

    monkeypatch.setattr(
        evaluator_module,
        'get_auto_effect_evaluation',
        lambda *_args: SimpleNamespace(evaluation=-10000),
    )
    assert evaluator._official_effect_prior(
        ProduceExamEffectType.StanceLock,
        state,
    ) < 0


def test_status_enchant_expands_nested_effects_and_trigger_weight(monkeypatch):
    evaluator = EffectHeuristicEvaluator()
    outer = _effect(ProduceExamEffectType.ExamStatusEnchant, 0)
    outer.produce_exam_effect._produce_exam_status_enchant_id = 'enchant-1'
    card = _card('enchant-card', outer, plan=ProducePlanType.Plan3)
    state = BattleState(
        remaining_turns=6,
        hp=20,
        genki=0,
        archetype=ProduceExamEffectType.ExamFullPower,
    )
    monkeypatch.setattr(
        evaluator_module,
        'get_exam_status_enchant',
        lambda _id: SimpleNamespace(
            trigger_id='trigger-1',
            effects=(SimpleNamespace(effect_type=ProduceExamEffectType.ExamLesson),),
            chained_effects=(),
        ),
    )
    monkeypatch.setattr(
        evaluator_module,
        'get_auto_trigger_evaluation',
        lambda _id: SimpleNamespace(coefficient_permil=650),
    )
    monkeypatch.setattr(
        evaluator_module,
        'get_auto_effect_evaluation',
        lambda *_args: SimpleNamespace(evaluation=100),
    )
    monkeypatch.setattr(
        evaluator_module,
        'evaluate_trigger_id',
        lambda *_args: TriggerVerdict.UNKNOWN,
    )
    monkeypatch.setattr(
        evaluator_module,
        'load_exam_trigger',
        lambda *_args: SimpleNamespace(
            phase_types=('ProduceExamPhaseType_ExamEndTurn',)
        ),
    )

    evaluation = evaluator.evaluate(cast(Any, card), state)

    assert evaluation.score > 0
    assert '状态附魔效果链' in evaluation.reason

    monkeypatch.setattr(
        evaluator_module,
        'load_exam_trigger',
        lambda *_args: SimpleNamespace(
            phase_types=('ProduceExamPhaseType_StartPlay',)
        ),
    )
    assert evaluator.evaluate(cast(Any, card), state).score == 0


def test_memory_tracks_zones_and_observer_dimensions_from_played_cards():
    memory = BattleMemory()
    preservation = _card(
        'preservation',
        _effect(ProduceExamEffectType.ExamPreservation, 2),
        _effect(ProduceExamEffectType.ExamFullPowerPoint, 4),
        _effect(ProduceExamEffectType.ExamEnthusiasticAdditive, 3),
    )
    memory.observe(7, [cast(Any, preservation)])
    memory.record_play(cast(Any, preservation))

    assert memory.discard == ['preservation']
    assert memory.anomaly.stance is AnomalyStance.PRESERVATION_2
    assert memory.anomaly.full_power_points == 4
    assert memory.anomaly.enthusiasm == 3


def test_memory_reconciles_exact_hud_buff_values():
    memory = BattleMemory()

    memory.reconcile_effect_values({'lesson_buff': 10, 'parameter_buff': 16})

    assert memory.buffs.lesson_buff == 10
    assert memory.buffs.parameter_buff_turns == 16


def test_memory_clears_expired_hud_buff_after_two_misses():
    memory = BattleMemory()
    memory.reconcile_effect_values({'lesson_buff': 10})

    memory.reconcile_effect_values({})
    assert memory.buffs.lesson_buff == 10

    memory.reconcile_effect_values({})
    assert memory.buffs.lesson_buff == 0


def test_memory_reconciles_hud_review_and_full_power_presence():
    memory = BattleMemory()

    memory.reconcile_effect_values({'review': None, 'full_power': None})
    assert memory.buffs.review == 1
    assert memory.anomaly.full_power_active

    memory.reconcile_effect_values({})
    assert memory.anomaly.full_power_active

    memory.reconcile_effect_values({})
    assert memory.buffs.review == 0
    assert not memory.anomaly.full_power_active
    assert memory.effect_misses['review'] == 2
    assert memory.effect_misses['full_power'] == 2


def test_inferred_full_power_survives_missing_hud_icon_for_payoff_card():
    memory = BattleMemory(anomaly=AnomalyState(full_power_points=10))
    memory.effect_misses['full_power'] = 3
    memory.observe(3, [])
    memory.observe(2, [])
    memory.reconcile_effect_values({})
    memory.reconcile_effect_values({})

    assert memory.anomaly.full_power_active
    assert memory.effect_misses['full_power'] == 0
    spurt = SkillCard.from_id('p_card-03-act-0_021')
    assert spurt is not None
    state = BattleState(
        remaining_turns=2, hp=12, genki=9, is_exam=True, exam_score_percent=100,
        playable_count=3,
        archetype=ProduceExamEffectType.ExamFullPower,
        plan_type=ProducePlanType.Plan3,
        anomaly=memory.anomaly,
    )
    assert CompositeEvaluator().evaluate(spurt, state).confirmed_output >= 75

    memory.observe(1, [])
    assert not memory.anomaly.full_power_active


def test_memory_uses_numeric_review_read_from_hud():
    memory = BattleMemory()

    memory.reconcile_effect_values({'review': 2})

    assert memory.buffs.review == 2


def test_low_hp_makes_defence_more_valuable():
    evaluator = EffectHeuristicEvaluator()
    defence = _card('defence', _effect(ProduceExamEffectType.ExamBlock, 5))

    healthy = BattleState(remaining_turns=4, hp=20, genki=0)
    danger = BattleState(remaining_turns=4, hp=5, genki=0)

    assert evaluator.evaluate(cast(Any, defence), danger).score > evaluator.evaluate(cast(Any, defence), healthy).score


def test_clear_gap_does_not_discard_output_toward_perfect():
    evaluator = EffectHeuristicEvaluator()
    exact = _card('exact', _effect(ProduceExamEffectType.ExamLesson, 8), cost=1)
    overkill = _card('overkill', _effect(ProduceExamEffectType.ExamLesson, 80), cost=5)
    state = BattleState(remaining_turns=2, hp=20, genki=10, score_gap=8, score_target_is_final=False)

    assert evaluator.evaluate(cast(Any, overkill), state).score > evaluator.evaluate(cast(Any, exact), state).score


def test_last_turn_rewards_reaching_clear_without_capping_extra_output():
    evaluator = EffectHeuristicEvaluator()
    short = _card('short', _effect(ProduceExamEffectType.ExamLesson, 7))
    exact = _card('exact', _effect(ProduceExamEffectType.ExamLesson, 8))
    more = _card('more', _effect(ProduceExamEffectType.ExamLesson, 20))
    state = BattleState(
        remaining_turns=1, hp=20, genki=10,
        score_gap=8, score_target_is_final=False,
    )

    short_score = evaluator.evaluate(cast(Any, short), state).score
    exact_result = evaluator.evaluate(cast(Any, exact), state)
    more_score = evaluator.evaluate(cast(Any, more), state).score
    assert exact_result.score > short_score + 30
    assert more_score > exact_result.score
    assert '终局CLEAR达标' in exact_result.reason


def test_last_play_removes_positive_official_prior_from_non_scoring_setup():
    class Official:
        def evaluate(self, card, state):
            return CardEvaluation(
                card_id=card._id,
                score=100 if card._id == 'setup' else 0,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test',
            )

    evaluator = CompositeEvaluator(official=cast(Any, Official()))
    setup = _card('setup', _effect(ProduceExamEffectType.ExamLessonBuff, 20))
    output = _card('output', _effect(ProduceExamEffectType.ExamLesson, 9))
    last_play = BattleState(
        remaining_turns=1, hp=20, genki=10, playable_count=1,
        score_gap=9, score_target_is_final=False,
    )
    extra_play = BattleState(
        remaining_turns=1, hp=20, genki=10, playable_count=2,
        score_gap=9, score_target_is_final=False,
    )

    assert evaluator.evaluate(cast(Any, output), last_play).score > evaluator.evaluate(cast(Any, setup), last_play).score
    assert '终局无确定得分' in evaluator.evaluate(cast(Any, setup), last_play).reason
    assert evaluator.evaluate(cast(Any, setup), extra_play).score > 45


def test_perfect_gap_caps_overkill_and_prefers_cheaper_finisher():
    evaluator = EffectHeuristicEvaluator()
    exact = _card('exact', _effect(ProduceExamEffectType.ExamLesson, 8), cost=1)
    overkill = _card('overkill', _effect(ProduceExamEffectType.ExamLesson, 80), cost=5)
    state = BattleState(remaining_turns=2, hp=20, genki=10, score_gap=8, score_target_is_final=True)

    assert evaluator.evaluate(cast(Any, exact), state).score > evaluator.evaluate(cast(Any, overkill), state).score
    assert '足量达标' in evaluator.evaluate(cast(Any, exact), state).reason


def test_uncertain_trigger_output_cannot_claim_goal_completion(monkeypatch):
    monkeypatch.setattr(
        evaluator_module,
        'evaluate_trigger_id',
        lambda trigger_id, _state: (
            TriggerVerdict.UNKNOWN if trigger_id == 'uncertain'
            else TriggerVerdict.ACTIVE
        ),
    )
    effect = _effect(ProduceExamEffectType.ExamLesson, 100)
    effect._produce_exam_trigger_id = 'uncertain'
    card = _card('conditional-output', effect)
    state = BattleState(
        remaining_turns=1, hp=20, genki=10,
        score_gap=20, score_target_is_final=False,
    )

    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)

    assert result.projected_output > 0
    assert result.confirmed_output == 0
    assert '终局CLEAR达标' not in result.reason


def test_fractional_output_is_not_a_hard_one_point_clear_guarantee():
    card = _card('tiny-output', _effect(ProduceExamEffectType.ExamLesson, 3))
    state = BattleState(
        remaining_turns=1, hp=3, genki=4,
        score_gap=1, score_target_is_final=False,
        anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION_2),
        buffs=ExamBuffState(parameter_buff_turns=6),
        observed_effect_values=(('parameter_buff', 6),),
    )

    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)

    assert result.confirmed_output == 1.125
    assert conservative_target_output(result.confirmed_output) == 0
    assert '终局CLEAR达标' not in result.reason
    assert conservative_target_output(9.0) == 9


@pytest.mark.parametrize('is_exam', [False, True])
def test_terminal_genki_does_not_outweigh_output_or_persistent_hp(is_exam):
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=1, hp=4, genki=14, is_exam=is_exam)
    genki = _card('genki', _effect(ProduceExamEffectType.ExamBlock, 10))
    heal = _card('heal', _effect(ProduceExamEffectType.ExamStaminaRecoverFix, 4))
    output = _card('output', _effect(ProduceExamEffectType.ExamLesson, 9))
    genki_result = evaluator.evaluate(cast(Any, genki), state)
    assert genki_result.score == 0
    assert '终局元气无后续利用' in genki_result.reason
    assert evaluator.evaluate(cast(Any, heal), state).score > genki_result.score
    assert evaluator.evaluate(cast(Any, output), state).score > genki_result.score


def test_terminal_genki_retains_value_for_another_play_or_extra_turn():
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=1, hp=4, genki=0, playable_count=2)
    genki = _card('genki', _effect(ProduceExamEffectType.ExamBlock, 10))
    extra_turn = _card(
        'extra-turn', _effect(ProduceExamEffectType.ExamBlock, 10),
        _effect(ProduceExamEffectType.ExamExtraTurn, 1),
    )
    assert evaluator.evaluate(cast(Any, genki), state).score > 0
    assert '终局元气无后续利用' not in evaluator.evaluate(
        cast(Any, extra_turn), BattleState(remaining_turns=1, hp=4, genki=0)
    ).reason


def test_terminal_genki_keeps_explicit_same_card_conversion():
    card = _card(
        'convert', _effect(ProduceExamEffectType.ExamBlock, 10),
        _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000),
    )
    result = EffectHeuristicEvaluator().evaluate(
        cast(Any, card), BattleState(remaining_turns=1, hp=4, genki=5)
    )
    assert '终局元气无后续利用' not in result.reason
    assert result.confirmed_output > 0


@pytest.mark.parametrize('is_exam', [False, True])
def test_terminal_genki_cannot_be_rescued_by_positive_official_prior(is_exam):
    official = SimpleNamespace(evaluate=lambda card, state: CardEvaluation(
        card._id, 100, EvaluationConfidence.HIGH, 'official',
    ))
    evaluator = CompositeEvaluator(official=cast(Any, official))
    state = BattleState(remaining_turns=1, hp=4, genki=0, is_exam=is_exam)
    genki = _card('genki', _effect(ProduceExamEffectType.ExamBlock, 10))
    output = _card('output', _effect(ProduceExamEffectType.ExamLesson, 9))
    assert evaluator.evaluate(cast(Any, genki), state).score == 0
    assert evaluator.evaluate(cast(Any, output), state).score > 0


def test_met_goal_does_not_reward_cards_without_output():
    evaluator = EffectHeuristicEvaluator()
    defence = _card('defence', _effect(ProduceExamEffectType.ExamBlock, 8))
    state = BattleState(remaining_turns=2, hp=20, genki=10, score_gap=0)

    result = evaluator.evaluate(cast(Any, defence), state)

    assert result.score < 35
    assert '足量达标' not in result.reason


def test_zero_clear_gap_still_rewards_output_toward_perfect():
    evaluator = EffectHeuristicEvaluator()
    output = _card('output', _effect(ProduceExamEffectType.ExamLesson, 20))
    state = BattleState(remaining_turns=2, hp=20, genki=10, score_gap=0)

    assert evaluator.evaluate(cast(Any, output), state).score > 20


def test_clear_gap_alone_does_not_suppress_long_term_setup():
    evaluator = EffectHeuristicEvaluator()
    setup = _card('setup', _effect(ProduceExamEffectType.ExamLessonBuff, 20))
    near_clear = BattleState(
        remaining_turns=4,
        hp=20,
        genki=10,
        score_gap=6,
        score_target_is_final=False,
        archetype=ProduceExamEffectType.ExamLessonBuff,
    )
    far_from_clear = BattleState(
        remaining_turns=4,
        hp=20,
        genki=10,
        score_gap=60,
        score_target_is_final=False,
        archetype=ProduceExamEffectType.ExamLessonBuff,
    )

    assert evaluator.evaluate(cast(Any, setup), near_clear).score == evaluator.evaluate(cast(Any, setup), far_from_clear).score


def test_all_six_archetypes_have_profiles():
    assert len(PROFILES) == 6
    assert PROFILES[ProduceExamEffectType.ExamParameterBuff].name == '好调'
    assert PROFILES[ProduceExamEffectType.ExamLessonBuff].name == '集中'
    assert PROFILES[ProduceExamEffectType.ExamReview].name == '好印象'
    assert PROFILES[ProduceExamEffectType.ExamCardPlayAggressive].name == '干劲'
    assert PROFILES[ProduceExamEffectType.ExamConcentration].name == '强气/温存'
    assert PROFILES[ProduceExamEffectType.ExamFullPower].name == '全力'


def test_event_card_setup_is_discounted_but_immediate_output_is_kept():
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(
        remaining_turns=6,
        hp=20,
        genki=10,
        archetype=ProduceExamEffectType.ExamFullPower,
    )
    setup = _effect(ProduceExamEffectType.ExamLessonBuff, 6)
    output = _effect(ProduceExamEffectType.ExamLesson, 20)
    native_setup = _card('native', setup, plan=ProducePlanType.Plan3)
    event_setup = _card('event', setup, plan=ProducePlanType.Plan1)
    event_output = _card('event-output', output, plan=ProducePlanType.Plan1)

    assert evaluator.evaluate(cast(Any, native_setup), state).score > evaluator.evaluate(cast(Any, event_setup), state).score
    assert evaluator.evaluate(cast(Any, event_output), state).score == 20


def test_full_power_dependent_output_uses_cumulative_points():
    evaluator = EffectHeuristicEvaluator()
    card = _card(
        'scaling',
        _effect(ProduceExamEffectType.ExamLessonFullPowerPoint, 4, value2=2000),
        plan=ProducePlanType.Plan3,
    )
    empty = BattleState(remaining_turns=3, hp=20, genki=10)
    charged = BattleState(
        remaining_turns=3,
        hp=20,
        genki=10,
        anomaly=AnomalyState(cumulative_full_power_points=5),
    )

    assert evaluator.evaluate(cast(Any, card), charged).score > evaluator.evaluate(cast(Any, card), empty).score


class _FixedEvaluator:
    def __init__(self, score: float) -> None:
        self.score = score

    def evaluate(self, card, state):
        return CardEvaluation(card._id, self.score, EvaluationConfidence.HIGH, 'fixed')


def test_official_extreme_score_is_only_a_bounded_prior():
    evaluator = CompositeEvaluator(
        official=_FixedEvaluator(1_000_000),
        heuristic=_FixedEvaluator(10),
    )
    state = BattleState(
        remaining_turns=3,
        hp=20,
        genki=10,
        archetype=ProduceExamEffectType.ExamFullPower,
    )
    event_card = _card('event', plan=ProducePlanType.Plan1)

    assert evaluator.evaluate(cast(Any, event_card), state).score == 22


def test_memory_activates_full_power_next_turn_and_keeps_remainder():
    memory = BattleMemory(anomaly=AnomalyState(full_power_points=8))
    card = _card(
        'charge',
        _effect(ProduceExamEffectType.ExamFullPowerPoint, 5),
        plan=ProducePlanType.Plan3,
    )

    memory.record_play(cast(Any, card))

    assert not memory.anomaly.full_power_active
    assert memory.anomaly.full_power_pending
    assert memory.anomaly.full_power_points == 13
    assert memory.anomaly.cumulative_full_power_points == 5

    memory.observe(3, [])
    memory.observe(2, [])

    assert memory.anomaly.full_power_active
    assert not memory.anomaly.full_power_pending
    assert memory.anomaly.full_power_points == 3

    memory.observe(1, [])
    assert not memory.anomaly.full_power_active


def test_leaving_preservation_for_full_power_grants_enthusiasm():
    memory = BattleMemory(
        anomaly=AnomalyState(
            stance=AnomalyStance.PRESERVATION_2,
            full_power_points=10,
            full_power_pending=True,
        )
    )
    memory.observe(3, [])
    memory.observe(2, [])

    assert memory.anomaly.stance is AnomalyStance.NORMAL
    assert memory.anomaly.full_power_active
    assert memory.anomaly.enthusiasm == 8
    assert memory.anomaly.extra_play_count == 2


def test_repeated_concentration_enters_second_aggressive_stance():
    memory = BattleMemory()
    concentration = _card(
        'concentration',
        _effect(ProduceExamEffectType.ExamConcentration, 1),
        plan=ProducePlanType.Plan3,
    )

    memory.record_play(cast(Any, concentration))
    memory.record_play(cast(Any, concentration))
    memory.record_play(cast(Any, concentration))

    assert memory.anomaly.stance is AnomalyStance.AGGRESSIVE_2
    assert memory.anomaly.stance_change_count == 3
    assert memory.anomaly.concentration_change_count == 3


def test_upgrading_preservation_does_not_count_as_leaving_it():
    memory = BattleMemory()
    preservation = _card(
        'preservation',
        _effect(ProduceExamEffectType.ExamPreservation, 1),
        plan=ProducePlanType.Plan3,
    )

    memory.record_play(cast(Any, preservation))
    memory.record_play(cast(Any, preservation))

    assert memory.anomaly.stance is AnomalyStance.PRESERVATION_2
    assert memory.anomaly.preservation_change_count == 2
    assert memory.anomaly.enthusiasm == 0
    assert memory.anomaly.extra_play_count == 0


def test_enthusiasm_and_effect_count_are_applied_to_each_hit():
    evaluator = EffectHeuristicEvaluator()
    card = _card('multi', _effect(ProduceExamEffectType.ExamLesson, 4, count=3))
    state = BattleState(
        remaining_turns=2,
        hp=20,
        genki=10,
        anomaly=AnomalyState(enthusiasm=2),
    )

    assert evaluator.evaluate(cast(Any, card), state).score == pytest.approx(21.6)


def test_full_power_points_have_almost_no_value_on_last_turn():
    evaluator = EffectHeuristicEvaluator()
    charge = _card('charge', _effect(ProduceExamEffectType.ExamFullPowerPoint, 10))
    output = _card('output', _effect(ProduceExamEffectType.ExamLesson, 5))
    state = BattleState(remaining_turns=1, hp=20, genki=10)

    assert evaluator.evaluate(cast(Any, output), state).score > evaluator.evaluate(cast(Any, charge), state).score


def test_exam_prefers_output_over_healthy_defence_and_cross_plan_setup():
    evaluator = EffectHeuristicEvaluator()
    output = _card('output', _effect(ProduceExamEffectType.ExamLesson, 10))
    defensive_setup = _card(
        'setup',
        _effect(ProduceExamEffectType.ExamBlock, 8),
        _effect(ProduceExamEffectType.ExamParameterBuff, 4, turn=3),
        plan=ProducePlanType.Plan1,
    )
    state = BattleState(
        remaining_turns=2,
        hp=20,
        genki=5,
        is_exam=True,
        archetype=ProduceExamEffectType.ExamFullPower,
    )

    assert evaluator.evaluate(cast(Any, output), state).score > evaluator.evaluate(cast(Any, defensive_setup), state).score


def test_drive_is_a_logic_buff_while_concentration_changes_stance():
    memory = BattleMemory()
    drive = _card(
        'drive',
        _effect(ProduceExamEffectType.ExamCardPlayAggressive, 4),
        plan=ProducePlanType.Plan2,
    )
    concentration = _card(
        'concentration',
        _effect(ProduceExamEffectType.ExamConcentration, 1),
        plan=ProducePlanType.Plan3,
    )

    memory.record_play(cast(Any, drive))
    assert memory.buffs.aggressive == 4
    assert memory.anomaly.stance is AnomalyStance.NORMAL
    memory.record_play(cast(Any, concentration))
    assert memory.anomaly.stance is AnomalyStance.AGGRESSIVE


def test_remaining_turn_ocr_is_monotonic_and_changes_at_most_one():
    memory = BattleMemory()

    assert memory.reconcile_remaining_turns(7) == 7
    memory.observe(7, [])
    assert memory.reconcile_remaining_turns(4) == 6
    memory.observe(6, [])
    assert memory.reconcile_remaining_turns(7) == 6
    assert memory.reconcile_remaining_turns(None) == 6


def test_remaining_turn_recovers_confirmed_lost_tens_digit():
    memory = BattleMemory()

    assert memory.reconcile_remaining_turns(2) == 2
    memory.observe(2, [])
    assert memory.reconcile_remaining_turns(12) == 2
    memory.observe(2, [])
    assert memory.reconcile_remaining_turns(11) == 11
    memory.observe(11, [])
    assert memory.reconcile_remaining_turns(10) == 10


def test_remaining_turn_ignores_isolated_double_digit_ocr_spike():
    memory = BattleMemory()

    memory.observe(memory.reconcile_remaining_turns(2), [])
    assert memory.reconcile_remaining_turns(12) == 2
    memory.observe(2, [])
    assert memory.reconcile_remaining_turns(2) == 2
    memory.observe(2, [])
    assert memory.reconcile_remaining_turns(12) == 2


def test_remaining_turn_recovers_initial_fallback():
    memory = BattleMemory()
    memory.observe(memory.reconcile_remaining_turns(None), [])
    assert memory.reconcile_remaining_turns(12) == 7
    memory.observe(7, [])
    assert memory.reconcile_remaining_turns(11) == 11


def test_remaining_turn_recovers_stale_count_at_exam_end():
    memory = BattleMemory()
    memory.observe(7, [])
    assert memory.reconcile_remaining_turns(1) == 6
    memory.observe(6, [])
    assert memory.reconcile_remaining_turns(1) == 1
    memory.observe(1, [])
    assert memory.reconcile_remaining_turns(1) == 1


def test_remaining_turn_rejects_out_of_range_readings():
    memory = BattleMemory()
    assert memory.reconcile_remaining_turns(0) == 7
    memory.observe(7, [])
    assert memory.reconcile_remaining_turns(31) == 7
    assert memory.reconcile_remaining_turns(-1) == 7


@pytest.mark.parametrize('reverse,expected', [(False, 33), (True, 31)])
def test_review_conversion_uses_only_preceding_effects(reverse, expected):
    gain = _effect(ProduceExamEffectType.ExamReview, 2)
    output = _effect(ProduceExamEffectType.ExamLessonDependExamReview, 1000)
    effects = [output, gain] if reverse else [gain, output]
    card = _card('review-order', *effects)
    before = BattleState(remaining_turns=3, hp=33, genki=92,
                         buffs=ExamBuffState(review=31), observed_effect_values=(('review', 31),))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), before)
    assert result.projected_output == expected and result.confirmed_output == expected
    assert card.play_effects == effects and before.buffs.review == 31


@pytest.mark.parametrize('observed', [(), (('review', None),), (('review', 30),)])
def test_inferred_missing_or_conflicting_review_is_not_confirmed_output(observed):
    card = _card('review-unknown', _effect(ProduceExamEffectType.ExamReview, 2),
                 _effect(ProduceExamEffectType.ExamLessonDependExamReview, 1000))
    before = BattleState(remaining_turns=3, hp=33, genki=92,
                         buffs=ExamBuffState(review=31), observed_effect_values=observed)
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), before)
    assert result.projected_output == 33 and result.confirmed_output == 0
    assert '产分依赖数值未确认' in result.reason


def test_review_cost_is_paid_before_conversion():
    card = _card('review-payment', _effect(ProduceExamEffectType.ExamLessonDependExamReview, 1000),
                 cost_type='ExamCostType_ExamReview', cost_value=3)
    before = BattleState(remaining_turns=3, hp=20, genki=20,
                         buffs=ExamBuffState(review=5), observed_effect_values=(('review', 5),))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), before)
    assert result.projected_output == result.confirmed_output == 2


@pytest.mark.parametrize('known,confirmed', [(True, 11), (False, 0)])
def test_genki_conversion_pays_then_applies_preceding_gain(known, confirmed):
    card = _card('genki-order', _effect(ProduceExamEffectType.ExamBlock, 10),
                 _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000), cost=4)
    before = BattleState(remaining_turns=3, hp=20, genki=5, genki_known=known)
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), before)
    assert result.projected_output == 11 and result.confirmed_output == confirmed


def test_empty_projected_review_does_not_reuse_raw_hud_for_output():
    card = _card('projected-unknown', _effect(ProduceExamEffectType.ExamLessonDependExamReview, 1000))
    state = BattleState(remaining_turns=3, hp=20, genki=20, buffs=ExamBuffState(review=31),
                        observed_effect_values=(('review', 31),), projected_effect_values=())
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert result.projected_output == 31 and result.confirmed_output == 0


def test_unknown_preceding_review_effect_invalidates_conversion(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    def trigger(trigger_id, state):
        return TriggerVerdict.UNKNOWN if trigger_id == 'uncertain' else TriggerVerdict.ACTIVE
    for module in (evaluator_module, memory_module, simulator):
        monkeypatch.setattr(module, 'evaluate_trigger_id', trigger)
    gain = _effect(ProduceExamEffectType.ExamReview, 2)
    gain._produce_exam_trigger_id = 'uncertain'
    card = _card('uncertain-sequence', gain, _effect(ProduceExamEffectType.ExamLessonDependExamReview, 1000))
    state = BattleState(remaining_turns=3, hp=20, genki=20, buffs=ExamBuffState(review=31),
                        observed_effect_values=(('review', 31),))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert result.confirmed_output == 0 and result.projected_output == 31


def test_same_card_review_reduction_updates_conversion():
    card = _card('reduced-review', _effect(ProduceExamEffectType.ExamReviewReduce, 2),
                 _effect(ProduceExamEffectType.ExamLessonDependExamReview, 1000))
    state = BattleState(remaining_turns=3, hp=20, genki=20, buffs=ExamBuffState(review=5),
                        observed_effect_values=(('review', 5),))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert result.confirmed_output == result.projected_output == 3


@pytest.mark.parametrize('reverse,expected', [(False, 30), (True, 15)])
def test_concentration_multiplier_follows_card_effect_order(reverse, expected):
    stance = _effect(ProduceExamEffectType.ExamConcentration, 1)
    output = _effect(ProduceExamEffectType.ExamLesson, 15)
    card = _card('stance-order', *([output, stance] if reverse else [stance, output]))
    state = BattleState(remaining_turns=3, hp=20, genki=10)
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert result.projected_output == result.confirmed_output == expected
    assert state.anomaly.stance is AnomalyStance.NORMAL


@pytest.mark.parametrize('anomaly,expected', [
    (AnomalyState(stance=AnomalyStance.PRESERVATION), 40),
    (AnomalyState(stance=AnomalyStance.PRESERVATION_2), 46),
    (AnomalyState(stance=AnomalyStance.AGGRESSIVE), 37.5),
    (AnomalyState(stance_locked=True), 15),
    (AnomalyState(full_power_active=True), 45),
])
def test_stance_output_reuses_transition_rewards_and_locks(anomaly, expected):
    card = _card('stance-transition', _effect(ProduceExamEffectType.ExamConcentration, 1),
                 _effect(ProduceExamEffectType.ExamLesson, 15))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card),
        BattleState(remaining_turns=3, hp=20, genki=10, anomaly=anomaly))
    assert result.projected_output == result.confirmed_output == expected


def test_each_scoring_segment_uses_its_own_parameter_buff_multiplier():
    card = _card('split-output', _effect(ProduceExamEffectType.ExamLesson, 10),
                 _effect(ProduceExamEffectType.ExamParameterBuff, 0, turn=2),
                 _effect(ProduceExamEffectType.ExamLesson, 10))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card),
        BattleState(remaining_turns=3, hp=20, genki=10,
            observed_effect_values=(('parameter_buff', 0),)))
    assert result.projected_output == result.confirmed_output == 25


def test_parameter_buff_cost_is_paid_before_output_multiplier():
    card = _card('buff-payment', _effect(ProduceExamEffectType.ExamLesson, 10),
                 cost_type='ExamCostType_ExamParameterBuff', cost_value=2)
    state = BattleState(remaining_turns=3, hp=20, genki=10,
        buffs=ExamBuffState(parameter_buff_turns=2),
        observed_effect_values=(('parameter_buff', 2),))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert result.projected_output == result.confirmed_output == 10


def test_enthusiasm_grant_only_affects_later_hits():
    card = _card('enthusiasm-order', _effect(ProduceExamEffectType.ExamLesson, 4, count=2),
                 _effect(ProduceExamEffectType.ExamEnthusiasticAdditive, 3),
                 _effect(ProduceExamEffectType.ExamLesson, 4, count=2))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card),
        BattleState(remaining_turns=3, hp=20, genki=10))
    assert result.projected_output == result.confirmed_output == 22


def test_unknown_preceding_stance_does_not_confirm_later_output(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    def trigger(trigger_id, state):
        return TriggerVerdict.UNKNOWN if trigger_id == 'uncertain-stance' else TriggerVerdict.ACTIVE
    for module in (evaluator_module, memory_module, simulator):
        monkeypatch.setattr(module, 'evaluate_trigger_id', trigger)
    stance = _effect(ProduceExamEffectType.ExamConcentration, 1)
    stance._produce_exam_trigger_id = 'uncertain-stance'
    card = _card('unknown-stance', _effect(ProduceExamEffectType.ExamLesson, 4), stance,
                 _effect(ProduceExamEffectType.ExamLesson, 15))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card),
        BattleState(remaining_turns=3, hp=20, genki=10))
    assert result.projected_output == 19 and result.confirmed_output == 4


@pytest.mark.parametrize('cost,expected', [(1, 15), (2, 0)])
def test_followup_condition_uses_remaining_parameter_buff_after_payment(monkeypatch, cost, expected):
    from kaa.tasks.produce.new.play_cards.planner import trigger
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    condition = trigger.ExamTrigger(field_status_types=('ProduceExamFieldStatusType_ParameterBuff',))
    monkeypatch.setattr(trigger, 'load_exam_trigger', lambda key: condition if key else None)
    payment = _card('payment', cost_type='ExamCostType_ExamParameterBuff', cost_value=cost)
    effect = _effect(ProduceExamEffectType.ExamLesson, 10)
    effect._produce_exam_trigger_id = 'has-parameter-buff'
    followup = _card('conditional-output', effect)
    state = BattleState(remaining_turns=3, hp=20, genki=20, playable_count=2,
        buffs=ExamBuffState(parameter_buff_turns=2),
        observed_effects=frozenset({'parameter_buff'}),
        observed_effect_values=(('parameter_buff', 2),))
    after = simulate_play(state, cast(Any, payment))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, followup), after)
    assert result.projected_output == result.confirmed_output == expected
    assert after.observed_effect_values == state.observed_effect_values


@pytest.mark.parametrize('total', [0, 5])
@pytest.mark.parametrize('coefficient', [0, 2000])
def test_cumulative_point_output_is_estimated_without_complete_history(total, coefficient):
    card = _card('cumulative', _effect(ProduceExamEffectType.ExamLessonFullPowerPoint, 4, value2=coefficient))
    state = BattleState(remaining_turns=3, hp=100, genki=0,
        anomaly=AnomalyState(full_power_points=5, cumulative_full_power_points=total),
        observed_effect_values=(('full_power_point', 5),))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert result.projected_output == 4 + total * coefficient / 1000
    assert result.confirmed_output == (4 if coefficient == 0 else 0)


def test_same_card_point_gain_does_not_prove_complete_cumulative_history():
    card = _card('gain-and-convert',
        _effect(ProduceExamEffectType.ExamFullPowerPoint, 3),
        _effect(ProduceExamEffectType.ExamLessonFullPowerPoint, 4, value2=2000))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), BattleState(remaining_turns=3, hp=100, genki=0))
    assert result.projected_output == 10 and result.confirmed_output == 0


@pytest.mark.parametrize('percent,expected,confirmed', [
    (100, 24, 24), (468, 112.32, 112.32), (0, 0, 0),
    (None, 24, 0), (-1, 24, 0),
])
def test_exam_percentage_scales_scoring_and_missing_value_cannot_prove_output(
    percent, expected, confirmed,
):
    card = _card('exam-output', _effect(ProduceExamEffectType.ExamLesson, 6, count=2))
    state = BattleState(
        remaining_turns=3, hp=20, genki=12, is_exam=True,
        exam_score_percent=percent,
        anomaly=AnomalyState(stance=AnomalyStance.AGGRESSIVE),
    )
    evaluation = EffectHeuristicEvaluator().evaluate(cast(SkillCard, card), state)
    assert evaluation.projected_output == pytest.approx(expected)
    assert evaluation.confirmed_output == pytest.approx(confirmed)


def test_lesson_output_ignores_exam_percentage():
    card = _card('lesson-output', _effect(ProduceExamEffectType.ExamLesson, 6, count=2))
    state = BattleState(remaining_turns=3, hp=20, genki=12, exam_score_percent=468)
    evaluation = EffectHeuristicEvaluator().evaluate(cast(SkillCard, card), state)
    assert evaluation.projected_output == evaluation.confirmed_output == 12


def test_same_turn_search_retains_observed_exam_percentage():
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    recovery = _card('recovery', _effect(ProduceExamEffectType.ExamStaminaRecoverFix, 2))
    output = _card('output', _effect(ProduceExamEffectType.ExamLesson, 10))
    state = BattleState(
        remaining_turns=3, hp=20, genki=12, is_exam=True,
        exam_score_percent=468, playable_count=2,
    )
    after = simulate_play(state, cast(SkillCard, recovery))
    assert after.remaining_turns == state.remaining_turns
    evaluation = EffectHeuristicEvaluator().evaluate(cast(SkillCard, output), after)
    assert evaluation.projected_output == evaluation.confirmed_output == pytest.approx(46.8)


@pytest.mark.parametrize('percent', [468, None])
def test_current_slot_preview_includes_growth_and_multipliers_once_for_ranking(percent):
    from dataclasses import replace
    card = cast(SkillCard, _card('grown',
        _effect(ProduceExamEffectType.ExamConcentration, 1),
        _effect(ProduceExamEffectType.ExamLesson, 6, count=2)))
    state = BattleState(
        remaining_turns=3, hp=21, genki=12, is_exam=True,
        exam_score_percent=percent, unresolved_growth_card_ids=frozenset({'grown'}),
    )
    base = EffectHeuristicEvaluator().evaluate(card, state)
    preview = EffectHeuristicEvaluator().evaluate(card, replace(
        state, observed_card_output=('grown', 150, 2)))
    assert preview.projected_output == 300
    assert preview.confirmed_output == 0
    assert preview.score > base.score


@pytest.mark.parametrize('observation,projected', [
    (('other-slot', 150, 2), False), (('grown', -1, 2), False),
    (('grown', 150, 0), False), (('grown', 150, 2), True),
])
def test_preview_cannot_apply_to_other_card_invalid_value_or_search_state(observation, projected):
    card = cast(SkillCard, _card('grown', _effect(ProduceExamEffectType.ExamLesson, 6, count=2)))
    state = BattleState(
        remaining_turns=3, hp=21, genki=12,
        observed_card_output=observation,
        projected_effect_values=() if projected else None,
    )
    assert EffectHeuristicEvaluator().evaluate(card, state).projected_output == 12


def test_preview_does_not_replace_multiple_or_dependent_scoring_segments():
    from kaa.tasks.produce.new.play_cards.planner.visible_output import current_preview_output
    state = BattleState(remaining_turns=3, hp=21, genki=12,
                        observed_card_output=('grown', 150, 2))
    for effects in [
        (_effect(ProduceExamEffectType.ExamLesson, 6), _effect(ProduceExamEffectType.ExamLesson, 5)),
        (_effect(ProduceExamEffectType.ExamLessonDependBlock, 1000),),
    ]:
        card = cast(SkillCard, _card('grown', *effects))
        assert current_preview_output(card, state) is None


def test_unconfirmed_condition_cannot_use_current_card_preview(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import visible_output
    card = cast(SkillCard, _card('grown', _effect(ProduceExamEffectType.ExamLesson, 6)))
    state = BattleState(remaining_turns=3, hp=21, genki=12,
                        observed_card_output=('grown', 150, 2))
    monkeypatch.setattr(visible_output, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    assert visible_output.current_preview_output(card, state) is None



def test_simulation_discards_current_slot_preview():
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    card = cast(SkillCard, _card('grown', _effect(ProduceExamEffectType.ExamLesson, 6, count=2)))
    state = BattleState(remaining_turns=3, hp=21, genki=12,
                        observed_card_output=('grown', 150, 2))
    future = simulate_play(state, card)
    assert future.observed_card_output is None
    assert EffectHeuristicEvaluator().evaluate(card, future).projected_output == 12



def test_same_name_slots_keep_individual_preview_and_leave_shared_snapshot_clean():
    from kaa.tasks.produce.new.play_cards.card_output import CardOutputPreview
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    from kaa.tasks.produce.new.play_cards.ui import CardGameObject
    card = cast(SkillCard, _card('same-kind', _effect(ProduceExamEffectType.ExamLesson, 6, count=2)))
    state = BattleState(remaining_turns=3, hp=21, genki=12)
    previews = []
    for value in (150, 70):
        hand = cast(CardGameObject, SimpleNamespace(
            card=card, has_growth_marker=True, output_preview=CardOutputPreview(value, 2)))
        local = PlannerStrategy._candidate_state(state, hand)
        previews.append(EffectHeuristicEvaluator().evaluate(card, local).projected_output)
    assert previews == [300, 140]
    assert state.observed_card_output is None


@pytest.mark.parametrize('case', ['multiple', 'dependent', 'unknown', 'chain', 'search', 'growth'])
def test_ineligible_card_preview_skips_image_properties(monkeypatch, case):
    from kaa.tasks.produce.new.play_cards.planner import visible_output
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    from kaa.tasks.produce.new.play_cards.ui import CardGameObject
    effects = [_effect(ProduceExamEffectType.ExamLesson, 6)]
    if case == 'multiple':
        effects.append(_effect(ProduceExamEffectType.ExamLesson, 5))
    if case == 'dependent':
        effects.append(_effect(ProduceExamEffectType.ExamLessonDependBlock, 1000))
    if case == 'chain':
        effects[0].produce_exam_effect._chain_produce_exam_effect_id = 'unparsed-chain'
    if case == 'unknown':
        monkeypatch.setattr(visible_output, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    card = cast(SkillCard, _card('ineligible', *effects))
    state = BattleState(remaining_turns=3, hp=20, genki=12,
        projected_effect_values=() if case == 'search' else None,
        projected_card_growth=(('unparsed', False),) if case == 'growth' else ())
    class Hand:
        card: SkillCard

        @property
        def has_growth_marker(self):
            raise AssertionError('unsupported preview must not inspect the image')
        @property
        def output_preview(self):
            raise AssertionError('unsupported preview must not request OCR')
    hand = Hand()
    hand.card = card
    assert PlannerStrategy._candidate_state(state, cast(CardGameObject, hand)) is state


@pytest.mark.parametrize('base_count,changes,expected_count,known', [
    (3, (-1,), 2, True), (3, (-1, -1), 1, True),
    (1, (-1,), 1, False), (2, (-3,), 2, False),
    (2, (-2, 2), 4, False), (2, (2, -2), 2, True),
])
def test_count_growth_preserves_order_and_unknown_lower_bound(
    monkeypatch, base_count, changes, expected_count, known,
):
    from kaa.db.card_growth import CardGrowthRule
    from kaa.tasks.produce.new.play_cards.planner import costs
    card = cast(SkillCard, _card('count-target',
        _effect(ProduceExamEffectType.ExamLesson, 6, count=base_count)))
    rule = CardGrowthRule(target_known=True, costs_known=True, output_known=True,
                          lesson_count_changes=changes)
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda _: rule)
    state = BattleState(remaining_turns=3, hp=20, genki=12,
        projected_card_growth=(('count-growth', True),),
        anomaly=AnomalyState(stance=AnomalyStance.AGGRESSIVE, enthusiasm=2))
    result = EffectHeuristicEvaluator().evaluate(card, state)
    assert result.projected_output == 8 * expected_count * 2
    assert result.confirmed_output == (result.projected_output if known else 0)
    assert card.play_effects[0].produce_exam_effect is not None
    assert card.play_effects[0].produce_exam_effect.effect_count == base_count


def test_unknown_count_growth_cannot_be_certified_by_later_addition(monkeypatch):
    from kaa.db.card_growth import CardGrowthRule
    from kaa.tasks.produce.new.play_cards.planner import costs
    rules = {
        'reduce': CardGrowthRule(target_known=True, costs_known=True, output_known=True,
                                lesson_count_changes=(-1,)),
        'add': CardGrowthRule(target_known=True, costs_known=True, output_known=True,
                             lesson_count_changes=(2,)),
    }
    monkeypatch.setattr(costs, 'get_card_growth_rule', rules.__getitem__)
    card = cast(SkillCard, _card('count-target', _effect(ProduceExamEffectType.ExamLesson, 6)))
    state = BattleState(remaining_turns=3, hp=20, genki=12,
        projected_card_growth=(('reduce', True), ('add', True)))
    result = EffectHeuristicEvaluator().evaluate(card, state)
    assert result.projected_output == 18 and result.confirmed_output == 0



@pytest.mark.parametrize('base,bonus,aggressive', [
    (2, 500, 0), (2, 500, 10), (4, 800, 10), (6, 800, 10),
    (2, 1300, 10), (20, 0, 10), (2, 500, 3),
])
def test_aggressive_genki_scores_base_and_bonus_with_defence_weight(base, bonus, aggressive):
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=3, hp=20, genki=0,
        buffs=ExamBuffState(aggressive=aggressive))
    scaled = _card('scaled', _effect(ProduceExamEffectType.ExamBlockAddMultipleAggressive,
        base, value2=bonus))
    ordinary = _card('ordinary', _effect(ProduceExamEffectType.ExamBlockFix,
        cast(Any, base + aggressive * (1 + bonus / 1000))))
    actual = evaluator.evaluate(cast(Any, scaled), state)
    expected = evaluator.evaluate(cast(Any, ordinary), state)
    assert actual.score == pytest.approx(expected.score)
    assert actual.confirmed_output == 0
    assert state.genki == 0


@pytest.mark.parametrize('gain_first', [True, False])
def test_aggressive_genki_estimate_uses_prior_effects_only(gain_first):
    evaluator = EffectHeuristicEvaluator()
    gain = _effect(ProduceExamEffectType.ExamCardPlayAggressive, 5)
    scaled = _effect(ProduceExamEffectType.ExamBlockAddMultipleAggressive, 2, value2=500)
    block = _effect(ProduceExamEffectType.ExamBlockFix, cast(Any, 2 + (15 if gain_first else 10) * 1.5))
    state = BattleState(remaining_turns=3, hp=20, genki=0,
        buffs=ExamBuffState(aggressive=10))
    a = _card('scaled', *( [gain, scaled] if gain_first else [scaled, gain] ))
    b = _card('ordinary', *( [gain, block] if gain_first else [block, gain] ))
    assert evaluator.evaluate(cast(Any, a), state).score == pytest.approx(
        evaluator.evaluate(cast(Any, b), state).score)
    assert state.buffs.aggressive == 10


@pytest.mark.parametrize('exam', [True, False])
def test_terminal_aggressive_genki_has_no_unusable_survival_bonus(exam):
    state = BattleState(remaining_turns=1, hp=5, genki=0, playable_count=1,
        is_exam=exam, buffs=ExamBuffState(aggressive=10))
    card = _card('scaled', _effect(ProduceExamEffectType.ExamBlockAddMultipleAggressive,
        2, value2=500))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert '终局元气无后续利用' in result.reason
    assert result.score == EffectHeuristicEvaluator().evaluate(cast(Any, _card('empty')), state).score


def test_aggressive_genki_scoring_does_not_certify_payment_or_conversion():
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play, can_afford_play
    state = BattleState(remaining_turns=3, hp=0, genki=0, playable_count=2,
        buffs=ExamBuffState(aggressive=10), observed_effect_values=(('aggressive', 10),))
    card = _card('scaled', _effect(ProduceExamEffectType.ExamBlockAddMultipleAggressive,
        2, value2=500), _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000))
    future = simulate_play(state, cast(Any, card))
    assert not future.genki_known
    assert can_afford_play(future, cast(Any, _card('paid', cost=5))) is None
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert result.confirmed_output == 0


@pytest.mark.parametrize('aggressive,review,expected', [(19,19,1.38), (37,17,1.34), (1,20,1.02), (50,25,1.5), (50,50,1.5), (0,20,1.0)])
def test_pride_uses_smaller_current_resource_and_caps_at_fifty_percent(aggressive, review, expected):
    state = BattleState(remaining_turns=4, hp=12, genki=90,
        buffs=ExamBuffState(aggressive=aggressive, review=review),
        observed_effects=frozenset({'pride'}),
        observed_effect_values=(('pride', None), ('aggressive', aggressive), ('review', review)))
    assert EffectHeuristicEvaluator._output_multiplier(state) == pytest.approx(expected)
    assert EffectHeuristicEvaluator._output_multiplier_known(state)


@pytest.mark.parametrize('values', [(('aggressive', None), ('review', 17)), (('review', 17),), (('aggressive', 57), ('review', 17))])
def test_pride_does_not_certify_unknown_or_conflicting_resource(values):
    state = BattleState(remaining_turns=4, hp=12, genki=90,
        buffs=ExamBuffState(aggressive=37, review=17),
        observed_effects=frozenset({'pride'}), observed_effect_values=values)
    assert not EffectHeuristicEvaluator._output_multiplier_known(state)


def test_pride_grant_propagates_through_same_card_prefix_without_editing_observation():
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play_prefix
    state = BattleState(remaining_turns=4, hp=12, genki=90,
        buffs=ExamBuffState(aggressive=20, review=10),
        observed_effect_values=(('aggressive',20),('review',10)))
    card = _card("pride-chain", _effect(ProduceExamEffectType.ExamLessonValueMultipleDependReviewOrAggressive,0,turn=5),
                 _effect(ProduceExamEffectType.ExamReview,9),
                 _effect(ProduceExamEffectType.ExamLesson,9))
    prefix = simulate_play_prefix(state,card,2)
    assert 'pride' not in state.observed_effects
    assert dict(prefix.projected_effect_values)['pride'] == 1
    assert EffectHeuristicEvaluator._output_multiplier(prefix) == pytest.approx(1.38)
    assert EffectHeuristicEvaluator._output_multiplier_known(prefix)


def test_unknown_projected_pride_cannot_certify_output():
    state = BattleState(remaining_turns=4, hp=12, genki=90,
        buffs=ExamBuffState(aggressive=20, review=20),
        projected_effect_values=(('pride',None),('aggressive',20),('review',20)))
    assert not EffectHeuristicEvaluator._output_multiplier_known(state)


def test_pride_combined_good_condition_is_estimate_until_calibrated():
    state = BattleState(remaining_turns=4, hp=12, genki=90,
        buffs=ExamBuffState(aggressive=20, review=20,parameter_buff_turns=1),
        observed_effects=frozenset({'pride'}),
        observed_effect_values=(('aggressive',20),('review',20)))
    assert not EffectHeuristicEvaluator._output_multiplier_known(state)


@pytest.mark.parametrize('grant_first,expected', [(True,12.42),(False,9.0)])
def test_pride_same_card_output_obeys_effect_order(grant_first,expected):
    state = BattleState(remaining_turns=4,hp=20,genki=20,
        buffs=ExamBuffState(aggressive=19,review=19),
        observed_effect_values=(('aggressive',19),('review',19)))
    grant = _effect(ProduceExamEffectType.ExamLessonValueMultipleDependReviewOrAggressive,0,turn=5)
    output = _effect(ProduceExamEffectType.ExamLesson,9)
    card = _card('ordered-pride', *( [grant,output] if grant_first else [output,grant]))
    result = EffectHeuristicEvaluator().evaluate(card,state)
    assert result.projected_output == pytest.approx(expected)
    assert result.confirmed_output == pytest.approx(expected)


def test_pride_grant_without_duration_does_not_certify_followup():
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    state = BattleState(remaining_turns=4,hp=20,genki=20,
        buffs=ExamBuffState(aggressive=19,review=19),
        observed_effect_values=(('aggressive',19),('review',19)))
    card = _card('invalid-pride', _effect(ProduceExamEffectType.ExamLessonValueMultipleDependReviewOrAggressive,0))
    after = simulate_play(state,card)
    assert dict(after.projected_effect_values)['pride'] is None
    assert not EffectHeuristicEvaluator._output_multiplier_known(after)


@pytest.mark.parametrize('pride', [False, True])
@pytest.mark.parametrize('exam', [False, True])
def test_unmodeled_visible_percentage_does_not_certify_output(pride, exam):
    from dataclasses import replace
    state = BattleState(remaining_turns=3, hp=20, genki=87,
        is_exam=exam, exam_score_percent=1091 if exam else None,
        buffs=ExamBuffState(aggressive=26, review=9),
        observed_effects=frozenset({'pride'} if pride else set()),
        observed_effect_values=(('aggressive',26),('review',9)))
    visible = replace(state, observed_effects=state.observed_effects | {'lesson_value_up'})
    card = _card('conversion', _effect(ProduceExamEffectType.ExamLessonDependBlock,1000))
    evaluator = EffectHeuristicEvaluator()
    baseline = evaluator.evaluate(cast(Any,card),state)
    result = evaluator.evaluate(cast(Any,card),visible)
    assert baseline.confirmed_output > 0
    assert result.confirmed_output == 0
    assert result.projected_output == baseline.projected_output
    assert not evaluator._output_multiplier_known(replace(visible,
        projected_effect_values=(('aggressive',26),('review',9))))


@pytest.mark.parametrize('is_exam', [False, True])
def test_consuming_conversion_ranks_below_equal_output_that_preserves_genki(is_exam):
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=5, hp=4, genki=20,
                        is_exam=is_exam, exam_score_percent=100)
    results = [evaluator.evaluate(cast(Any, _card(
        'conversion', _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000, value2=ratio),
    )), state) for ratio in (0, 500, 1000)]
    assert results[0].score > results[1].score > results[2].score
    assert len({r.projected_output for r in results}) == 1
    assert len({r.confirmed_output for r in results}) == 1


@pytest.mark.parametrize('turns,plays,extra_turn', [(1, 1, False), (1, 2, False), (1, 1, True), (5, 1, False)])
def test_conversion_only_charges_consumption_when_genki_has_future_use(turns, plays, extra_turn):
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=turns, hp=20, genki=10, playable_count=plays)
    def evaluate(ratio):
        effects = [_effect(ProduceExamEffectType.ExamLessonDependBlock, 1000, value2=ratio)]
        if extra_turn:
            effects.append(_effect(ProduceExamEffectType.ExamExtraTurn, 1))
        return evaluator.evaluate(cast(Any, _card('conversion', *effects)), state)
    keep, spend = evaluate(0), evaluate(1000)
    if turns == 1 and plays == 1 and not extra_turn:
        assert keep.score == spend.score
    else:
        assert keep.score > spend.score
    assert keep.confirmed_output == spend.confirmed_output


def test_conversion_consumption_uses_current_prefix_balance_not_pre_grant_balance():
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=4, hp=20, genki=0)
    def evaluate(ratio):
        return evaluator.evaluate(cast(Any, _card('conversion',
            _effect(ProduceExamEffectType.ExamBlockFix, 10),
            _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000, value2=ratio),
        )), state)
    assert evaluate(0).score > evaluate(1000).score
    assert evaluate(0).projected_output == evaluate(1000).projected_output


def test_unknown_genki_does_not_invent_consumption_cost_or_confirm_output():
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(remaining_turns=4, hp=20, genki=10, genki_known=False)
    results = [evaluator.evaluate(cast(Any, _card('conversion',
        _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000, value2=ratio),
    )), state) for ratio in (0, 1000)]
    assert results[0].score == results[1].score
    assert all(r.confirmed_output == 0 for r in results)


def test_consumption_cost_retains_existing_low_hp_survival_priority():
    evaluator = EffectHeuristicEvaluator()
    def loss(hp):
        state = BattleState(remaining_turns=4, hp=hp, genki=10, is_exam=True, exam_score_percent=100)
        results = [evaluator.evaluate(cast(Any, _card('conversion',
            _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000, value2=ratio),
        )), state) for ratio in (0, 1000)]
        return results[0].score - results[1].score
    assert loss(4) > loss(20) > 0


@pytest.mark.parametrize('genki,percent,expected', [(45, 1676, 2414), (193, 1676, 10358), (0, 1676, 0), (193, 0, 0)])
def test_calibrated_full_genki_conversion_rounds_parameter_before_exam(genki, percent, expected):
    card = _card('full', _effect(ProduceExamEffectType.ExamLessonDependBlock, 3200, value2=1000))
    state = BattleState(remaining_turns=6, hp=17, genki=genki, is_exam=True, exam_score_percent=percent)
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card), state)
    assert result.projected_output == result.confirmed_output == expected
    assert '分段进位' in result.reason


@pytest.mark.parametrize('boundary', ['unknown_genki', 'missing_percent', 'lesson', 'half', 'other_ratio', 'repeat', 'pride', 'arrow', 'good_condition', 'full_power', 'stance', 'unknown_trigger', 'growth'])
def test_conversion_rounding_does_not_expand_beyond_calibrated_scope(monkeypatch, boundary):
    from dataclasses import replace
    state = BattleState(remaining_turns=6, hp=17, genki=193, is_exam=True, exam_score_percent=1676)
    effect = _effect(ProduceExamEffectType.ExamLessonDependBlock, 3200, value2=1000)
    if boundary == 'unknown_genki': state = replace(state, genki_known=False)
    elif boundary == 'missing_percent': state = replace(state, exam_score_percent=None)
    elif boundary == 'lesson': state = replace(state, is_exam=False)
    elif boundary == 'half': effect.produce_exam_effect.effect_value2 = 500
    elif boundary == 'other_ratio': effect.produce_exam_effect.effect_value1 = 3000
    elif boundary == 'repeat': effect.produce_exam_effect.effect_count = 2
    elif boundary == 'pride': state = replace(state, observed_effects=frozenset({'pride'}))
    elif boundary == 'arrow': state = replace(state, observed_effects=frozenset({'lesson_value_up'}))
    elif boundary == 'good_condition': state = replace(state, buffs=ExamBuffState(parameter_buff_turns=3))
    elif boundary == 'full_power': state = replace(state, anomaly=AnomalyState(full_power_active=True))
    elif boundary == 'stance': state = replace(state, anomaly=AnomalyState(stance=AnomalyStance.AGGRESSIVE))
    elif boundary == 'unknown_trigger':
        monkeypatch.setattr(evaluator_module, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    elif boundary == 'growth':
        monkeypatch.setattr(evaluator_module, 'projected_growth_output', lambda _state, _card, _kind, value, count, **_: (value, count, False))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, _card('full', effect)), state)
    assert '分段进位' not in result.reason
    if boundary in {'unknown_genki', 'missing_percent', 'arrow', 'unknown_trigger', 'growth'}:
        assert result.confirmed_output == 0


@pytest.mark.parametrize('percent', [50, 100, 468, 1091, 1676])
@pytest.mark.parametrize('conversion', [False, True])
def test_motivation_exam_ranking_uses_parameter_units_without_changing_points(percent, conversion):
    state = BattleState(
        remaining_turns=6, hp=20, genki=20, is_exam=True,
        exam_score_percent=percent,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive,
    )
    kind = (ProduceExamEffectType.ExamLessonDependBlock if conversion
            else ProduceExamEffectType.ExamLessonFix)
    card = _card('output', _effect(kind, 1000 if conversion else 20))
    setup = _card('setup', _effect(ProduceExamEffectType.ExamCardPlayAggressive, 5))
    evaluator = EffectHeuristicEvaluator()
    reference = replace(state, exam_score_percent=100)
    result = evaluator.evaluate(cast(Any, card), state)
    baseline = evaluator.evaluate(cast(Any, card), reference)
    assert result.score == pytest.approx(baseline.score)
    assert result.projected_output == pytest.approx(20 * percent / 100)
    assert result.confirmed_output == pytest.approx(result.projected_output)
    assert evaluator.evaluate(cast(Any, setup), state).score == pytest.approx(
        evaluator.evaluate(cast(Any, setup), reference).score)


@pytest.mark.parametrize('percent', [None, 0, -1])
def test_motivation_exam_missing_or_invalid_percent_keeps_conservative_fallback(percent):
    state = BattleState(remaining_turns=6, hp=20, genki=20, is_exam=True, exam_score_percent=percent,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    assert EffectHeuristicEvaluator._score_output(20, state) == 32


@pytest.mark.parametrize('archetype', list(PROFILES))
def test_motivation_normalization_leaves_courses_and_other_exam_profiles_unchanged(archetype):
    state = BattleState(remaining_turns=6, hp=20, genki=20, archetype=archetype, exam_score_percent=468)
    assert EffectHeuristicEvaluator._score_output(93.6, state) == 93.6
    if archetype != ProduceExamEffectType.ExamCardPlayAggressive:
        assert EffectHeuristicEvaluator._score_output(93.6, replace(state, is_exam=True)) == pytest.approx(149.76)


@pytest.mark.parametrize('percent', [100, 468, 1676])
def test_motivation_terminal_conversion_still_beats_unusable_setup(percent):
    state = BattleState(remaining_turns=1, hp=20, genki=20,
        is_exam=True, exam_score_percent=percent,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    scorer = _card('conversion', _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000))
    setup = _card('motivation', _effect(ProduceExamEffectType.ExamCardPlayAggressive, 5))
    evaluator = EffectHeuristicEvaluator()
    assert evaluator.evaluate(cast(Any, scorer), state).score > evaluator.evaluate(cast(Any, setup), state).score


@pytest.mark.parametrize('gain_first', [True, False])
def test_ordinary_genki_uses_only_motivation_before_its_effect(gain_first):
    gain = _effect(ProduceExamEffectType.ExamCardPlayAggressive, 5)
    block = _effect(ProduceExamEffectType.ExamBlock, 2)
    state = BattleState(remaining_turns=4, hp=20, genki=0,
        buffs=ExamBuffState(aggressive=10), observed_effect_values=(('aggressive', 10),))
    actual = _card('ordinary', *([gain, block] if gain_first else [block, gain]))
    reference = _card('fixed', *([gain, _effect(ProduceExamEffectType.ExamBlockFix, 17)] if gain_first
        else [_effect(ProduceExamEffectType.ExamBlockFix, 12), gain]))
    evaluator = EffectHeuristicEvaluator()
    assert evaluator.evaluate(cast(Any, actual), state).score == pytest.approx(
        evaluator.evaluate(cast(Any, reference), state).score)


@pytest.mark.parametrize('known', [True, False])
def test_ordinary_genki_motivation_bonus_changes_followup_estimate_without_certifying_unknown(known):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    state = BattleState(remaining_turns=4, hp=20, genki=0,
        buffs=ExamBuffState(aggressive=44),
        observed_effect_values=(('aggressive', 44 if known else None),))
    card = _card('extra', _effect(ProduceExamEffectType.ExamBlock, 1),
        _effect(ProduceExamEffectType.ExamPlayableValueAdd, 1))
    future = simulate_play(state, cast(Any, card))
    assert future.genki == 45
    assert future.genki_known is known
    conversion = _card('conversion', _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000))
    output = EffectHeuristicEvaluator().evaluate(cast(Any, conversion), future)
    assert output.projected_output == 45
    assert output.confirmed_output == (45 if known else 0)
    assert state.genki == 0


@pytest.mark.parametrize('gain_first,expected', [(True, 17), (False, 12)])
def test_ordinary_genki_simulation_orders_motivation_and_preserves_card(gain_first, expected):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    gain = _effect(ProduceExamEffectType.ExamCardPlayAggressive, 5)
    block = _effect(ProduceExamEffectType.ExamBlock, 2)
    card = _card('ordered', *([gain, block] if gain_first else [block, gain]))
    state = BattleState(remaining_turns=4, hp=20, genki=0,
        buffs=ExamBuffState(aggressive=10), observed_effect_values=(('aggressive', 10),))
    future = simulate_play(state, cast(Any, card))
    assert future.genki == expected and future.genki_known
    assert state.buffs.aggressive == 10 and block.produce_exam_effect.effect_value1 == 2


@pytest.mark.parametrize('fixed,expected', [(False, 9), (True, 2)])
def test_ordinary_genki_bonus_uses_motivation_after_special_payment_but_fixed_gain_does_not(fixed, expected):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    state = BattleState(remaining_turns=4, hp=20, genki=0,
        buffs=ExamBuffState(aggressive=10), observed_effect_values=(('aggressive', 10),))
    card = _card('paid', _effect(ProduceExamEffectType.ExamBlockFix if fixed else ProduceExamEffectType.ExamBlock, 2),
        cost_type='ExamCostType_ExamCardPlayAggressive', cost_value=3)
    result = simulate_play(state, cast(Any, card))
    assert result.genki == expected and result.genki_known


@pytest.mark.parametrize('marker', ['unknown_trigger', 'unknown_prediction', 'unknown_growth'])
def test_ordinary_genki_never_restores_unknown_motivation_budget(monkeypatch, marker):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    state = BattleState(remaining_turns=4, hp=0, genki=0,
        buffs=ExamBuffState(aggressive=10), observed_effect_values=(('aggressive', 10),))
    grant = _effect(ProduceExamEffectType.ExamCardPlayAggressive, 5)
    grant._produce_exam_trigger_id = 'uncertain'
    if marker == 'unknown_trigger':
        monkeypatch.setattr(simulator, 'evaluate_trigger_id',
            lambda key, _: TriggerVerdict.UNKNOWN if key == 'uncertain' else TriggerVerdict.ACTIVE)
    elif marker == 'unknown_prediction':
        state = replace(state, projected_effect_values=())
    else:
        state = replace(state, unresolved_growth_card_ids=frozenset({'uncertain-card'}))
    card = _card('uncertain-card', grant, _effect(ProduceExamEffectType.ExamBlock, 2))
    future = simulator.simulate_play(state, cast(Any, card))
    assert not future.genki_known
    assert simulator.can_afford_play(future, cast(Any, _card('followup', cost=5))) is None


@pytest.mark.parametrize('archetype,genki,extra,protected', [
    (ProduceExamEffectType.ExamCardPlayAggressive, 84, False, True),
    (ProduceExamEffectType.ExamCardPlayAggressive, 0, False, False),
    (ProduceExamEffectType.ExamCardPlayAggressive, 84, True, False),
    (ProduceExamEffectType.ExamReview, 84, False, False),
])
def test_exam_deadline_does_not_stack_surplus_motivation_instead_of_visible_output(archetype, genki, extra, protected):
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    state = BattleState(remaining_turns=3, hp=20, genki=genki, is_exam=True,
        exam_score_percent=402, archetype=archetype,
        buffs=ExamBuffState(aggressive=12), observed_effect_values=(('aggressive', 12),))
    effects = [_effect(ProduceExamEffectType.ExamBlock, 1),
        _effect(ProduceExamEffectType.ExamCardPlayAggressive, 2)]
    if extra: effects.append(_effect(ProduceExamEffectType.ExamPlayableValueAdd, 1))
    setup = SimpleNamespace(card=_card('setup', *effects))
    output = SimpleNamespace(card=_card('output', _effect(ProduceExamEffectType.ExamLesson, 6)))
    candidates = [(setup, CardEvaluation('setup', 30, EvaluationConfidence.LOW, 'setup')),
        (output, CardEvaluation('output', 6, EvaluationConfidence.LOW, 'output', projected_output=24.12, confirmed_output=24.12))]
    result = PlannerStrategy()._protect_exam_finish(cast(Any, candidates), state)
    assert (result[0][1].score < result[1][1].score) is protected



def test_exam_deadline_caps_both_plain_and_delayed_motivation_setup():
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    state = BattleState(remaining_turns=3, hp=20, genki=84, is_exam=True,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    plain = SimpleNamespace(card=_card('plain',
        _effect(ProduceExamEffectType.ExamBlock, 1),
        _effect(ProduceExamEffectType.ExamCardPlayAggressive, 2)))
    delayed = SimpleNamespace(card=_card('delayed',
        _effect(ProduceExamEffectType.ExamCardPlayAggressive, 3),
        _effect(ProduceExamEffectType.ExamEffectTimer, 0),
        _effect(ProduceExamEffectType.ExamStatusEnchant, 0)))
    output = SimpleNamespace(card=_card('output', _effect(ProduceExamEffectType.ExamLesson, 6)))
    candidates = [(plain, CardEvaluation('plain', 18, EvaluationConfidence.LOW, 'plain')),
        (delayed, CardEvaluation('delayed', 17, EvaluationConfidence.LOW, 'delayed')),
        (output, CardEvaluation('output', 6, EvaluationConfidence.LOW, 'output', projected_output=24.12, confirmed_output=24.12))]
    result = PlannerStrategy()._protect_exam_finish(cast(Any, candidates), state)
    assert max(result, key=lambda candidate: candidate[1].score)[0] is output
    unknown = PlannerStrategy()._protect_exam_finish(cast(Any, candidates), replace(state, genki_known=False))
    assert unknown == candidates



def _single_conversion_effect():
    effect = _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000)
    effect.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    return effect


@pytest.mark.parametrize('count', [None, 1])
def test_single_genki_conversion_preview_is_ranking_only_with_unobserved_count(count):
    card = cast(SkillCard, _card('convert', _single_conversion_effect(),
                               cost_type=None))
    state = BattleState(remaining_turns=1, hp=30, genki=208, is_exam=True,
        exam_score_percent=803, archetype=ProduceExamEffectType.ExamCardPlayAggressive,
        observed_effect_values=(('lesson_value_up', None),),
        observed_card_output=('convert', 2506, count))
    result = EffectHeuristicEvaluator().evaluate(card, state)
    assert result.projected_output == 2506
    assert result.confirmed_output == 0
    assert state.observed_card_output == ('convert', 2506, count)
    assert '单次转换预览' in result.reason
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    assert simulate_play(state, card).observed_card_output is None



@pytest.mark.parametrize('case', ['partial_consume', 'multiple_hits', 'missing_count', 'missing_consumption',
    'chain', 'grow_metadata', 'missing_growth_metadata', 'unresolved_growth', 'enchant', 'unknown_trigger', 'search'])
def test_conversion_preview_scope_rejects_complex_or_unknown_metadata(monkeypatch, case):
    from kaa.tasks.produce.new.play_cards.planner import visible_output
    effect = _single_conversion_effect()
    card = cast(SkillCard, _card('convert', effect))
    state = BattleState(remaining_turns=2, hp=30, genki=208,
                        observed_card_output=('convert', 2506, None))
    if case == 'partial_consume': effect.produce_exam_effect.effect_value2 = 500
    elif case == 'multiple_hits': effect.produce_exam_effect.effect_count = 2
    elif case == 'missing_count': effect.produce_exam_effect.effect_count = None
    elif case == 'missing_consumption': effect.produce_exam_effect.effect_value2 = None
    elif case == 'chain': effect.produce_exam_effect._chain_produce_exam_effect_id = 'chain'
    elif case == 'grow_metadata': effect.produce_exam_effect._produce_card_grow_effect_ids = '["grow"]'
    elif case == 'missing_growth_metadata': effect.produce_exam_effect._produce_card_grow_effect_ids = None
    elif case == 'unresolved_growth': state = replace(state, unresolved_growth_card_ids=frozenset({'convert'}))
    elif case == 'enchant': setattr(card, '_produce_card_status_enchant_id', 'enchant')
    elif case == 'unknown_trigger': monkeypatch.setattr(visible_output, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    else: state = replace(state, projected_effect_values=())
    assert visible_output.current_preview_output(card, state) is None


@pytest.mark.parametrize('count', [0, 2, 10])
def test_conversion_preview_does_not_infer_repeated_hits(count):
    from kaa.tasks.produce.new.play_cards.planner.visible_output import current_preview_output
    card = cast(SkillCard, _card('convert', _single_conversion_effect()))
    state = BattleState(remaining_turns=2, hp=30, genki=208,
                        observed_card_output=('convert', 2506, count))
    assert current_preview_output(card, state) is None


def test_conversion_candidate_uses_wide_own_slot_without_inventing_count():
    from kaa.tasks.produce.new.play_cards.card_output import CardOutputPreview
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    card = cast(SkillCard, _card('convert', _single_conversion_effect()))
    state = BattleState(remaining_turns=2, hp=30, genki=208)
    hands = [SimpleNamespace(card=card, has_growth_marker=False,
        output_preview=CardOutputPreview(506, None), wide_output_preview=CardOutputPreview(value, None))
        for value in (2506, 1800)]
    local = [PlannerStrategy._candidate_state(state, hand) for hand in hands]
    assert [s.observed_card_output for s in local] == [('convert', 2506, None), ('convert', 1800, None)]
    assert state.observed_card_output is None
    hands[0].has_growth_marker = True
    assert PlannerStrategy._candidate_state(state, hands[0]) is state


@pytest.mark.parametrize('preview,confirmed', [(50, 50), (51, 0)])
def test_conversion_preview_never_establishes_but_preserves_unchanged_model_proof(preview, confirmed):
    card = cast(SkillCard, _card('convert', _single_conversion_effect()))
    state = BattleState(remaining_turns=1, hp=20, genki=50)
    evaluator = EffectHeuristicEvaluator()
    assert evaluator.evaluate(card, state).confirmed_output == 50
    result = evaluator.evaluate(card, replace(state, observed_card_output=('convert', preview, None)))
    assert result.projected_output == preview
    assert result.confirmed_output == confirmed


@pytest.mark.parametrize('genki', [10, 20, 50, 100])
def test_late_motivation_lesson_discounts_only_marginal_shield_value(genki):
    evaluator = EffectHeuristicEvaluator()
    shield = _card('shield', _effect(ProduceExamEffectType.ExamBlock, 8))
    state = BattleState(
        remaining_turns=3, hp=15, genki=genki,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive,
    )
    base = evaluator.evaluate(cast(Any, shield), replace(state, genki=10))
    result = evaluator.evaluate(cast(Any, shield), state)
    assert result.score == pytest.approx(base.score * 10 / genki)
    assert result.projected_output == result.confirmed_output == 0


@pytest.mark.parametrize('override', [
    {'is_exam': True}, {'remaining_turns': 4}, {'remaining_turns_known': False},
    {'genki_known': False}, {'hp_known': False}, {'hp': 8},
    {'archetype': ProduceExamEffectType.ExamReview}, {'playable_count': 2},
])
def test_late_shield_discount_preserves_other_plans_and_survival_uncertainty(override):
    evaluator = EffectHeuristicEvaluator()
    shield = _card('shield', _effect(ProduceExamEffectType.ExamBlock, 8))
    state = BattleState(
        remaining_turns=3, hp=15, genki=50,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive,
    )
    protected = replace(state, **override)
    result = evaluator.evaluate(cast(Any, shield), protected)
    reference = evaluator.evaluate(cast(Any, shield), replace(protected, genki=10))
    assert result.score == reference.score
    assert '余量递减' not in result.reason


@pytest.mark.parametrize('followup', [
    ProduceExamEffectType.ExamPlayableValueAdd,
    ProduceExamEffectType.ExamExtraTurn,
    ProduceExamEffectType.ExamLessonDependBlock,
    ProduceExamEffectType.ExamReviewDependExamBlock,
])
def test_late_shield_discount_keeps_explicit_payoff_or_extra_opportunity(followup):
    evaluator = EffectHeuristicEvaluator()
    card = _card('combo', _effect(ProduceExamEffectType.ExamBlock, 8), _effect(followup, 1000))
    state = BattleState(
        remaining_turns=3, hp=15, genki=50,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive,
    )
    result = evaluator.evaluate(cast(Any, card), state)
    assert '余量递减' not in result.reason


def test_late_shield_discount_leaves_direct_output_unchanged():
    evaluator = EffectHeuristicEvaluator()
    state = BattleState(
        remaining_turns=3, hp=15, genki=50,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive,
    )
    card = _card('mixed', _effect(ProduceExamEffectType.ExamBlock, 8),
                 _effect(ProduceExamEffectType.ExamLesson, 9))
    shield = _card('shield', _effect(ProduceExamEffectType.ExamBlock, 8))
    mixed = evaluator.evaluate(cast(Any, card), state)
    defence = evaluator.evaluate(cast(Any, shield), state)
    assert mixed.score - defence.score == pytest.approx(9)
    assert mixed.projected_output == mixed.confirmed_output == 9


def test_late_shield_discount_does_not_discount_persistent_hp_recovery():
    evaluator = EffectHeuristicEvaluator()
    card = _card('heal', _effect(ProduceExamEffectType.ExamStaminaRecoverFix, 4))
    state = BattleState(
        remaining_turns=3, hp=15, genki=50,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive,
    )
    result = evaluator.evaluate(cast(Any, card), state)
    reference = evaluator.evaluate(cast(Any, card), replace(state, genki=10))
    assert result.score == reference.score
    assert '余量递减' not in result.reason


def test_late_shield_discount_does_not_restore_unknown_post_payment_genki():
    evaluator = EffectHeuristicEvaluator()
    card = _card('shield', _effect(ProduceExamEffectType.ExamBlock, 8), cost=3)
    state = BattleState(
        remaining_turns=3, hp=15, genki=50,
        archetype=ProduceExamEffectType.ExamCardPlayAggressive,
        observed_effects=frozenset({'stamina_consumption_modifier'}),
        observed_effect_values=(('stamina_consumption_modifier', None),),
    )
    result = evaluator.evaluate(cast(Any, card), state)
    assert '余量递减' not in result.reason


@pytest.mark.parametrize('kind', [ProduceExamEffectType.ExamLesson,
                                  ProduceExamEffectType.ExamLessonDependBlock])
@pytest.mark.parametrize('exam', [False, True])
@pytest.mark.parametrize('arrow', ['lesson_value_up', 'lesson_value_down'])
@pytest.mark.parametrize('percent', [None, 10, 75])
def test_visible_score_percent_never_certifies_uncalibrated_output(kind, exam, arrow, percent):
    evaluator = EffectHeuristicEvaluator()
    card = _card('output', _effect(kind, 9 if kind == ProduceExamEffectType.ExamLesson else 1000))
    state = BattleState(remaining_turns=3, hp=20, genki=50,
                        is_exam=exam, exam_score_percent=100 if exam else None)
    reference = evaluator.evaluate(cast(Any, card), state)
    decrease = replace(state, observed_effects=frozenset({arrow}),
                       observed_effect_values=((arrow, percent),))
    result = evaluator.evaluate(cast(Any, card), decrease)
    factor = (1.0 if percent is None else
              1.0 + percent / 100 if arrow == 'lesson_value_up' else
              max(0.0, 1.0 - percent / 100))
    assert result.projected_output == pytest.approx(reference.projected_output * factor)
    assert reference.confirmed_output > 0
    assert result.confirmed_output == 0
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play

    after = simulate_play(decrease, cast(Any, _card('setup')))
    assert not evaluator._output_multiplier_known(after)


@pytest.mark.parametrize('verdict', [TriggerVerdict.ACTIVE, TriggerVerdict.INACTIVE,
                                   TriggerVerdict.UNKNOWN])
def test_plain_extra_turn_updates_search_horizon_without_changing_hud(monkeypatch, verdict):
    from kaa.tasks.produce.new.play_cards.planner import simulator

    turn = _effect(ProduceExamEffectType.ExamExtraTurn, 0, count=0)
    turn.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    state = BattleState(remaining_turns=1, observed_remaining_turns=1, hp=20, genki=10)
    future = simulator.simulate_play(state, cast(Any, _card('turn', turn)))
    assert future.remaining_turns == (2 if verdict == TriggerVerdict.ACTIVE else 1)
    assert future.remaining_turns_known == (verdict == TriggerVerdict.INACTIVE)
    assert future.observed_remaining_turns == state.observed_remaining_turns == 1
    assert future.playable_count == 0  # A turn extension is not another current play.
    assert state.remaining_turns == 1


@pytest.mark.parametrize('field,value', [
    ('effect_value1', 2), ('effect_count', 2), ('_produce_card_grow_effect_ids', None),
    ('_produce_card_grow_effect_ids', '["growth"]'),
    ('_chain_produce_exam_effect_id', 'chain'), ('_produce_card_search_id', 'search'),
])
def test_complex_extra_turn_does_not_guess_search_horizon(field, value):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play

    turn = _effect(ProduceExamEffectType.ExamExtraTurn, 0, count=0)
    turn.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    setattr(turn.produce_exam_effect, field, value)
    state = BattleState(remaining_turns=1, hp=20, genki=10)
    future = simulate_play(state, cast(Any, _card('turn', turn)))
    assert future.remaining_turns == 1
    assert not future.remaining_turns_known


def test_extra_turn_with_extra_play_keeps_followup_shield_useful():
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play

    turn = _effect(ProduceExamEffectType.ExamExtraTurn, 0, count=0)
    turn.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    extra = _effect(ProduceExamEffectType.ExamPlayableValueAdd, 1)
    state = BattleState(remaining_turns=1, hp=20, genki=10)
    after = simulate_play(state, cast(Any, _card('turn-and-play', turn, extra)))
    shield = cast(Any, _card('shield', _effect(ProduceExamEffectType.ExamBlock, 8)))
    evaluator = EffectHeuristicEvaluator()
    assert after.playable_count == 1
    assert evaluator.evaluate(shield, after).score > evaluator.evaluate(shield, state).score
    next_state = simulate_play(after, shield)
    assert next_state.remaining_turns == 2
    assert not next_state.remaining_turns_known
    assert next_state.observed_remaining_turns is None


@pytest.mark.parametrize('projection', [False, True])
def test_unresolved_or_projected_growth_does_not_invent_extra_turn(projection):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play

    turn = _effect(ProduceExamEffectType.ExamExtraTurn, 0, count=0)
    turn.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    state = BattleState(remaining_turns=1, hp=20, genki=10,
        unresolved_growth_card_ids=frozenset() if projection else frozenset({'turn'}),
        projected_card_growth=(('growth', True),) if projection else ())
    future = simulate_play(state, cast(Any, _card('turn', turn)))
    assert future.remaining_turns == 1
    assert not future.remaining_turns_known


@pytest.mark.parametrize('consumption', [0, 1000])
@pytest.mark.parametrize('count', [None, 1])
def test_single_conversion_preview_preserves_payment_and_consumption(consumption, count):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play
    from kaa.tasks.produce.new.play_cards.planner.costs import play_cost
    effect = _single_conversion_effect()
    effect.produce_exam_effect.effect_value1 = 4000
    effect.produce_exam_effect.effect_value2 = consumption
    card = cast(SkillCard, _card('convert', effect))
    card.force_stamina = 1
    state = BattleState(remaining_turns=4, hp=14, genki=149,
        observed_effects=frozenset({'lesson_value_up'}))
    observed = replace(state, observed_card_output=('convert', 167, count))
    evaluator = EffectHeuristicEvaluator()
    assert evaluator.evaluate(card, state).projected_output == 596
    result = evaluator.evaluate(card, observed)
    assert result.projected_output == 167
    assert result.confirmed_output == 0
    assert play_cost(state, card) == play_cost(observed, card)
    before = simulate_play(state, card)
    after = simulate_play(observed, card)
    assert after.observed_card_output is None
    assert before == after
    assert after.genki == (0 if consumption else 149)


@pytest.mark.parametrize('observed,remaining', [(96, 8), (None, 8), (96, 1)])
def test_full_conversion_unknown_payment_prices_only_observed_loss_bound(observed, remaining):
    from kaa.tasks.produce.new.play_cards.planner.simulator import simulate_play_prefix
    effect = _single_conversion_effect()
    effect.produce_exam_effect.effect_value1 = 4000
    effect.produce_exam_effect.effect_value2 = 1000
    card = cast(SkillCard, _card('convert', effect, cost=1))
    state = BattleState(remaining_turns=remaining, hp=29, genki=96,
        observed_genki=observed, is_exam=True, exam_score_percent=1635,
        observed_effects=frozenset({'stamina_consumption_modifier'}),
        observed_effect_values=(('stamina_consumption_modifier', None),),
        observed_card_output=('convert', 1570, None))
    assert simulate_play_prefix(state, card, 0).genki_known is False
    evaluator = EffectHeuristicEvaluator()
    unknown = evaluator.evaluate(card, replace(state, observed_genki=None))
    result = evaluator.evaluate(card, state)
    assert result.projected_output == unknown.projected_output == 1570
    assert result.confirmed_output == unknown.confirmed_output == 0
    expected_loss = 105.6 if observed is not None and remaining > 1 else 0
    assert result.score == pytest.approx(unknown.score - expected_loss)
    assert state.genki == 96


@pytest.mark.parametrize('kind', [ProduceExamEffectType.ExamLessonValueMultiple,
                                 ProduceExamEffectType.ExamLessonValueMultipleDown])
@pytest.mark.parametrize('verdict', [TriggerVerdict.ACTIVE, TriggerVerdict.UNKNOWN,
                                    TriggerVerdict.INACTIVE])
def test_possible_percent_change_invalidates_old_ranking_amount(monkeypatch, kind, verdict):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    state = BattleState(remaining_turns=3, hp=20, genki=50,
                        observed_effects=frozenset({'lesson_value_up'}),
                        observed_effect_values=(('lesson_value_up', 20),))
    future = simulator.simulate_play(state, cast(Any, _card('percent', _effect(kind, 100))))
    values = dict(future.effect_values_for_planning)
    if verdict == TriggerVerdict.INACTIVE:
        assert values['lesson_value_up'] == 20
        assert EffectHeuristicEvaluator._output_multiplier(future) == pytest.approx(1.2)
    else:
        assert values['lesson_value_up'] is values['lesson_value_down'] is None
        assert EffectHeuristicEvaluator._output_multiplier(future) == 1
    assert not EffectHeuristicEvaluator._output_multiplier_known(future)
    assert dict(state.observed_effect_values) == {'lesson_value_up': 20}


@pytest.mark.parametrize('values', [
    (('lesson_value_up', None),), (('lesson_value_up', -1),),
    (('lesson_value_up', True),), (('lesson_value_up', 1000),),
    (('lesson_value_up', 20), ('lesson_value_down', 10)),
])
def test_unusable_percent_does_not_invent_ranking_multiplier(values):
    state = BattleState(remaining_turns=3, hp=20, genki=50, observed_effect_values=values)
    assert EffectHeuristicEvaluator._output_multiplier(state) == 1
    assert not EffectHeuristicEvaluator._output_multiplier_known(state)


@pytest.mark.parametrize('kind', [ProduceExamEffectType.ExamLessonValueMultiple,
                                 ProduceExamEffectType.ExamLessonValueMultipleDown])
@pytest.mark.parametrize('verdict', [TriggerVerdict.ACTIVE, TriggerVerdict.UNKNOWN])
def test_new_unmodeled_percent_cannot_certify_followup_output(monkeypatch, kind, verdict):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: verdict)
    state = BattleState(remaining_turns=3, hp=20, genki=50)
    future = simulator.simulate_play(state, cast(Any, _card('percent', _effect(kind, 100))))
    assert not future.observed_effects
    assert not EffectHeuristicEvaluator._output_multiplier_known(future)
    result = EffectHeuristicEvaluator().evaluate(cast(Any, _card(
        'output', _effect(ProduceExamEffectType.ExamLesson, 10))), future)
    assert result.projected_output == 10
    assert result.confirmed_output == 0


def test_percent_change_only_invalidates_later_scoring_segments():
    card = _card('sequence', _effect(ProduceExamEffectType.ExamLesson, 10),
                 _effect(ProduceExamEffectType.ExamLessonValueMultiple, 100),
                 _effect(ProduceExamEffectType.ExamLesson, 10))
    result = EffectHeuristicEvaluator().evaluate(cast(Any, card),
        BattleState(remaining_turns=3, hp=20, genki=50))
    assert result.projected_output == 20
    assert result.confirmed_output == 10
