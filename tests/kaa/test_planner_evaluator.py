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
        remaining_turns=2, hp=12, genki=9, is_exam=True,
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
