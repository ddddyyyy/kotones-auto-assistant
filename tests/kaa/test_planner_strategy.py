from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kaa.db.constants import ProduceExamEffectType, ProducePlanType
from kaa.tasks.produce.new.play_cards.planner import strategy as strategy_module
from kaa.tasks.produce.new.play_cards.planner.evaluator import (
    CardEvaluation,
    EvaluationConfidence,
)
from kaa.tasks.produce.new.play_cards.planner.simulator import can_afford_play, simulate_play
from kaa.tasks.produce.new.play_cards.planner.state import AnomalyStance, AnomalyState, BattleState
from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy


class FakeEvaluator:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def evaluate(self, card, state):
        score = self.scores.get(card._id)
        if score is None:
            return None
        return CardEvaluation(
            card_id=card._id,
            score=score,
            confidence=EvaluationConfidence.MEDIUM,
            reason='test',
        )


class FakeContext:
    def __init__(self, hands) -> None:
        self.hands = hands
        self.committed = None

    def fetch_hands(self):
        return self.hands

    def fetch_remaining_turns(self):
        return 3

    def fetch_hp(self):
        return 12

    def fetch_stamina(self):
        return 4

    def commit(self, card):
        self.committed = card


def _hand(card_id: str, *, available: bool = True):
    return SimpleNamespace(
        available=available,
        card=SimpleNamespace(_id=card_id, name=card_id),
    )


def _effect(kind: ProduceExamEffectType, value: int):
    return SimpleNamespace(
        produce_exam_effect=SimpleNamespace(
            effect_type=kind,
            effect_value1=value,
            effect_value2=0,
            effect_count=1,
            effect_turn=0,
        )
    )


def _planner_hand(card_id: str, *effects, stamina: int = 0, force_stamina: int = 0):
    return SimpleNamespace(
        available=True,
        card=SimpleNamespace(
            _id=card_id,
            name=card_id,
            stamina=stamina,
            force_stamina=force_stamina,
            cost_type=None,
            cost_value=0,
            play_effects=list(effects),
            play_move_position_type='ProduceCardMovePositionType_Grave',
        ),
    )


def test_planner_selects_highest_evaluated_playable_card():
    low = _hand('low')
    high = _hand('high')
    unavailable = _hand('unavailable', available=False)
    ctx = FakeContext([low, high, unavailable])
    strategy = PlannerStrategy(FakeEvaluator({
        'low': 10,
        'high': 100,
        'unavailable': 1000,
    }))

    assert strategy.on_action(cast(Any, ctx))
    assert ctx.committed is high


def test_planner_logs_previous_play_feedback_on_next_decision(caplog):
    first = _planner_hand('first')
    second = _planner_hand('second')
    strategy = PlannerStrategy(FakeEvaluator({'first': 100, 'second': 10}))
    first_ctx = FakeContext([first, second])

    assert strategy.on_action(cast(Any, first_ctx))
    assert strategy._pending_play is not None

    caplog.set_level('INFO', logger=strategy_module.__name__)
    second_ctx = FakeContext([second])
    assert strategy.on_action(cast(Any, second_ctx))

    assert 'card=first status=confirmed evidence=card_left_hand' in caplog.text


def test_enabled_decision_trace_does_not_change_selection_or_require_screenshots(monkeypatch, tmp_path):
    import json
    from kaa.tasks.produce.new.play_cards.planner.decision_trace import TRACE_ENV

    monkeypatch.setenv(TRACE_ENV, str(tmp_path))
    first = _planner_hand('first')
    second = _planner_hand('second')
    strategy = PlannerStrategy(FakeEvaluator({'first': 100, 'second': 10}))
    first_ctx = FakeContext([first, second])
    assert strategy.on_action(cast(Any, first_ctx))
    assert first_ctx.committed is first
    second_ctx = FakeContext([second])
    assert strategy.on_action(cast(Any, second_ctx))
    assert second_ctx.committed is second
    path, = tmp_path.glob('decisions-*.jsonl')
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert [event['event'] for event in events] == ['decision', 'feedback', 'decision']
    assert events[0]['selected']['card_id'] == 'first'
    assert events[1]['feedback']['evidence'] == 'card_left_hand'
    assert events[0]['state']['observed_remaining_turns'] == 3
    assert events[0]['state']['observed_hp'] == 12
    assert events[1]['after_state']['observed_genki'] == 4


def test_planner_does_not_choose_known_unaffordable_first_card(monkeypatch):
    expensive = _planner_hand('expensive', stamina=6)
    cheap = _planner_hand('cheap', stamina=0)
    strategy = PlannerStrategy(FakeEvaluator({'expensive': 1000, 'cheap': 1}))
    state = BattleState(remaining_turns=1, hp=1, genki=4)
    monkeypatch.setattr(strategy, '_build_observed_state', lambda ctx, hands: state)
    ctx = FakeContext([expensive, cheap])
    assert strategy.on_action(cast(Any, ctx))
    assert ctx.committed is cheap


def test_planner_falls_back_when_all_first_cards_are_unaffordable(monkeypatch):
    expensive = _planner_hand('expensive', force_stamina=6)
    strategy = PlannerStrategy(FakeEvaluator({'expensive': 1000}))
    state = BattleState(remaining_turns=1, hp=1, genki=50)
    monkeypatch.setattr(strategy, '_build_observed_state', lambda ctx, hands: state)
    ctx = FakeContext([expensive])
    assert not strategy.on_action(cast(Any, ctx))
    assert ctx.committed is None
    assert strategy._pending_play is None


def test_planner_keeps_first_card_when_resources_are_unknown(monkeypatch):
    card = _planner_hand('unknown-cost', stamina=6)
    strategy = PlannerStrategy(FakeEvaluator({'unknown-cost': 1000}))
    state = BattleState(remaining_turns=1, hp=0, genki=0, hp_known=False, genki_known=False)
    monkeypatch.setattr(strategy, '_build_observed_state', lambda ctx, hands: state)
    ctx = FakeContext([card])
    assert strategy.on_action(cast(Any, ctx))
    assert ctx.committed is card


def test_unchanged_paid_request_does_not_stack_inferred_stance(caplog):
    # Live failure: the same paid concentration card remained in hand with
    # unchanged HP, genki and CLEAR gap, but its stance had been applied twice.
    card = _planner_hand('paid-stance', _effect(ProduceExamEffectType.ExamConcentration, 1), stamina=6)
    strategy = PlannerStrategy(FakeEvaluator({'paid-stance': 100}))
    class UnchangedContext(FakeContext):
        def fetch_score_gap(self):
            return 90

        def fetch_score_target_is_final(self):
            return False

    ctx = UnchangedContext([card])
    assert strategy.on_action(cast(Any, ctx))
    assert strategy._memory.anomaly.stance is AnomalyStance.AGGRESSIVE
    caplog.set_level('WARNING', logger=strategy_module.__name__)
    assert strategy.on_action(cast(Any, ctx))
    assert strategy._memory.anomaly.stance is AnomalyStance.AGGRESSIVE
    assert strategy._memory.anomaly.stance_change_count == 1
    assert strategy._memory.plays_this_turn == 1
    assert strategy._memory.discard == ['paid-stance']
    assert 'discarded speculative effects' in caplog.text


def test_planner_secures_clear_before_late_turn_setup(monkeypatch):
    class OutputEvaluator:
        def evaluate(self, card, state):
            is_output = card._id == 'guaranteed-output'
            return CardEvaluation(
                card_id=card._id,
                score=5 if is_output else 100,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test',
                confirmed_output=4.5 if is_output else 0,
            )

    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    setup = _planner_hand('setup')
    output = _planner_hand('guaranteed-output')
    ctx = FakeContext([setup, output])
    setattr(ctx, 'fetch_remaining_turns', lambda: 4)
    setattr(ctx, 'fetch_score_gap', lambda: 3)
    setattr(ctx, 'fetch_score_target_is_final', lambda: False)
    strategy = PlannerStrategy(cast(Any, OutputEvaluator()))
    strategy._memory.anomaly = AnomalyState(extra_play_count=1)

    assert strategy.on_action(cast(Any, ctx))
    assert ctx.committed is output

    early_ctx = FakeContext([setup, output])
    setattr(early_ctx, 'fetch_remaining_turns', lambda: 5)
    setattr(early_ctx, 'fetch_score_gap', lambda: 3)
    setattr(early_ctx, 'fetch_score_target_is_final', lambda: False)
    early_strategy = PlannerStrategy(cast(Any, OutputEvaluator()))
    early_strategy._memory.anomaly = AnomalyState(extra_play_count=1)

    assert early_strategy.on_action(cast(Any, early_ctx))
    assert early_ctx.committed is setup


def test_tiny_fractional_output_does_not_force_a_clear_choice(monkeypatch):
    class TinyOutputEvaluator:
        def evaluate(self, card, state):
            is_output = card._id == 'tiny-output'
            return CardEvaluation(
                card_id=card._id,
                score=5 if is_output else 100,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test',
                confirmed_output=1.125 if is_output else 0,
            )

    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    setup = _planner_hand('setup')
    output = _planner_hand('tiny-output')
    ctx = FakeContext([setup, output])
    setattr(ctx, 'fetch_score_gap', lambda: 1)
    setattr(ctx, 'fetch_score_target_is_final', lambda: False)

    assert PlannerStrategy(cast(Any, TinyOutputEvaluator())).on_action(cast(Any, ctx))
    assert ctx.committed is setup


def test_late_clear_progress_beats_surplus_defence_without_hard_filter(monkeypatch):
    class EndgameEvaluator:
        def evaluate(self, card, state):
            score, output = {
                'small-output': (28.5, 12.0),
                'larger-output': (26.7, 24.0),
                'defence': (30.0, 0.0),
                'strong-setup': (50.0, 0.0),
            }[card._id]
            return CardEvaluation(
                card_id=card._id,
                score=score,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test',
                confirmed_output=output,
            )

    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    cards = [_planner_hand(card_id) for card_id in (
        'small-output', 'larger-output', 'defence',
    )]
    ctx = FakeContext(cards)
    setattr(ctx, 'fetch_remaining_turns', lambda: 2)
    setattr(ctx, 'fetch_score_gap', lambda: 58)
    setattr(ctx, 'fetch_score_target_is_final', lambda: False)
    setattr(ctx, 'fetch_hp', lambda: 6)
    setattr(ctx, 'fetch_stamina', lambda: 23)

    assert PlannerStrategy(cast(Any, EndgameEvaluator())).on_action(cast(Any, ctx))
    assert ctx.committed is cards[1]

    # A genuinely stronger setup may still win; the progress rule is not a
    # blanket ban on non-scoring cards.
    setup = _planner_hand('strong-setup')
    setup_ctx = FakeContext(cards + [setup])
    setattr(setup_ctx, 'fetch_remaining_turns', lambda: 2)
    setattr(setup_ctx, 'fetch_score_gap', lambda: 58)
    setattr(setup_ctx, 'fetch_score_target_is_final', lambda: False)
    assert PlannerStrategy(cast(Any, EndgameEvaluator())).on_action(cast(Any, setup_ctx))
    assert setup_ctx.committed is setup


def test_three_turn_near_clear_output_beats_surplus_defence(monkeypatch):
    class NearClearEvaluator:
        def evaluate(self, card, state):
            output = card._id == 'appeal'
            return CardEvaluation(
                card_id=card._id,
                score=2.0 if output else 9.6,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test',
                confirmed_output=18.0 if output else 0.0,
            )

    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    defence, appeal = _planner_hand('defence'), _planner_hand('appeal')
    ctx = FakeContext([defence, appeal])
    setattr(ctx, 'fetch_remaining_turns', lambda: 3)
    setattr(ctx, 'fetch_score_gap', lambda: 19)
    setattr(ctx, 'fetch_score_target_is_final', lambda: False)

    assert PlannerStrategy(cast(Any, NearClearEvaluator())).on_action(cast(Any, ctx))
    assert ctx.committed is appeal


def test_planner_falls_back_when_no_card_has_evaluation():
    ctx = FakeContext([_hand('unknown')])
    strategy = PlannerStrategy(FakeEvaluator({}))

    assert not strategy.on_action(cast(Any, ctx))
    assert ctx.committed is None


def test_planner_falls_back_when_no_playable_card_is_recognized():
    ctx = FakeContext([None, _hand('disabled', available=False)])
    strategy = PlannerStrategy(FakeEvaluator({'disabled': 100}))

    assert not strategy.on_action(cast(Any, ctx))
    assert ctx.committed is None


def test_planner_passes_current_archetype_and_memory_to_evaluator(monkeypatch):
    monkeypatch.setattr(
        strategy_module,
        'produce_session',
        lambda: SimpleNamespace(archetype=ProduceExamEffectType.ExamFullPower),
    )
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))
    ctx = FakeContext([_hand('card')])

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.archetype is ProduceExamEffectType.ExamFullPower
    assert state.plan_type is ProducePlanType.Plan3
    assert state.buffs is strategy._memory.buffs


def test_strategy_preserves_verified_large_genki_from_page(monkeypatch):
    monkeypatch.setattr(
        strategy_module,
        'produce_session',
        lambda: SimpleNamespace(archetype=ProduceExamEffectType.ExamFullPower),
    )
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))
    strategy._last_move_state = BattleState(
        remaining_turns=7, hp=5, genki=13, plan_type=ProducePlanType.Plan3,
    )
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_stamina', lambda: 713)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.genki == 713
    assert state.genki_known


def test_hp_ocr_stray_prefix_uses_prior_hp(monkeypatch, caplog):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}), is_exam=True)
    strategy._last_move_state = BattleState(remaining_turns=7, hp=21, genki=0)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_exam_hp', lambda: 921)
    setattr(ctx, 'fetch_exam_stamina', lambda: 0)

    caplog.set_level('WARNING', logger=strategy_module.__name__)
    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.hp == 21
    assert state.hp_known
    assert 'corrected suspect hp OCR 921 to 21' in caplog.text


def test_hp_ocr_marks_uncorroborated_three_digit_value_unknown(monkeypatch, caplog):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))
    strategy._last_move_state = BattleState(remaining_turns=7, hp=2, genki=0)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_hp', lambda: 920)

    caplog.set_level('WARNING', logger=strategy_module.__name__)
    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.hp == 0
    assert not state.hp_known
    assert 'discarded suspect hp OCR 920' in caplog.text


def test_hp_ocr_keeps_corroborated_large_hp(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))
    strategy._last_move_state = BattleState(remaining_turns=7, hp=120, genki=0)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_hp', lambda: 121)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.hp == 121
    assert state.hp_known


def test_plan3_genki_ocr_keeps_large_value_without_corrobating_prior(monkeypatch):
    monkeypatch.setattr(
        strategy_module,
        'produce_session',
        lambda: SimpleNamespace(archetype=ProduceExamEffectType.ExamFullPower),
    )
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))
    strategy._last_move_state = BattleState(
        remaining_turns=7, hp=5, genki=230, plan_type=ProducePlanType.Plan3,
    )
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_stamina', lambda: 319)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.genki == 319


def test_other_plan_does_not_trim_large_genki_ocr(monkeypatch):
    monkeypatch.setattr(
        strategy_module,
        'produce_session',
        lambda: SimpleNamespace(archetype=ProduceExamEffectType.ExamCardPlayAggressive),
    )
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))
    strategy._last_move_state = BattleState(
        remaining_turns=7, hp=5, genki=13, plan_type=ProducePlanType.Plan2,
    )
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_stamina', lambda: 713)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.genki == 713


def test_observed_hand_includes_unavailable_and_unrecognized_card_slots(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    ctx = FakeContext([
        _hand('playable'),
        _hand('unavailable', available=False),
        None,
    ])
    strategy = PlannerStrategy(FakeEvaluator({'playable': 1}))

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.hand_size == 3
    assert state.hand_card_ids == ('playable', 'unavailable')
    assert strategy._memory.known_cards['unavailable'] == 1


def test_simulated_play_reduces_visible_hand_size():
    card = _planner_hand('first').card
    state = BattleState(
        remaining_turns=4,
        hp=20,
        genki=0,
        hand_card_ids=('first', 'second'),
        hand_size=3,
    )

    future = simulate_play(state, cast(Any, card))

    assert future.hand_size == 2
    assert future.hand_card_ids == ('second',)


def test_exam_never_treats_background_ocr_as_a_lesson_score_gap(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_score_gap', lambda: 123)
    setattr(ctx, 'fetch_score_target_is_final', lambda: True)
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}), is_exam=True)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.score_gap is None
    assert state.score_target_is_final is None
    assert state.is_exam


def test_exam_passes_confirmed_standings_to_planner(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_exam_standings', lambda: SimpleNamespace(
        own_score=7742, leader_score=13195, own_is_leader=False,
    ))
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}), is_exam=True)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.current_score == 7742
    assert state.target_score == 13196
    assert state.score_gap is None


def test_exam_leader_has_no_overtake_target(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_exam_standings', lambda: SimpleNamespace(
        own_score=1400, leader_score=1400, own_is_leader=True,
    ))
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}), is_exam=True)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.current_score == 1400
    assert state.target_score is None


def test_exam_near_finish_rewards_confirmed_output_without_score_unit_conversion():
    strategy = PlannerStrategy(FakeEvaluator({}), is_exam=True)
    state = BattleState(
        remaining_turns=2, hp=10, genki=0, is_exam=True,
        current_score=100, target_score=1100,
    )
    card = _planner_hand('score')
    uncertain = CardEvaluation(
        'uncertain', 50, EvaluationConfidence.MEDIUM, 'uncertain',
        projected_output=20, confirmed_output=0,
    )
    enough = CardEvaluation(
        'enough', 20, EvaluationConfidence.MEDIUM, 'enough',
        projected_output=14, confirmed_output=14,
    )
    short = CardEvaluation(
        'short', 25, EvaluationConfidence.MEDIUM, 'short',
        projected_output=10, confirmed_output=10,
    )
    candidates = [(card, uncertain), (card, enough), (card, short)]

    ranked = strategy._prefer_exam_catch_up_output(cast(Any, candidates), state)

    assert [round(evaluation.score, 1) for _, evaluation in ranked] == [50, 34.9, 38.5]
    assert '考试落后时优先确定产分' in ranked[1][1].reason
    assert strategy._prefer_exam_catch_up_output(
        cast(Any, candidates), replace(state, remaining_turns=4),
    ) == candidates
    assert strategy._prefer_exam_catch_up_output(
        cast(Any, candidates), replace(state, target_score=None),
    ) == candidates
    assert strategy._prefer_exam_catch_up_output(
        cast(Any, candidates), replace(state, target_score=100),
    ) == candidates


def test_exam_catch_up_bonus_is_bounded_and_lesson_unchanged():
    strategy = PlannerStrategy(FakeEvaluator({}), is_exam=True)
    card = _planner_hand('score')
    big = CardEvaluation(
        'big', 200, EvaluationConfidence.MEDIUM, 'big',
        projected_output=1000, confirmed_output=1000,
    )
    candidates = [(card, big)]
    exam = BattleState(
        remaining_turns=1, hp=10, genki=0, is_exam=True,
        current_score=0, target_score=50000,
    )

    ranked = strategy._prefer_exam_catch_up_output(cast(Any, candidates), exam)

    assert ranked[0][1].score == 235
    assert strategy._prefer_exam_catch_up_output(
        cast(Any, candidates), replace(exam, is_exam=False),
    ) == candidates


def test_exam_uses_shifted_resource_boxes_instead_of_lesson_boxes(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_exam_hp', lambda: 27)
    setattr(ctx, 'fetch_exam_stamina', lambda: 29)
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}), is_exam=True)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.hp == 27
    assert state.genki == 29


def test_lesson_passes_perfect_target_stage_to_evaluator(monkeypatch, caplog):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_score_gap', lambda: 8)
    setattr(ctx, 'fetch_score_target_is_final', lambda: True)
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))

    caplog.set_level('DEBUG', logger=strategy_module.__name__)
    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.score_gap == 8
    assert state.score_target_is_final is True
    assert 'Planner lesson target: PERFECT, remaining score: 8' in caplog.text


def test_planner_reconciles_numeric_hud_effect_observations(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    monkeypatch.setattr(
        strategy_module,
        'observe_effect_icons',
        lambda _screen: [SimpleNamespace(effect='lesson_buff', value=10)],
    )
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_screen', lambda: object())
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.observed_effects == frozenset({'lesson_buff'})
    assert state.buffs.lesson_buff == 10


def test_planner_requires_repeated_hud_absence_before_confirming_zero(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    monkeypatch.setattr(strategy_module, 'observe_effect_icons', lambda _screen: [])
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_screen', lambda: object())
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))

    first = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))
    second = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert 'lesson_buff' not in first.confirmed_absent_effects
    assert second.confirmed_absent_effects == frozenset({
        'lesson_buff', 'parameter_buff', 'review', 'full_power',
    })


def test_missing_screen_does_not_count_as_confirmed_hud_absence(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    ctx = FakeContext([_hand('card')])
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))

    strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))
    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.confirmed_absent_effects == frozenset()


class _OrderingEvaluator:
    def evaluate(self, card, state):
        score = 5.0 if card._id == 'setup' else 10.0
        if card._id == 'output' and state.anomaly.stance.value == 'aggressive':
            score = 30.0
        return CardEvaluation(
            card_id=card._id,
            score=score,
            confidence=EvaluationConfidence.LOW,
            reason='ordering',
        )


@pytest.mark.parametrize('full_power', [False, True])
def test_planner_uses_shallow_search_to_order_two_card_combo(full_power):
    setup = _planner_hand(
        'setup',
        _effect(ProduceExamEffectType.ExamConcentration, 1),
    )
    output = _planner_hand('output')
    ctx = FakeContext([output, setup])
    strategy = PlannerStrategy(_OrderingEvaluator())
    strategy._memory.anomaly = AnomalyState(
        full_power_active=full_power,
        extra_play_count=1,
    )

    assert strategy.on_action(cast(Any, ctx))
    # Strong setup is useful normally, but cannot change the full-power stance.
    assert ctx.committed is (output if full_power else setup)


def test_lesson_preserves_hp_for_multiple_scoring_plays_before_clear():
    setup = _planner_hand('setup', stamina=7)
    output = _planner_hand('output', stamina=4)
    strategy = PlannerStrategy(FakeEvaluator({}))
    state = BattleState(
        remaining_turns=7, hp=14, genki=0,
        score_gap=90, score_target_is_final=False,
    )
    candidates = [
        (setup, CardEvaluation('setup', 74, EvaluationConfidence.HIGH, 'setup')),
        (output, CardEvaluation(
            'output', 8, EvaluationConfidence.MEDIUM, 'output',
            projected_output=14, confirmed_output=14,
        )),
    ]

    ranked = strategy._protect_lesson_scoring_budget(cast(Any, candidates), state)

    assert ranked[0][1].score == 7
    assert '保留后续产分体力' in ranked[0][1].reason
    assert ranked[1][1].score == 8


def test_lesson_keeps_setup_priority_with_adequate_hp_or_short_gap():
    setup = _planner_hand('setup', stamina=7)
    output = _planner_hand('output', stamina=4)
    strategy = PlannerStrategy(FakeEvaluator({}))
    candidates = [
        (setup, CardEvaluation('setup', 74, EvaluationConfidence.HIGH, 'setup')),
        (output, CardEvaluation(
            'output', 8, EvaluationConfidence.MEDIUM, 'output',
            projected_output=14, confirmed_output=14,
        )),
    ]
    rich = BattleState(
        remaining_turns=7, hp=30, genki=0,
        score_gap=90, score_target_is_final=False,
    )
    near_clear = BattleState(
        remaining_turns=7, hp=14, genki=0,
        score_gap=20, score_target_is_final=False,
    )

    assert strategy._protect_lesson_scoring_budget(cast(Any, candidates), rich)[0][1].score == 74
    assert strategy._protect_lesson_scoring_budget(cast(Any, candidates), near_clear)[0][1].score == 74


def test_lesson_does_not_blame_setup_for_an_existing_scoring_shortage():
    setup = _planner_hand('setup')
    output = _planner_hand('output', stamina=8)
    strategy = PlannerStrategy(FakeEvaluator({}))
    state = BattleState(
        remaining_turns=7, hp=14, genki=0,
        score_gap=90, score_target_is_final=False,
    )
    candidates = [
        (setup, CardEvaluation('setup', 74, EvaluationConfidence.HIGH, 'setup')),
        (output, CardEvaluation(
            'output', 8, EvaluationConfidence.MEDIUM, 'output',
            projected_output=14, confirmed_output=14,
        )),
    ]

    ranked = strategy._protect_lesson_scoring_budget(cast(Any, candidates), state)

    assert ranked[0][1].score == 74


def test_exam_last_three_turns_prefers_certain_output_over_single_play_defence():
    defence = _planner_hand('defence', _effect(ProduceExamEffectType.ExamBlock, 7))
    output = _planner_hand('output', _effect(ProduceExamEffectType.ExamLesson, 9))
    strategy = PlannerStrategy(FakeEvaluator({}), is_exam=True)
    state = BattleState(remaining_turns=3, hp=5, genki=21, is_exam=True)
    candidates = [
        (defence, CardEvaluation('defence', 23, EvaluationConfidence.MEDIUM, 'defence')),
        (output, CardEvaluation(
            'output', 8, EvaluationConfidence.MEDIUM, 'output',
            projected_output=9, confirmed_output=9,
        )),
    ]

    ranked = strategy._protect_exam_finish(cast(Any, candidates), state)

    assert ranked[0][1].score == 7
    assert '考试末段优先确定产分' in ranked[0][1].reason
    assert ranked[1][1].score == 8
    early = replace(state, remaining_turns=4)
    assert strategy._protect_exam_finish(cast(Any, candidates), early)[0][1].score == 23


def test_exam_finish_keeps_end_of_turn_output_and_same_turn_combo():
    review = _planner_hand('review', _effect(ProduceExamEffectType.ExamReview, 7))
    output = _planner_hand('output', _effect(ProduceExamEffectType.ExamLesson, 9))
    strategy = PlannerStrategy(FakeEvaluator({}), is_exam=True)
    candidates = [
        (review, CardEvaluation('review', 23, EvaluationConfidence.MEDIUM, 'review')),
        (output, CardEvaluation(
            'output', 8, EvaluationConfidence.MEDIUM, 'output',
            projected_output=9, confirmed_output=9,
        )),
    ]
    single = BattleState(remaining_turns=3, hp=5, genki=21, is_exam=True)
    combo = replace(single, playable_count=2)

    assert strategy._protect_exam_finish(cast(Any, candidates), single)[0][1].score == 23
    assert strategy._protect_exam_finish(cast(Any, candidates), combo)[0][1].score == 23


def test_exam_three_turns_keeps_non_defensive_full_power_setup():
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamFullPowerPoint, 2))
    output = _planner_hand('output', _effect(ProduceExamEffectType.ExamLesson, 9))
    strategy = PlannerStrategy(FakeEvaluator({}), is_exam=True)
    state = BattleState(remaining_turns=3, hp=5, genki=21, is_exam=True)
    candidates = [
        (setup, CardEvaluation('setup', 23, EvaluationConfidence.MEDIUM, 'setup')),
        (output, CardEvaluation(
            'output', 8, EvaluationConfidence.MEDIUM, 'output',
            projected_output=9, confirmed_output=9,
        )),
    ]

    assert strategy._protect_exam_finish(cast(Any, candidates), state)[0][1].score == 23


def test_followup_uses_genki_to_pay_green_cost():
    expensive = _planner_hand('expensive', stamina=12)
    cheap = _planner_hand('cheap', stamina=0)
    state = BattleState(remaining_turns=3, hp=10, genki=4, playable_count=2)
    strategy = PlannerStrategy(FakeEvaluator({'expensive': 20, 'cheap': 10}))
    candidates = [
        (expensive, strategy._evaluator.evaluate(expensive.card, state)),
        (cheap, strategy._evaluator.evaluate(cheap.card, state)),
    ]
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)

    assert can_afford_play(simulate_play(state, cast(Any, expensive.card)), cast(Any, cheap.card)) is True
    assert can_afford_play(simulate_play(state, cast(Any, cheap.card)), cast(Any, expensive.card)) is True
    # The initial resource pool pays 12 green cost (4 genki + 8 HP).
    assert ranked[0][1].score == 29
    assert ranked[1][1].score == 28


def test_unaffordable_followup_does_not_receive_two_step_bonus():
    drain = _planner_hand('drain', force_stamina=10)
    finisher = _planner_hand('finisher', force_stamina=5)
    state = BattleState(remaining_turns=3, hp=12, genki=4, playable_count=2)
    strategy = PlannerStrategy(FakeEvaluator({'drain': 10, 'finisher': 20}))
    candidates = [
        (drain, strategy._evaluator.evaluate(drain.card, state)),
        (finisher, strategy._evaluator.evaluate(finisher.card, state)),
    ]
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)

    assert ranked[0][1].score == 10
    assert ranked[1][1].score == 20


def test_affordability_accounts_for_genki_red_cost_and_stance():
    card = cast(Any, _planner_hand('mixed', stamina=4, force_stamina=3).card)
    normal = BattleState(remaining_turns=2, hp=3, genki=4)
    aggressive = BattleState(
        remaining_turns=2, hp=3, genki=4,
        anomaly=AnomalyState(stance=AnomalyStance.AGGRESSIVE),
    )

    assert can_afford_play(normal, card) is True
    assert can_afford_play(aggressive, card) is False
    assert can_afford_play(BattleState(remaining_turns=2, hp=3, genki=4, hp_known=False), card) is None


def test_affordability_uses_known_bounds_when_one_hud_value_is_missing():
    free = cast(Any, _planner_hand('free').card)
    green = cast(Any, _planner_hand('green', stamina=5).card)
    red = cast(Any, _planner_hand('red', force_stamina=6).card)
    hp_missing = BattleState(
        remaining_turns=2, hp=0, genki=5, hp_known=False
    )
    genki_missing = BattleState(
        remaining_turns=2, hp=5, genki=0, genki_known=False
    )

    assert can_afford_play(hp_missing, free) is True
    assert can_afford_play(hp_missing, green) is True
    assert can_afford_play(hp_missing, red) is None
    assert can_afford_play(genki_missing, green) is True
    assert can_afford_play(genki_missing, red) is False


def test_lookahead_keeps_provably_payable_combo_when_hp_ocr_is_missing():
    first = _planner_hand('first')
    second = _planner_hand('second', stamina=5)
    state = BattleState(
        remaining_turns=2, hp=0, genki=8, hp_known=False,
        playable_count=2,
    )
    strategy = PlannerStrategy(FakeEvaluator({'first': 10, 'second': 20}))
    candidates = [
        (first, strategy._evaluator.evaluate(first.card, state)),
        (second, strategy._evaluator.evaluate(second.card, state)),
    ]

    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)

    assert ranked[0][1].score == 28


def test_lookahead_updates_only_confirmed_perfect_gap():
    class GapEvaluator:
        def __init__(self):
            self.observed = []

        def evaluate(self, card, state):
            self.observed.append((card._id, state.score_gap))
            return CardEvaluation(
                card_id=card._id,
                score=10,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test',
                projected_output=20 if card._id == 'first' else 0,
                confirmed_output=20 if card._id == 'first' else 0,
            )

    evaluator = GapEvaluator()
    strategy = PlannerStrategy(cast(Any, evaluator))
    first, second = _planner_hand('first'), _planner_hand('second')
    state = BattleState(
        remaining_turns=2, hp=20, genki=10, playable_count=2,
        score_gap=30, score_target_is_final=True,
    )
    candidates = [
        (first, evaluator.evaluate(first.card, state)),
        (second, evaluator.evaluate(second.card, state)),
    ]

    strategy._add_shallow_lookahead(cast(Any, candidates), state)

    assert ('second', 10) in evaluator.observed


def test_lookahead_drops_unknown_gap_after_crossing_clear():
    class GapEvaluator:
        def __init__(self):
            self.observed = []

        def evaluate(self, card, state):
            self.observed.append((card._id, state.score_gap, state.score_target_is_final))
            return CardEvaluation(
                card_id=card._id, score=10,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test', confirmed_output=20 if card._id == 'first' else 0,
            )

    evaluator = GapEvaluator()
    strategy = PlannerStrategy(cast(Any, evaluator))
    first, second = _planner_hand('first'), _planner_hand('second')
    state = BattleState(
        remaining_turns=2, hp=20, genki=10, playable_count=2,
        score_gap=8, score_target_is_final=False,
    )
    candidates = [
        (first, evaluator.evaluate(first.card, state)),
        (second, evaluator.evaluate(second.card, state)),
    ]

    strategy._add_shallow_lookahead(cast(Any, candidates), state)

    assert ('second', None, None) in evaluator.observed


def test_lookahead_keeps_clear_gap_after_fractional_output():
    class GapEvaluator:
        def __init__(self):
            self.observed = []

        def evaluate(self, card, state):
            self.observed.append((card._id, state.score_gap, state.score_target_is_final))
            return CardEvaluation(
                card_id=card._id, score=10,
                confidence=EvaluationConfidence.MEDIUM,
                reason='test', confirmed_output=1.125 if card._id == 'first' else 0,
            )

    evaluator = GapEvaluator()
    strategy = PlannerStrategy(cast(Any, evaluator))
    first, second = _planner_hand('first'), _planner_hand('second')
    state = BattleState(
        remaining_turns=2, hp=20, genki=10, playable_count=2,
        score_gap=1, score_target_is_final=False,
    )
    candidates = [
        (first, evaluator.evaluate(first.card, state)),
        (second, evaluator.evaluate(second.card, state)),
    ]

    strategy._add_shallow_lookahead(cast(Any, candidates), state)

    assert ('second', 1, False) in evaluator.observed


@pytest.mark.parametrize('stamina', [0, 6])
def test_unconfirmed_retries_do_not_accumulate_invisible_effects(stamina, caplog):
    card = _planner_hand('stance', _effect(ProduceExamEffectType.ExamConcentration, 1), stamina=stamina)
    strategy = PlannerStrategy(FakeEvaluator({'stance': 100}))
    ctx = FakeContext([card])
    caplog.set_level('INFO', logger=strategy_module.__name__)
    for _ in range(4):
        assert strategy.on_action(cast(Any, ctx))
        assert strategy._memory.anomaly.stance_change_count == 1
        assert strategy._memory.plays_this_turn == 1
        assert strategy._memory.discard == ['stance']
    assert 'status=unconfirmed evidence=insufficient_evidence' in caplog.text
    assert 'status=failed' not in caplog.text


def test_move_dialog_retains_inferred_zero_cost_effect():
    card = _planner_hand('stance', _effect(ProduceExamEffectType.ExamConcentration, 1))
    strategy = PlannerStrategy(FakeEvaluator({'stance': 100}))
    ctx = FakeContext([card])
    assert strategy.on_action(cast(Any, ctx))
    strategy.rank_move_cards([])
    assert strategy.on_action(cast(Any, ctx))
    assert strategy._memory.plays_this_turn == 2


def test_unknown_turns_use_short_horizon_and_recover_long_course():
    strategy = PlannerStrategy(FakeEvaluator({}))
    ctx = FakeContext([])
    ctx.fetch_remaining_turns = lambda: None
    for _ in range(3):
        state = strategy._build_observed_state(cast(Any, ctx), [])
        assert state.remaining_turns == 1
        assert not state.remaining_turns_known
    ctx.fetch_remaining_turns = lambda: 12
    strategy._build_observed_state(cast(Any, ctx), [])
    state = strategy._build_observed_state(cast(Any, ctx), [])
    assert state.remaining_turns_known and state.remaining_turns == 12
    ctx.fetch_remaining_turns = lambda: None
    state = strategy._build_observed_state(cast(Any, ctx), [])
    assert not state.remaining_turns_known and state.remaining_turns == 1
    assert strategy._memory.previous_remaining_turns == 12



def test_first_recovered_late_turn_reading_does_not_invent_elapsed_turns():
    strategy = PlannerStrategy(FakeEvaluator({}))
    ctx = FakeContext([])
    ctx.fetch_remaining_turns = lambda: None
    strategy._build_observed_state(cast(Any, ctx), [])
    strategy._memory.anomaly = AnomalyState(full_power_points=5)
    ctx.fetch_remaining_turns = lambda: 1
    strategy._build_observed_state(cast(Any, ctx), [])
    state = strategy._build_observed_state(cast(Any, ctx), [])
    assert state.remaining_turns_known and state.remaining_turns == 1
    assert strategy._memory.anomaly.full_power_points == 5



def test_unaccepted_first_turn_reading_does_not_decay_buff_duration():
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    strategy = PlannerStrategy(FakeEvaluator({}))
    ctx = FakeContext([])
    ctx.fetch_remaining_turns = lambda: None
    strategy._build_observed_state(cast(Any, ctx), [])
    strategy._memory.buffs = ExamBuffState(parameter_buff_turns=3)
    ctx.fetch_remaining_turns = lambda: 1
    pending = strategy._build_observed_state(cast(Any, ctx), [])
    assert not pending.remaining_turns_known
    assert strategy._memory.buffs.parameter_buff_turns == 3
    recovered = strategy._build_observed_state(cast(Any, ctx), [])
    assert recovered.remaining_turns_known and recovered.remaining_turns == 1
    assert strategy._memory.buffs.parameter_buff_turns == 3


@pytest.mark.parametrize('is_exam,expected', [(True, 12), (False, 3)])
def test_ranked_planner_uses_its_own_turn_reader(is_exam, expected):
    class Context(FakeContext):
        def fetch_exam_remaining_turns(self):
            return 12
    ctx = Context([])
    planner = PlannerStrategy(is_exam=is_exam)
    state = planner._build_observed_state(cast(Any, ctx), [])
    assert state.observed_remaining_turns == expected
