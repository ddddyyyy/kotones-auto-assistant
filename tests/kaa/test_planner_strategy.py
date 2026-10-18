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
    assert [event['event'] for event in events] == ['candidate_screening', 'decision', 'feedback', 'candidate_screening', 'decision']
    assert [e['decision_sequence'] for e in events if e['event']=='candidate_screening'] == [1, 2]
    events = [e for e in events if e['event']!='candidate_screening']
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


@pytest.mark.parametrize('is_exam', [False, True])
@pytest.mark.parametrize('prior,observed', [(21, 921), (2, 920), (21, 121), (20, 29)])
def test_verified_hp_observation_is_not_rewritten_by_history(monkeypatch, prior, observed, is_exam):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}), is_exam=is_exam)
    strategy._last_move_state = BattleState(remaining_turns=7, hp=prior, genki=0)
    ctx = FakeContext([_hand('card')])
    # These are page-layer results, after same-crop verification, not raw OCR.
    setattr(ctx, 'fetch_exam_hp' if is_exam else 'fetch_hp', lambda: observed)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.observed_hp == observed
    assert state.hp == observed
    assert state.hp_known


@pytest.mark.parametrize('is_exam', [False, True])
def test_unknown_hp_observation_is_not_recovered_from_history(monkeypatch, is_exam):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}), is_exam=is_exam)
    strategy._last_move_state = BattleState(remaining_turns=7, hp=21, genki=0)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_exam_hp' if is_exam else 'fetch_hp', lambda: None)

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.observed_hp is None
    assert state.hp == 0
    assert not state.hp_known


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
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_effect_observations', lambda: [SimpleNamespace(effect='lesson_buff', value=10)])
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))

    state = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert state.observed_effects == frozenset({'lesson_buff'})
    assert state.buffs.lesson_buff == 10


def test_planner_requires_repeated_hud_absence_before_confirming_zero(monkeypatch):
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    ctx = FakeContext([_hand('card')])
    setattr(ctx, 'fetch_effect_observations', lambda: [])
    strategy = PlannerStrategy(FakeEvaluator({'card': 1}))

    first = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))
    second = strategy._build_observed_state(cast(Any, ctx), cast(Any, ctx.hands))

    assert 'lesson_buff' not in first.confirmed_absent_effects
    assert second.confirmed_absent_effects == frozenset({
        'lesson_buff', 'parameter_buff', 'review', 'full_power', 'aggressive',
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


@pytest.mark.parametrize('mode, reason', [
    ('unavailable', 'no_playable_cards'),
    ('unrecognized', 'no_playable_cards'),
    ('unrated', 'no_evaluation'),
    ('unaffordable', 'no_affordable_cards'),
])
def test_diagnostic_fallback_is_recorded_without_commit(monkeypatch, tmp_path, mode, reason):
    import json
    from kaa.tasks.produce.new.play_cards.planner.decision_trace import TRACE_ENV
    monkeypatch.setenv(TRACE_ENV, str(tmp_path))
    monkeypatch.setattr(strategy_module, 'produce_session', lambda: None)
    hand = _planner_hand('card', stamina=100 if mode=='unaffordable' else 0)
    hand.available = mode!='unavailable'
    if mode=='unrecognized':hand.card=None
    ctx = FakeContext([hand])
    strategy = PlannerStrategy(FakeEvaluator({} if mode=='unrated' else {'card':10}))
    assert strategy.on_action(ctx) is False
    assert ctx.committed is None
    events = [json.loads(line) for line in strategy._decision_trace.path.read_text(encoding='utf-8').splitlines()]
    assert len(events)==1 and events[0]['event']=='candidate_screening'
    assert events[0]['outcome']==reason and events[0]['decision_sequence'] is None
    assert strategy._decision_trace.pending is None


@pytest.mark.parametrize('modifier', ['half', 'unknown', None])
def test_hold_ranking_does_not_certify_same_turn_budgets_in_future(modifier):
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    class CaptureEvaluator:
        future = None
        def evaluate(self, card, state):
            self.future = state
            return CardEvaluation(card._id, 1, EvaluationConfidence.LOW, 'future')
    evaluator = CaptureEvaluator()
    strategy = PlannerStrategy(evaluator)
    state = BattleState(remaining_turns=3, hp=2, genki=0,
        observed_remaining_turns=3, observed_hp=2,
        observed_effect_values=(('review', 5),), projected_effect_values=(('review', 7),),
        projected_stamina_consumption=modifier,
        buffs=ExamBuffState(review=7, parameter_buff_turns=2))
    strategy._last_move_state = state
    card = _planner_hand('fee', stamina=3).card
    assert strategy.rank_move_cards([card]) == [0]
    future = evaluator.future
    assert future is not None
    assert future.remaining_turns == 2 and not future.remaining_turns_known
    assert not future.hp_known and not future.genki_known
    assert future.buffs.review == 7 and future.buffs.parameter_buff_turns == 1
    assert dict(future.effect_values_for_planning)['review'] is None
    assert future.projected_stamina_consumption == ('unknown' if modifier else None)
    assert can_afford_play(future, card) is None
    assert state.hp_known and state.buffs.parameter_buff_turns == 2
    assert state.projected_stamina_consumption == modifier
    assert future.observed_remaining_turns == 3 and future.observed_hp == 2


def test_hold_ranking_keeps_growth_uncertain_instead_of_erasing_it(monkeypatch):
    from kaa.db.card_growth import CardGrowthRule
    from kaa.tasks.produce.new.play_cards.planner import costs
    monkeypatch.setattr(costs, 'get_card_growth_rule', lambda key: CardGrowthRule(
        target_known=True, costs_known=True, green_changes=(1,)))
    class CaptureEvaluator:
        future = None
        def evaluate(self, card, state):
            self.future = state
            return CardEvaluation(card._id, 1, EvaluationConfidence.LOW, 'future')
    evaluator = CaptureEvaluator()
    strategy = PlannerStrategy(evaluator)
    strategy._last_move_state = BattleState(remaining_turns=3, hp=100, genki=0,
        anomaly=AnomalyState(full_power_points=7), projected_card_growth=(('growth', True),))
    assert strategy.rank_move_cards([_planner_hand('fee').card]) == [0]
    future = evaluator.future
    assert future is not None and future.projected_card_growth == (('growth', False),)
    assert future.anomaly.full_power_points == 7
    assert can_afford_play(future, _planner_hand('fee').card) is None


@pytest.mark.parametrize('initial_genki,cost,gain,known,phase,missing,gate_id,expected', [
    (10, 5, 0, True, 'ProduceExamPhaseType_None', False, 'gate', 10),
    (3, 0, 5, True, 'ProduceExamPhaseType_None', False, 'gate', 28),
    (10, 0, 0, False, 'ProduceExamPhaseType_None', False, 'gate', 10),
    (10, 0, 0, True, 'ProduceExamPhaseType_None', True, 'gate', 10),
    (10, 0, 0, True, 'ProduceExamPhaseType_ExamStartTurn', False, 'gate', 10),
    (10, 0, 0, True, 'ProduceExamPhaseType_ExamCardPlay', False, 'gate', 28),
    (10, 0, 0, True, 'ProduceExamPhaseType_None', False, '', 28),
])
def test_followup_requires_usage_condition_in_projected_state(
    monkeypatch, initial_genki, cost, gain, known, phase, missing, gate_id, expected,
):
    from kaa.tasks.produce.new.play_cards.planner import trigger
    condition = trigger.ExamTrigger(phase_types=(phase,),
        field_status_types=('ProduceExamFieldStatusType_BlockUp',), field_status_values=(7,))
    monkeypatch.setattr(trigger, 'load_exam_trigger', lambda key: (
        condition if key and not missing else None))
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamBlock, gain), stamina=cost)
    finish = _planner_hand('finish')
    finish.card._play_produce_exam_trigger_id = gate_id
    state = BattleState(remaining_turns=3, hp=20, genki=initial_genki,
        genki_known=known, playable_count=2)
    strategy = PlannerStrategy(FakeEvaluator({'setup': 10, 'finish': 20}))
    candidates = [(hand, strategy._evaluator.evaluate(hand.card, state)) for hand in (setup, finish)]
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert ranked[0][1].score == expected
    assert state.genki == initial_genki


def test_followup_usage_condition_rechecks_special_resource_after_payment(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner import trigger
    from kaa.tasks.produce.new.play_cards.planner.state import ExamBuffState
    condition = trigger.ExamTrigger(field_status_types=('ProduceExamFieldStatusType_ReviewUp',),
        field_status_values=(3,))
    monkeypatch.setattr(trigger, 'load_exam_trigger', lambda key: condition if key else None)
    setup = _planner_hand('setup')
    setup.card.cost_type = 'ExamCostType_ExamReview'
    setup.card.cost_value = 4
    finish = _planner_hand('finish')
    finish.card._play_produce_exam_trigger_id = 'gate'
    state = BattleState(remaining_turns=3, hp=20, genki=0, playable_count=2,
        buffs=ExamBuffState(review=5), observed_effect_values=(('review', 5),))
    strategy = PlannerStrategy(FakeEvaluator({'setup': 10, 'finish': 20}))
    candidates = [(hand, strategy._evaluator.evaluate(hand.card, state)) for hand in (setup, finish)]
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert ranked[0][1].score == 10
    assert state.buffs.review == 5


def _conversion_hand(card_id, ratio):
    effect = _effect(ProduceExamEffectType.ExamLessonDependBlock, 3200)
    effect.produce_exam_effect.effect_value2 = ratio
    hand = _planner_hand(card_id, effect)
    hand.card.plan_type = None
    hand.card.evaluation = 0
    return hand


def test_two_step_conversion_orders_preserving_output_before_full_consumption():
    from kaa.tasks.produce.new.play_cards.planner.evaluator import EffectHeuristicEvaluator
    evaluator = EffectHeuristicEvaluator()
    spend = _conversion_hand('spend', 1000)
    keep = _conversion_hand('keep', 0)
    keep.card.play_effects[0].produce_exam_effect.effect_value1 = 1000
    state = BattleState(remaining_turns=4, hp=20, genki=45,
                        playable_count=2, is_exam=True, exam_score_percent=100)
    strategy = PlannerStrategy(evaluator)
    candidates = [(hand, evaluator.evaluate(hand.card, state)) for hand in (spend, keep)]
    assert candidates[0][1].score > candidates[1][1].score
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert ranked[1][1].score > ranked[0][1].score
    assert '两步预估后续 spend' in ranked[1][1].reason
    assert state.genki == 45 and state.genki_known


@pytest.mark.parametrize('ratio', [500, 1000])
def test_search_does_not_bonus_green_followup_using_consumed_unknown_genki(ratio):
    first = _conversion_hand('conversion', ratio)
    paid = _planner_hand('paid', stamina=30)
    state = BattleState(remaining_turns=4, hp=0, genki=45, playable_count=2)
    strategy = PlannerStrategy(FakeEvaluator({'conversion': 10, 'paid': 100}))
    candidates = [(hand, strategy._evaluator.evaluate(hand.card, state)) for hand in (first, paid)]
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert ranked[0][1].score == 10
    assert '两步预估后续' not in ranked[0][1].reason


@pytest.mark.parametrize('red_cost', [0, 3])
def test_search_keeps_free_or_hp_only_followup_after_consumption(red_cost):
    first = _conversion_hand('conversion', 1000)
    independent = _planner_hand('independent', force_stamina=red_cost)
    state = BattleState(remaining_turns=4, hp=10, genki=45, playable_count=2)
    strategy = PlannerStrategy(FakeEvaluator({'conversion': 10, 'independent': 100}))
    candidates = [(hand, strategy._evaluator.evaluate(hand.card, state)) for hand in (first, independent)]
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert ranked[0][1].score == 100
    assert '两步预估后续 independent' in ranked[0][1].reason


@pytest.mark.parametrize('status,stance', [
    ('ConcentrationUp', AnomalyStance.AGGRESSIVE),
    ('PreservationUp', AnomalyStance.PRESERVATION),
    ('NoStance', AnomalyStance.NORMAL),
])
def test_search_does_not_bonus_free_followup_with_unknown_stance_condition(monkeypatch, status, stance):
    from kaa.tasks.produce.new.play_cards.planner import strategy as strategy_module
    from kaa.tasks.produce.new.play_cards.planner.trigger import ExamTrigger, evaluate_exam_trigger
    setup = _planner_hand('setup')
    finish = _planner_hand('finish')
    finish.card._play_produce_exam_trigger_id = 'stance-gate'
    gate = ExamTrigger(field_status_types=('ProduceExamFieldStatusType_' + status,))
    monkeypatch.setattr(strategy_module, 'evaluate_trigger_id', lambda _id, state: evaluate_exam_trigger(gate, state))
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=2,
                        projected_stance_uncertain=True,
                        anomaly=AnomalyState(stance=stance),
                        confirmed_absent_effects=frozenset({'concentration', 'preservation'}))
    strategy = PlannerStrategy(FakeEvaluator({'setup': 10, 'finish': 100}))
    candidates = [(hand, strategy._evaluator.evaluate(hand.card, state)) for hand in (setup, finish)]
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert ranked[0][1].score == 10
    assert '两步预估后续' not in ranked[0][1].reason


def test_search_does_not_bonus_followup_from_unknown_preservation_exit():
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamConcentration, 1))
    finish = _planner_hand('finish')
    state = BattleState(remaining_turns=4, hp=20, genki=20, playable_count=1,
                        projected_stance_uncertain=True,
                        anomaly=AnomalyState(stance=AnomalyStance.PRESERVATION))
    strategy = PlannerStrategy(FakeEvaluator({'setup': 10, 'finish': 100}))
    candidates = [(hand, strategy._evaluator.evaluate(hand.card, state)) for hand in (setup, finish)]
    ranked = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert ranked[0][1].score == 10
    assert '两步预估后续' not in ranked[0][1].reason


class MotivationSequenceEvaluator:
    def evaluate(self, card, state):
        output = float(state.genki) if card._id == 'convert' else 0.0
        return CardEvaluation(
            card._id, output if output else (1.0 if card._id == 'prepare' else 0.0),
            EvaluationConfidence.MEDIUM, card._id,
            projected_output=output,
            confirmed_output=output if state.genki_known else 0.0,
        )


def _motivation_sequence():
    extra = _planner_hand('extra', _effect(ProduceExamEffectType.ExamPlayableValueAdd, 0))
    prepare = _planner_hand('prepare', _effect(ProduceExamEffectType.ExamBlockFix, 20))
    convert = _planner_hand('convert', _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000))
    state = BattleState(remaining_turns=5, hp=20, genki=2, playable_count=2,
                        archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    strategy = PlannerStrategy(MotivationSequenceEvaluator())
    hands = [extra, prepare, convert]
    candidates = [(hand, strategy._evaluator.evaluate(hand.card, state)) for hand in hands]
    return strategy, state, candidates


def test_motivation_search_values_extra_prepare_convert_sequence():
    strategy, state, candidates = _motivation_sequence()
    two = strategy._add_shallow_lookahead(cast(Any, candidates), state, depth=2)
    three = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert two[0][1].score == pytest.approx(1.8)
    assert three[0][1].score == pytest.approx(0.9 * (1 + 0.9 * 22))
    assert 'prepare' in three[0][1].reason and 'convert' in three[0][1].reason
    # This is ranking value only; future conversion is not current output.
    assert three[0][1].projected_output == three[0][1].confirmed_output == 0
    assert state.genki == 2 and state.playable_count == 2


@pytest.mark.parametrize('boundary', ['one_play', 'other_archetype', 'no_converter', 'two_instances'])
def test_motivation_search_does_not_invent_third_play(boundary):
    strategy, state, candidates = _motivation_sequence()
    if boundary == 'one_play': state = replace(state, playable_count=1)
    elif boundary == 'other_archetype': state = replace(state, archetype=ProduceExamEffectType.ExamReview)
    elif boundary == 'no_converter': candidates = candidates[:2]
    else: candidates = [candidates[0], candidates[2]]
    two = strategy._add_shallow_lookahead(cast(Any, candidates), state, depth=2)
    current = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert [e.score for _, e in current] == [e.score for _, e in two]


def test_third_play_can_become_payable_after_preparation():
    strategy, state, candidates = _motivation_sequence()
    state = replace(state, hp=0, genki=2)
    candidates[2][0].card.stamina = 10
    assert can_afford_play(state, candidates[2][0].card) is False
    three = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert three[0][1].score == pytest.approx(0.9 * (1 + 0.9 * 22))


@pytest.mark.parametrize('gate', ['unknown_payment', 'unknown_trigger', 'inactive_trigger'])
def test_third_play_requires_verified_payment_and_usage(monkeypatch, gate):
    strategy, state, candidates = _motivation_sequence()
    if gate == 'unknown_payment':
        state = replace(state, hp=0, hp_known=False)
        candidates[2][0].card.force_stamina = 1
    else:
        from kaa.tasks.produce.new.play_cards.planner import trigger
        candidates[2][0].card._play_produce_exam_trigger_id = 'gate'
        condition = trigger.ExamTrigger(field_status_types=('ProduceExamFieldStatusType_BlockUp',),
                                        field_status_values=(100,))
        monkeypatch.setattr(trigger, 'load_exam_trigger', lambda key: (
            condition if key and gate == 'inactive_trigger' else None))
    three = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert three[0][1].score == pytest.approx(0.9)



def test_three_card_search_uses_production_motivation_resource_model():
    from kaa.tasks.produce.new.play_cards.planner.evaluator import CompositeEvaluator
    strategy, state, candidates = _motivation_sequence()
    strategy = PlannerStrategy(CompositeEvaluator())
    for hand, _ in candidates:
        hand.card.plan_type = ProducePlanType.Common.value
    # Start with one play, and an explicit two-play grant. No future draw is
    # needed: all three distinct cards are in the observed hand.
    candidates[0][0].card.play_effects[0].produce_exam_effect.effect_value1 = 2
    state = replace(state, playable_count=1)
    candidates = [(hand, strategy._evaluator.evaluate(hand.card, state)) for hand, _ in candidates]
    two = strategy._add_shallow_lookahead(cast(Any, candidates), state, depth=2)
    three = strategy._add_shallow_lookahead(cast(Any, candidates), state)
    assert three[0][1].score > two[0][1].score
    after_extra = simulate_play(state, candidates[0][0].card)
    after_prepare = simulate_play(after_extra, candidates[1][0].card)
    assert after_extra.playable_count == 2 and after_prepare.playable_count == 1
    conversion = strategy._evaluator.evaluate(candidates[2][0].card, after_prepare)
    assert conversion.confirmed_output == 22
    assert three[0][1].confirmed_output == 0


@pytest.mark.parametrize('verdict,plain,expected', [
    ('ACTIVE', True, 23), ('UNKNOWN', True, 7),
    ('INACTIVE', True, 7), ('ACTIVE', False, 7),
])
def test_exam_finish_preserves_only_confirmed_extra_turn_window(monkeypatch, verdict, plain, expected):
    from kaa.tasks.produce.new.play_cards.planner import simulator
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    monkeypatch.setattr(simulator, 'evaluate_trigger_id', lambda *_: TriggerVerdict[verdict])
    effect = _effect(ProduceExamEffectType.ExamExtraTurn, 0)
    effect.produce_exam_effect.effect_count = 0
    effect.produce_exam_effect._produce_card_grow_effect_ids = '[]' if plain else None
    extension = _planner_hand('extension', effect)
    output = _planner_hand('output', _effect(ProduceExamEffectType.ExamLesson, 9))
    state = BattleState(remaining_turns=2, hp=16, genki=21, is_exam=True,
                        observed_remaining_turns=2)
    candidates = [
        (extension, CardEvaluation('extension', 23, EvaluationConfidence.MEDIUM, 'extension')),
        (output, CardEvaluation('output', 8, EvaluationConfidence.MEDIUM, 'output',
                               projected_output=9, confirmed_output=9)),
    ]
    ranked = PlannerStrategy(FakeEvaluator({}), is_exam=True)._protect_exam_finish(
        cast(Any, candidates), state)
    assert ranked[0][1].score == expected
    assert ranked[1][1].score == 8
    assert ranked[0][1].confirmed_output == 0
    assert state.observed_remaining_turns == 2


@pytest.mark.parametrize('remaining,known,archetype,expected', [
    (8, True, ProduceExamEffectType.ExamCardPlayAggressive, 'setup'),
    (4, True, ProduceExamEffectType.ExamCardPlayAggressive, 'setup'),
    (3, True, ProduceExamEffectType.ExamCardPlayAggressive, 'output'),
    (1, True, ProduceExamEffectType.ExamCardPlayAggressive, 'output'),
    (8, False, ProduceExamEffectType.ExamCardPlayAggressive, 'output'),
    (8, True, ProduceExamEffectType.ExamReview, 'output'),
])
def test_motivation_setup_phase_keeps_deadline_and_unknown_window(remaining, known, archetype, expected):
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamBlock, 8))
    output = _planner_hand('output', _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000))
    rows = [(setup, CardEvaluation('setup', 20, EvaluationConfidence.MEDIUM, 'setup')),
            (output, CardEvaluation('output', 100, EvaluationConfidence.MEDIUM, 'output',
                                    projected_output=100, confirmed_output=0))]
    state = BattleState(remaining_turns=remaining, remaining_turns_known=known,
                        archetype=archetype, hp=20, genki=30)
    ranked = PlannerStrategy()._prefer_motivation_setup(cast(Any, rows), state)
    assert max(ranked, key=lambda pair: pair[1].score)[1].card_id == expected
    assert ranked[1][1].projected_output == 100
    assert ranked[1][1].confirmed_output == 0
    assert len(ranked) == 2


@pytest.mark.parametrize('boundary', ['no_setup', 'inactive', 'unaffordable', 'negative_setup', 'clear'])
def test_motivation_setup_phase_keeps_safety_and_no_preparation_fallback(monkeypatch, boundary):
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamCardPlayAggressive, 4))
    output = _planner_hand('output', _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000))
    rows = [(setup, CardEvaluation('setup', 20, EvaluationConfidence.MEDIUM, 'setup')),
            (output, CardEvaluation('output', 100, EvaluationConfidence.MEDIUM, 'output',
                                    projected_output=100, confirmed_output=100))]
    state = BattleState(remaining_turns=8, archetype=ProduceExamEffectType.ExamCardPlayAggressive,
                        hp=20, genki=30)
    if boundary == 'no_setup': rows = rows[1:]
    elif boundary == 'inactive':
        from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
        monkeypatch.setattr(strategy_module, 'evaluate_trigger_id', lambda *_: TriggerVerdict.UNKNOWN)
    elif boundary == 'unaffordable': setup.card.stamina = 100
    elif boundary == 'negative_setup': rows[0] = (setup, replace(rows[0][1], score=-1))
    else: state = replace(state, score_gap=100, score_target_is_final=False)
    assert PlannerStrategy()._prefer_motivation_setup(cast(Any, rows), state) == rows


@pytest.mark.parametrize('boundary,expected', [('mixed', 'mixed'), ('consume', 'setup'),
    ('unknown_gain', 'setup'), ('zero_gain', 'setup'), ('unaffordable', 'setup')])
def test_motivation_phase_retains_native_gain_on_mixed_output(monkeypatch, boundary, expected):
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    gain = _effect(ProduceExamEffectType.ExamCardPlayAggressive, 4)
    output = _effect(ProduceExamEffectType.ExamLessonDependBlock, 800)
    mixed = _planner_hand('mixed', gain, output)
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamBlock, 8))
    if boundary == 'consume': output.produce_exam_effect.effect_value2 = 1000
    elif boundary == 'zero_gain': gain.produce_exam_effect.effect_value1 = 0
    elif boundary == 'unaffordable': mixed.card.stamina = 100
    elif boundary == 'unknown_gain':
        gain._produce_exam_trigger_id = 'gain'
        monkeypatch.setattr(strategy_module, 'evaluate_trigger_id',
            lambda trigger, _: TriggerVerdict.UNKNOWN if trigger == 'gain' else TriggerVerdict.ACTIVE)
    rows = [(setup, CardEvaluation('setup', 20, EvaluationConfidence.MEDIUM, 'setup')),
            (mixed, CardEvaluation('mixed', 200, EvaluationConfidence.MEDIUM, 'mixed',
                                   projected_output=100, confirmed_output=0))]
    state = BattleState(remaining_turns=8, archetype=ProduceExamEffectType.ExamCardPlayAggressive,
                        hp=20, genki=30)
    ranked = PlannerStrategy()._prefer_motivation_setup(cast(Any, rows), state)
    assert max(ranked, key=lambda pair: pair[1].score)[1].card_id == expected
    assert ranked[1][1].projected_output == 100
    assert ranked[1][1].confirmed_output == 0


@pytest.mark.parametrize('boundary,preserve', [('deadline', True), ('early', False),
    ('unknown_target', False), ('unconfirmed_output', False), ('exam', False)])
def test_motivation_phase_preserves_needed_visible_conversion_near_lesson_deadline(boundary, preserve):
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamBlock, 8))
    output = _planner_hand('output', _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000))
    rows = [(setup, CardEvaluation('setup', 20, EvaluationConfidence.MEDIUM, 'setup')),
            (output, CardEvaluation('output', 100, EvaluationConfidence.MEDIUM, 'output',
                                    projected_output=48.4, confirmed_output=48.4))]
    state = BattleState(remaining_turns=4, archetype=ProduceExamEffectType.ExamCardPlayAggressive,
                        hp=20, genki=44, score_gap=73, score_target_is_final=False)
    if boundary == 'early': state = replace(state, remaining_turns=5)
    elif boundary == 'unknown_target': state = replace(state, score_target_is_final=None)
    elif boundary == 'exam': state = replace(state, is_exam=True)
    elif boundary == 'unconfirmed_output': rows[1] = (output, replace(rows[1][1], confirmed_output=0))
    ranked = PlannerStrategy()._prefer_motivation_setup(cast(Any, rows), state)
    assert (ranked == rows) is preserve
    assert ranked[1][1].projected_output == 48.4


def _late_motivation_candidates():
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamBlock, 11),
                          _effect(ProduceExamEffectType.ExamCardPlayAggressive, 7),
                          _effect(ProduceExamEffectType.ExamStaminaConsumptionAdd, 0))
    for segment in setup.card.play_effects:
        segment.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    output = _planner_hand('output', _effect(ProduceExamEffectType.ExamLesson, 9), stamina=4)
    candidates = [(setup, CardEvaluation('setup', 20, EvaluationConfidence.MEDIUM, 'setup')),
                  (output, CardEvaluation('output', 10, EvaluationConfidence.HIGH, 'output', 9, 9))]
    state = BattleState(remaining_turns=2, hp=21, genki=43, hand_size=2,
                        score_gap=39, score_target_is_final=False,
                        archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    return candidates, state


@pytest.mark.parametrize('turns', [1, 2])
def test_motivation_late_clear_keeps_visible_output_over_surplus_setup(turns):
    candidates, state = _late_motivation_candidates()
    ranked = PlannerStrategy()._prefer_lesson_finish_progress(cast(Any, candidates),
                                                            replace(state, remaining_turns=turns))
    assert max(ranked, key=lambda x: x[1].score)[0] is candidates[1][0]
    assert '无可见转换' in ranked[0][1].reason
    assert [(h, ev.projected_output, ev.confirmed_output) for h, ev in ranked] == [
        (h, ev.projected_output, ev.confirmed_output) for h, ev in candidates]
    assert candidates[0][1].score == 20


@pytest.mark.parametrize('changes', [
    {'is_exam': True}, {'score_target_is_final': True}, {'score_target_is_final': None},
    {'score_gap': None}, {'score_gap': 0}, {'remaining_turns': 3},
    {'remaining_turns_known': False}, {'hp_known': False}, {'hp': 8},
    {'genki_known': False}, {'genki': 10}, {'playable_count': 2},
    {'hand_size': None}, {'hand_size': 3}, {'archetype': None},
    {'projected_effect_values': ()}, {'unresolved_growth_card_ids': frozenset({'setup'})},
])
def test_late_motivation_protection_requires_complete_safe_current_input(changes):
    candidates, state = _late_motivation_candidates()
    ranked = PlannerStrategy()._protect_motivation_lesson_finish(cast(Any, candidates),
                                                               replace(state, **changes))
    assert ranked == candidates


@pytest.mark.parametrize('kind', [ProduceExamEffectType.ExamLessonDependBlock,
                                 ProduceExamEffectType.ExamCardDraw,
                                 ProduceExamEffectType.ExamPlayableValueAdd,
                                 ProduceExamEffectType.ExamExtraTurn,
                                 ProduceExamEffectType.ExamReview])
def test_late_motivation_protection_keeps_conversion_and_other_opportunities(kind):
    candidates, state = _late_motivation_candidates()
    candidates[0][0].card.play_effects.append(_effect(kind, 1))
    assert PlannerStrategy()._protect_motivation_lesson_finish(cast(Any, candidates), state) == candidates


@pytest.mark.parametrize('affordable', [None, False])
def test_late_motivation_protection_requires_payable_independent_output(monkeypatch, affordable):
    candidates, state = _late_motivation_candidates()
    monkeypatch.setattr(strategy_module, 'can_afford_play',
                        lambda _state, card: affordable if card._id == 'output' else True)
    assert PlannerStrategy()._protect_motivation_lesson_finish(cast(Any, candidates), state) == candidates


@pytest.mark.parametrize('boundary', ['unknown_trigger', 'inactive_trigger', 'unknown_growth',
                                      'chain', 'mixed_output', 'uncertain_output'])
def test_late_motivation_protection_preserves_unresolved_or_mixed_candidates(monkeypatch, boundary):
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    candidates, state = _late_motivation_candidates()
    if boundary.endswith('trigger'):
        verdict = TriggerVerdict.UNKNOWN if boundary == 'unknown_trigger' else TriggerVerdict.INACTIVE
        monkeypatch.setattr(strategy_module, 'evaluate_trigger_id', lambda *_: verdict)
    elif boundary == 'unknown_growth':
        candidates[0][0].card.play_effects[0].produce_exam_effect._produce_card_grow_effect_ids = None
    elif boundary == 'chain':
        candidates[0][0].card.play_effects[0].produce_exam_effect._chain_produce_exam_effect_id = 'chain'
    elif boundary == 'mixed_output':
        candidates[0] = (candidates[0][0], replace(candidates[0][1], projected_output=1))
    else:
        candidates[1] = (candidates[1][0], replace(candidates[1][1], confirmed_output=0))
    assert PlannerStrategy()._protect_motivation_lesson_finish(cast(Any, candidates), state) == candidates


@pytest.mark.parametrize('payment', [True, None])
@pytest.mark.parametrize('count', [None, 1])
def test_exam_four_turn_visible_conversion_keeps_ranking_not_payment_proof(monkeypatch, payment, count):
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamBlock, 8))
    effect = _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000)
    effect.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    output = _planner_hand('output', effect, force_stamina=5)
    output.wide_output_preview = SimpleNamespace(per_hit=749, count=count)
    rows = [(setup, CardEvaluation('setup', 4.4, EvaluationConfidence.MEDIUM, 'setup')),
            (output, CardEvaluation('output', 100, EvaluationConfidence.MEDIUM, 'preview', 749, 0))]
    state = BattleState(remaining_turns=4, hp=28, genki=88, is_exam=True,
                        archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    monkeypatch.setattr(strategy_module, 'can_afford_play', lambda *_: payment)
    ranked = PlannerStrategy()._prefer_motivation_setup(cast(Any, rows), state)
    assert ranked is rows
    assert ranked[1][1].confirmed_output == 0
    assert output.wide_output_preview.count is count
    assert state.observed_card_output is None


@pytest.mark.parametrize('boundary', ['early', 'unknown_turns', 'lesson', 'extra_play',
    'low_hp', 'unknown_hp', 'unknown_genki', 'zero_genki', 'projection', 'shared_preview',
    'no_preview', 'zero_preview', 'repeat_preview', 'mismatch', 'growth', 'unknown_growth',
    'partial_consumption', 'chain', 'unaffordable', 'unknown_use', 'inactive_use'])
def test_exam_conversion_preview_exception_stays_narrow(monkeypatch, boundary):
    from kaa.tasks.produce.new.play_cards.planner.trigger import TriggerVerdict
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamBlock, 8))
    effect = _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000)
    effect.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    output = _planner_hand('output', effect)
    output.wide_output_preview = SimpleNamespace(per_hit=749, count=None)
    rows = [(setup, CardEvaluation('setup', 4.4, EvaluationConfidence.MEDIUM, 'setup')),
            (output, CardEvaluation('output', 100, EvaluationConfidence.MEDIUM, 'preview', 749, 0))]
    state = BattleState(remaining_turns=4, hp=28, genki=88, is_exam=True,
                        archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    if boundary == 'early': state = replace(state, remaining_turns=5)
    elif boundary == 'unknown_turns': state = replace(state, remaining_turns_known=False)
    elif boundary == 'lesson': state = replace(state, is_exam=False)
    elif boundary == 'extra_play': state = replace(state, playable_count=2)
    elif boundary == 'low_hp': state = replace(state, hp=8)
    elif boundary == 'unknown_hp': state = replace(state, hp_known=False)
    elif boundary == 'unknown_genki': state = replace(state, genki_known=False)
    elif boundary == 'zero_genki': state = replace(state, genki=0)
    elif boundary == 'projection': state = replace(state, projected_effect_values=())
    elif boundary == 'shared_preview': state = replace(state, observed_card_output=('output', 749, None))
    elif boundary == 'no_preview': output.wide_output_preview = None
    elif boundary == 'zero_preview': output.wide_output_preview.per_hit = 0
    elif boundary == 'repeat_preview': output.wide_output_preview.count = 2
    elif boundary == 'mismatch': rows[1] = (output, replace(rows[1][1], projected_output=748))
    elif boundary == 'growth': output.has_growth_marker = True
    elif boundary == 'unknown_growth': effect.produce_exam_effect._produce_card_grow_effect_ids = None
    elif boundary == 'partial_consumption': effect.produce_exam_effect.effect_value2 = 500
    elif boundary == 'chain': effect.produce_exam_effect._chain_produce_exam_effect_id = 'chain'
    elif boundary == 'unaffordable':
        monkeypatch.setattr(strategy_module, 'can_afford_play',
                            lambda _, card: False if card._id == 'output' else True)
    elif boundary in {'unknown_use', 'inactive_use'}:
        output.card._play_produce_exam_trigger_id = 'use'
        monkeypatch.setattr(strategy_module, 'evaluate_trigger_id',
            lambda trigger, _: (TriggerVerdict.UNKNOWN if boundary == 'unknown_use' else TriggerVerdict.INACTIVE)
            if trigger == 'use' else TriggerVerdict.ACTIVE)
    ranked = PlannerStrategy()._prefer_motivation_setup(cast(Any, rows), state)
    assert max(ranked, key=lambda pair: pair[1].score)[1].card_id == ('output' if boundary == 'unknown_turns' else 'setup')
    assert ranked[1][1].confirmed_output == 0



def test_exam_preview_exception_does_not_transfer_between_same_id_copies():
    effect = _effect(ProduceExamEffectType.ExamLessonDependBlock, 1000)
    effect.produce_exam_effect._produce_card_grow_effect_ids = '[]'
    with_preview = _planner_hand('duplicate', effect)
    without_preview = _planner_hand('duplicate', effect)
    with_preview.wide_output_preview = SimpleNamespace(per_hit=749, count=None)
    without_preview.wide_output_preview = None
    setup = _planner_hand('setup', _effect(ProduceExamEffectType.ExamBlock, 8))
    rows = [(setup, CardEvaluation('setup', 4.4, EvaluationConfidence.MEDIUM, 'setup')),
            (with_preview, CardEvaluation('duplicate', 0, EvaluationConfidence.LOW, 'other copy')),
            (without_preview, CardEvaluation('duplicate', 100, EvaluationConfidence.LOW, 'no preview', 749, 0))]
    state = BattleState(remaining_turns=4, hp=28, genki=88, is_exam=True,
                        archetype=ProduceExamEffectType.ExamCardPlayAggressive)
    ranked = PlannerStrategy()._prefer_motivation_setup(cast(Any, rows), state)
    assert max(ranked, key=lambda pair: pair[1].score)[0] is setup
    assert ranked[2][1].projected_output == 749
    assert ranked[2][1].confirmed_output == 0
