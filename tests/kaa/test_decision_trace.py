import json
from types import SimpleNamespace

from kaa.tasks.produce.new.play_cards.planner.decision_trace import DecisionTrace, TRACE_ENV
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, EvaluationConfidence
from kaa.tasks.produce.new.play_cards.planner.feedback import PlayFeedback, PlayFeedbackStatus
from kaa.tasks.produce.new.play_cards.planner.state import AnomalyState, BattleState


def _record(trace):
    state = BattleState(
        remaining_turns=2, hp=7, genki=8,
        observed_effects=frozenset({'full_power'}),
        anomaly=AnomalyState(full_power_points=10, full_power_pending=True),
    )
    evaluation = CardEvaluation('card', 6.3, EvaluationConfidence.LOW, '测试')
    card = SimpleNamespace(upgrade_count=1, stamina=3, force_stamina=0, play_effects=[])
    trace.decision(state, [(SimpleNamespace(card=card), evaluation)], evaluation)


def test_trace_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv(TRACE_ENV, raising=False)
    assert DecisionTrace.from_environment() is None
    monkeypatch.setenv(TRACE_ENV, str(tmp_path))
    assert DecisionTrace.from_environment() is not None
    assert list(tmp_path.iterdir()) == []


def test_trace_pairs_serializable_state_and_feedback(tmp_path):
    trace = DecisionTrace(tmp_path)
    _record(trace)
    trace.feedback(PlayFeedback(PlayFeedbackStatus.CONFIRMED, 'card_left_hand', 39))
    events = [json.loads(line) for line in trace.path.read_text(encoding='utf-8').splitlines()]
    assert [event['event'] for event in events] == ['decision', 'feedback']
    assert [event['sequence'] for event in events] == [1, 1]
    assert events[0]['state']['observed_effects'] == ['full_power']
    assert events[0]['state']['anomaly']['full_power_pending'] is True
    assert events[0]['candidates'][0]['upgrade_count'] == 1
    assert events[0]['selected']['reason'] == '测试'
    assert events[1]['feedback']['observed_gap_reduction'] == 39
    trace.feedback(PlayFeedback(PlayFeedbackStatus.UNCONFIRMED, 'duplicate'))
    assert len(trace.path.read_text(encoding='utf-8').splitlines()) == 2


def test_trace_is_bounded_per_battle(tmp_path):
    trace = DecisionTrace(tmp_path)
    trace.count = 255
    _record(trace)
    trace.feedback(PlayFeedback(PlayFeedbackStatus.UNCONFIRMED, 'insufficient_evidence'))
    _record(trace)
    trace.feedback(PlayFeedback(PlayFeedbackStatus.UNCONFIRMED, 'over_limit'))
    assert len(trace.path.read_text(encoding='utf-8').splitlines()) == 2
    assert trace.count == 256


def test_trace_write_failure_does_not_break_decision_or_leave_pending(tmp_path):
    occupied = tmp_path / 'not-a-directory'
    occupied.touch()
    trace = DecisionTrace(occupied)
    _record(trace)
    assert trace.pending is None
    trace.feedback(PlayFeedback(PlayFeedbackStatus.UNCONFIRMED, 'insufficient_evidence'))


def test_trace_keeps_static_growth_chain_and_payment_boundary(tmp_path):
    trace = DecisionTrace(tmp_path)
    state = BattleState(remaining_turns=2, hp=7, genki=8)
    evaluation = CardEvaluation('card', 6.3, EvaluationConfidence.LOW, 'test')
    effect = SimpleNamespace(
        _id='timer', effect_type=None, effect_value1=1, effect_value2=0,
        effect_count=1, effect_turn=1,
        _chain_produce_exam_effect_id='grow-hand',
        _produce_card_grow_effect_ids='["grow-plus-8"]',
        _produce_card_search_id='hand-all',
        _produce_exam_status_enchant_id='enchant',
    )
    card = SimpleNamespace(upgrade_count=1, stamina=3, force_stamina=0,
                           cost_type='ExamCostType_ExamFullPowerPoint', cost_value=3,
                           customize_ids=['custom'],
                           play_effects=[SimpleNamespace(produce_exam_effect=effect)])
    trace.decision(state, [(SimpleNamespace(card=card), evaluation)], evaluation)
    candidate = json.loads(trace.path.read_text(encoding='utf-8'))['candidates'][0]
    assert candidate['modeled_payment']['green_red'] == [3, 0]
    assert candidate['modeled_payment']['special_affordable'] is None
    assert candidate['cost_value'] == 3
    assert candidate['customize_ids'] == ['custom']
    assert candidate['effects'][0]['chain_effect_id'] == 'grow-hand'
    assert candidate['effects'][0]['grow_effect_ids'] == '["grow-plus-8"]'
    assert candidate['effects'][0]['turn'] == 1


def test_battle_end_records_pending_request_as_unobserved_and_is_idempotent(tmp_path, monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay
    from kaa.tasks.produce.new.play_cards.planner.replay import screen_trace
    monkeypatch.delenv(TRACE_ENV, raising=False)
    strategy = PlannerStrategy()
    trace = DecisionTrace(tmp_path)
    _record(trace)
    strategy._decision_trace = trace
    strategy._pending_play = PendingPlay('card', BattleState(remaining_turns=1, hp=1, genki=0), 10, 10)
    evidence = SimpleNamespace(feedback=lambda reason, frame: calls.append((reason, frame)))
    calls = []
    strategy._play_evidence = evidence
    strategy.on_battle_end()
    strategy.on_battle_end()
    events = [json.loads(line) for line in trace.path.read_text(encoding='utf-8').splitlines()]
    assert len(events) == 2
    assert events[1]['feedback']['status'] == 'unconfirmed'
    assert events[1]['feedback']['evidence'] == 'battle_ended_without_observation'
    assert events[1]['feedback']['observed_gap_reduction'] is None
    assert events[1]['after_state'] is None
    assert calls == [('battle_ended_without_observation', None)]
    assert strategy._pending_play is None
    assert strategy._memory_before_pending_play is None
    row = screen_trace(trace.path)[0]
    assert row['bucket'] == 'excluded'
    assert row['after_source'] == 'battle_end_unobserved'
    assert 'missing_feedback' not in row['exclusion_reasons']
    assert 'missing_after_state' in row['exclusion_reasons']


def test_battle_end_without_pending_play_does_not_invent_feedback(tmp_path, monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
    monkeypatch.delenv(TRACE_ENV, raising=False)
    strategy = PlannerStrategy()
    trace = DecisionTrace(tmp_path)
    strategy._decision_trace = trace
    strategy.on_battle_end()
    assert not trace.path.exists()
