from dataclasses import replace
from types import SimpleNamespace

import pytest

from kaa.db.skill_card import SkillCard
from kaa.idol_road.category_priority import CategoryPriority
from kaa.idol_road.category_rule import CategoryObjective, category_objectives, read_category_condition
from kaa.idol_road.rules import RuleTextRead, RuleTextLine, StageRules, RuleViewport, VisibleStageRule
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, EvaluationConfidence
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay, PlayFeedback, PlayFeedbackStatus
from kaa.tasks.produce.new.play_cards.planner.state import BattleState, ExamBuffState
from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy

MENTAL = 'ProduceCardCategory_MentalSkill'
ACTIVE = 'ProduceCardCategory_ActiveSkill'
TEXT = '直前にメンタルスキルカードを使用していた場合、低下状態無効1回'


def reads(text=TEXT, frame='frame', confidence=.99):
    return tuple(RuleTextRead(frame, variant, (RuleTextLine(text, confidence),))
                 for variant in ('original', 'grayscale'))


def rule(turn=3, text=TEXT, frame='frame', confidence=.99):
    return VisibleStageRule((1, 1, 10, 10), turn, reads(f'{turn}ターン目', frame), reads(text, frame, confidence))


def objectives(*rows):
    return StageRules('ttmr', 18, (RuleViewport('viewport', tuple(rows)),))


class Scores:
    def __init__(self, scores):
        self.scores = scores

    def evaluate(self, card, state):
        return CardEvaluation(card._id, self.scores.get(card._id, 1), EvaluationConfidence.HIGH, 'test')


def cards():
    return tuple(SkillCard.from_id(key) for key in
                 ('p_card-01-ido-3_014', 'p_card-01-men-2_035', 'p_card-01-act-2_001'))


def state(**changes):
    return replace(BattleState(remaining_turns=17, hp=50, genki=0, is_exam=True,
                              buffs=ExamBuffState(lesson_buff=10),
                              observed_effect_values=(('lesson_buff', 10),)), **changes)


def policy(evaluator=None):
    return CategoryPriority(18, (CategoryObjective(3, MENTAL, 1, ('frame',)),), evaluator=evaluator)


def test_complete_two_representation_rule_has_previous_turn_deadline():
    goal = read_category_condition(3, reads())
    assert goal is not None and goal.deadline_turn == 2 and goal.category == MENTAL
    assert goal.reward_count == 1 and 'user_confirmed' in goal.timing_source
    split = tuple(replace(row, lines=(RuleTextLine(TEXT[:15], .99), RuleTextLine(TEXT[15:], .99))) for row in reads())
    assert read_category_condition(3, split) == goal
    assert read_category_condition(6, reads(TEXT.replace('メンタル', 'アクティブ').replace('1回', '2回'))).category == ACTIVE


@pytest.mark.parametrize('bad', [reads(confidence=.979), reads()[:1],
    (reads()[0], replace(reads()[1], frame_id='old')), reads(TEXT.replace('場合、', '場合以外、')),
    reads(TEXT+'さらに抽出'), reads(TEXT.replace('1回', '3回')),
    (reads()[0], reads(TEXT.replace('メンタル', 'アクティブ'))[1])])
def test_unknown_conflicting_or_unsupported_rule_is_not_a_goal(bad):
    assert read_category_condition(3, bad) is None


def test_stage_binding_dedup_and_conflicts_never_sum_rewards():
    valid = objectives(rule(), rule(frame='second'))
    goals = category_objectives('ttmr', 18, valid)
    assert len(goals) == 1 and goals[0].reward_count == 1
    assert goals[0].evidence_frames == ('frame', 'second')
    assert not category_objectives('ttmr', 19, valid)
    assert not category_objectives('ttmr', 18, replace(valid, identity_source='history'))
    assert not category_objectives('ttmr', 18, objectives(rule(), rule(text=TEXT.replace('メンタル', 'アクティブ'))))
    assert category_objectives('ttmr', 18, objectives(rule(), rule(confidence=.97))) == category_objectives('ttmr', 18, objectives(rule()))
    assert not category_objectives('ttmr', 18, objectives(replace(rule(), description_reads=reads(frame='other'))))


def test_lower_score_matching_category_precedes_high_immediate_output():
    active, mental, _ = cards()
    result = policy(Scores({active._id: 1000, mental._id: 1})).select(state(), (active, mental))
    assert result.preferred_indices == (1,) and 'reward_unconfirmed' in result.reason
    assert not result.end_turn


def test_extra_active_then_mental_is_compared_as_a_complete_visible_route():
    _, mental, extra = cards()
    result = policy(Scores({mental._id: 1, extra._id: 100})).select(state(), (extra, mental))
    assert result.preferred_indices == (0,)
    assert str((extra._id, mental._id)) in result.reason


def test_unknown_future_cost_or_missing_goal_card_cannot_certify_route():
    active, mental, extra = cards()
    unknown = state(hp_known=False, genki_known=False, observed_effect_values=())
    result = policy(Scores({})).select(unknown, (extra, mental))
    assert not result.preferred_indices and 'no_supported_visible_route' in result.reason
    assert not policy(Scores({})).select(state(), (active,)).preferred_indices
    assert not policy(Scores({})).select(state(remaining_turns=16), (active, mental)).preferred_indices
    assert not policy(Scores({})).select(state(remaining_turns_known=False), (active, mental)).preferred_indices
    after_battle = CategoryPriority(18, (CategoryObjective(19, MENTAL, 1, ('frame',)),), evaluator=Scores({}))
    assert not after_battle.select(state(remaining_turns=1), (active, mental)).preferred_indices


@pytest.mark.parametrize('evidence,confirmed', [('card_left_hand', True), ('resource_changed', False),
    ('hud_value_changed', False), ('insufficient_evidence', False)])
def test_only_complete_hand_feedback_can_authorize_preserving_last_category(evidence, confirmed):
    active, mental, _ = cards()
    goal = policy(Scores({}))
    before = state()
    goal.committed(before, mental)
    assert not goal.select(before, (active,)).end_turn
    pending = PendingPlay(mental._id, before, 0, 0)
    goal.feedback(pending, PlayFeedback(PlayFeedbackStatus.CONFIRMED, evidence), before)
    assert goal.select(before, (active,)).end_turn is confirmed
    assert not goal.select(state(remaining_turns=16), (active,)).end_turn
    goal.committed(before, active)
    assert not goal.select(before, (active,)).end_turn


def test_unconfirmed_and_turn_advanced_feedback_never_authorize_skip():
    active, mental, _ = cards()
    for status, after in ((PlayFeedbackStatus.UNCONFIRMED, state()),
                          (PlayFeedbackStatus.CONFIRMED, state(remaining_turns=16))):
        goal = policy(Scores({}))
        goal.committed(state(), mental)
        goal.feedback(PendingPlay(mental._id, state(), 0, 0), PlayFeedback(status, 'card_left_hand'), after)
        assert not goal.select(state(), (active,)).end_turn


def test_matching_last_does_not_waste_another_useful_matching_play():
    active, mental, _ = cards()
    goal = policy(Scores({mental._id: 5, active._id: 100}))
    goal.committed(state(), mental)
    goal.feedback(PendingPlay(mental._id, state(), 0, 0),
                  PlayFeedback(PlayFeedbackStatus.CONFIRMED, 'card_left_hand'), state())
    result = goal.select(state(), (active, mental))
    assert result.preferred_indices == (1,) and not result.end_turn


def test_shared_strategy_executes_optional_end_turn_and_stops_when_unavailable(monkeypatch):
    from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
    active, _, _ = cards()
    hand = SimpleNamespace(card=active, available=True)
    context = SimpleNamespace(fetch_hands=lambda: [hand], commit=lambda hand: pytest.fail('Must not play'))
    goal = SimpleNamespace(select=lambda *args: SelectionConstraint(end_turn=True, reason='test'))
    called = []
    for enabled in (True, False):
        strategy = PlannerStrategy(evaluator=Scores({}), is_exam=True, selection_policy=goal,
                                   end_turn=lambda ctx: called.append(ctx) is None and enabled)
        monkeypatch.setattr(strategy, '_build_observed_state', lambda *args: state())
        monkeypatch.setattr(strategy, '_candidate_state', lambda observed, hand: observed)
        monkeypatch.setattr(strategy, '_add_shallow_lookahead', lambda candidates, observed: candidates)
        if enabled:
            assert strategy.on_action(context)
        else:
            with pytest.raises(ValueError, match='结束按钮未确认'):
                strategy.on_action(context)
    assert called == [context, context]


def test_shared_strategy_optional_constraint_changes_actual_selected_card(monkeypatch):
    active, mental, _ = cards()
    hands = [SimpleNamespace(card=card, available=True) for card in (active, mental)]
    evaluator = Scores({active._id: 1000, mental._id: 1})
    selected = []
    context = SimpleNamespace(fetch_hands=lambda: hands, commit=lambda hand: selected.append(hand.card._id))
    for enabled in (False, True):
        strategy = PlannerStrategy(evaluator=evaluator, is_exam=True,
                                   selection_policy=policy(evaluator) if enabled else None)
        monkeypatch.setattr(strategy, '_build_observed_state', lambda *args: state())
        monkeypatch.setattr(strategy, '_candidate_state', lambda observed, hand: observed)
        monkeypatch.setattr(strategy, '_add_shallow_lookahead', lambda candidates, observed: candidates)
        assert strategy.on_action(context)
    assert selected == [active._id, mental._id]
