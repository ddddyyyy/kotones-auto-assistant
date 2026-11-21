"""场地时点与末手缓存的纯规则测试；合成回合不作为实机证明。"""
from dataclasses import replace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.idol_road.planning_turn import PlanningTurnTracker,read_planning_turn
from kaa.idol_road.category_priority import CategoryPriority
from kaa.idol_road.category_rule import CategoryObjective
from kaa.idol_road.concentration_priority import ConcentrationPriority,ConcentrationObjective
from kaa.idol_road.concentration_rule import ConcentrationDoubleDeclaration
from kaa.idol_road.good_condition_priority import GoodConditionPriority
from kaa.idol_road.good_condition_rule import GoodConditionRequirement
from kaa.tasks.produce.new.play_cards.planner.state import BattleState,ExamBuffState
from kaa.tasks.produce.new.play_cards.planner.turn_observation import AbsoluteTurnObservation
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation,EvaluationConfidence
from kaa.tasks.produce.new.play_cards.planner.feedback import PendingPlay,PlayFeedback,PlayFeedbackStatus

MENTAL='ProduceCardCategory_MentalSkill'


def state(turn=16,remaining=5,frame='frame',battle='battle',concentration=10,good=4):
    return BattleState(remaining,50,0,is_exam=True,
        absolute_turn_observation=AbsoluteTurnObservation(battle,frame,turn,remaining),
        buffs=ExamBuffState(lesson_buff=concentration,parameter_buff_turns=good),
        observed_effect_values=(('lesson_buff',concentration),('parameter_buff',good)))


class Scores:
    def evaluate(self,card,state):
        return CardEvaluation(card._id,1000 if card.category=='ProduceCardCategory_ActiveSkill' else 1,
                              EvaluationConfidence.HIGH,'test')


def cards():
    active=SkillCard.from_id('p_card-01-act-1_001')
    mental=SkillCard.from_id('p_card-01-men-0_007')
    assert active is not None and mental is not None
    return active,mental


def category(*turns):
    return CategoryPriority(20,tuple(CategoryObjective(turn,MENTAL,1,('frame',)) for turn in turns),evaluator=Scores())


def confirm(policy,before,card,after=None):
    policy.committed(before,card)
    policy.feedback(PendingPlay(card._id,before,0,0),
        PlayFeedback(PlayFeedbackStatus.CONFIRMED,'card_left_hand'),after or before)


def test_actual_turn_is_not_derived_from_fixed_budget_and_remaining():
    current=read_planning_turn(state(21,2),20)
    assert current is not None and current.turn==21 and current.last_scheduled_turn==22
    assert current.key==('absolute','battle',21)
    assert read_planning_turn(state(),0).turn==16


@pytest.mark.parametrize('change',[
    {'remaining_turns_known':False},{'remaining_turns':4},
    {'projected_effect_values':()},
    {'absolute_turn_observation':AbsoluteTurnObservation('battle','frame',True,5)},
    {'absolute_turn_observation':AbsoluteTurnObservation('battle','',16,5)},
])
def test_unknown_conflicting_or_simulated_turn_does_not_authorize_field_plan(change):
    assert read_planning_turn(replace(state(),**change),20) is None


def test_tracker_latches_source_battle_and_rejects_regressions_without_reset():
    tracker=PlanningTurnTracker()
    assert tracker.observe(state(),20).turn==16
    assert tracker.observe(replace(state(),absolute_turn_observation=None),20) is None
    assert tracker.observe(state(16,5,'new','other'),20) is None
    assert tracker.observe(state(15,5,'old'),20) is None
    assert tracker.observe(state(17,5,'next'),20).turn==17


def test_tracker_rejects_same_frame_conflict_and_legacy_source_switch():
    tracker=PlanningTurnTracker()
    assert tracker.observe(state(),20) is not None
    assert tracker.observe(state(16,6),20) is None
    legacy=PlanningTurnTracker()
    assert legacy.observe(replace(state(),absolute_turn_observation=None),20).turn==16
    assert legacy.observe(state(),20) is None


def test_confirmed_last_card_does_not_leak_to_next_round_with_same_remaining():
    active,mental=cards()
    policy=category(17,18)
    confirm(policy,state(),mental)
    assert policy.select(state(),(active,)).end_turn
    assert not policy.select(state(17,5,'next'),(active,)).end_turn
    assert not policy.select(state(),(active,)).end_turn


def test_same_round_horizon_change_keeps_current_last_card():
    active,mental=cards()
    policy=category(17)
    confirm(policy,state(),mental,state(16,6,'new'))
    assert policy.select(state(16,6,'new'),(active,)).end_turn


@pytest.mark.parametrize('after',[
    state(17,5,'next'),state(16,5,'new','other'),replace(state(),absolute_turn_observation=None),
    replace(state(),remaining_turns_known=False),replace(state(),projected_effect_values=()),
])
def test_pending_last_card_feedback_requires_the_same_current_round(after):
    active,mental=cards()
    policy=category(17,18)
    confirm(policy,state(),mental,after)
    assert not policy.select(state(),(active,)).end_turn


def test_unknown_turn_discards_confirmed_cache_instead_of_reviving_on_recovery():
    active,mental=cards()
    policy=category(17)
    confirm(policy,state(),mental)
    assert policy.select(state(),(active,)).end_turn
    assert not policy.select(replace(state(),remaining_turns_known=False),(active,)).end_turn
    assert not policy.select(state(),(active,)).end_turn


def test_feedback_from_an_older_same_card_request_cannot_confirm_the_latest_commit():
    active,mental=cards()
    policy=category(17)
    before=state()
    policy.committed(before,mental)
    policy.committed(state(16,5,'later-request'),mental)
    policy.feedback(PendingPlay(mental._id,before,0,0),
        PlayFeedback(PlayFeedbackStatus.CONFIRMED,'card_left_hand'),state(16,5,'after'))
    assert not policy.select(state(16,5,'after'),(active,)).end_turn


def test_goals_after_initial_budget_use_only_the_current_scheduled_horizon():
    active,mental=cards()
    policy=category(22)
    assert policy.select(state(21,2),(active,mental)).preferred_indices==(1,)
    # No current remaining round after21: do not assume a later addition.
    assert not category(22).select(state(21,1),(active,mental)).preferred_indices


def test_concentration_goals_expire_using_actual_round_not_unchanged_remaining():
    active,mental=cards()
    policy=ConcentrationPriority(20,(ConcentrationObjective(ConcentrationDoubleDeclaration(17,42),('frame',)),),evaluator=Scores())
    before=policy.select(state(),(active,mental))
    assert 'prepare_by=16' in before.reason
    assert not policy.select(state(17,5,'next'),(active,mental)).reason


def test_good_condition_reserve_decreases_on_actual_round_with_unchanged_remaining():
    active,mental=cards()
    policy=GoodConditionPriority(20,(GoodConditionRequirement(18,4,('frame',)),),evaluator=Scores())
    assert 'current_reserve=6' in policy.select(state(),(active,mental)).reason
    assert 'current_reserve=5' in policy.select(state(17,5,'next'),(active,mental)).reason
    assert not policy.select(state(18,5,'third'),(active,mental)).reason


@pytest.mark.parametrize('kind',['concentration','good'])
def test_resource_goals_beyond_initial_budget_are_reachable_only_with_observed_horizon(kind):
    active,mental=cards()
    if kind=='concentration':
        factory=lambda:ConcentrationPriority(20,(ConcentrationObjective(ConcentrationDoubleDeclaration(22,42),('frame',)),),evaluator=Scores())
    else:
        factory=lambda:GoodConditionPriority(20,(GoodConditionRequirement(22,4,('frame',)),),evaluator=Scores())
    assert 'prepare_by=21' in factory().select(state(21,2),(active,mental)).reason
    assert not factory().select(state(21,1),(active,mental)).reason
