from dataclasses import replace
import pytest
from kaa.idol_road.hp_preparation import HpThresholdGoal,RecurringHpCost,HpPreparationBudget,assess_hp_route,hp_preparation_budget
from kaa.idol_road.rules import StageRules,RuleViewport,VisibleStageRule,RuleTextRead,RuleTextLine
from kaa.idol_road.planning_turn import PlanningTurn
from kaa.tasks.produce.new.play_cards.planner.state import BattleState


def reads(*texts,confidence=.99):
    return tuple(RuleTextRead('frame',kind,tuple(RuleTextLine(text,confidence) for text in texts)) for kind in ('original','grayscale'))


def rule(turn,*texts):
    return VisibleStageRule((0,0,600,190),turn,reads(f'{turn}ターン目'),reads(*texts))


def rules(*rows):
    return StageRules('ttmr',20,(RuleViewport('view',rows),))


def test_complete_declarations_supply_budget_without_reward_projection():
    result=hp_preparation_budget('ttmr',20,rules(rule(1,'以降、ターン開始時、','体力消費3'),rule(18,'体力が50%以上の場合、')))
    assert result.recurring_known and result.recurring.amount==3
    assert result.goals==(HpThresholdGoal(18,50,('frame',)),)
    assert 'unconfirmed' in result.timing


def test_weak_timing_cannot_be_replaced_by_a_known_cost_number():
    row=rule(1,'以降、ターン開始時、','体力消費3')
    first=replace(row.description_reads[0],lines=(RuleTextLine('以降、ターン開始時、',.95),RuleTextLine('体力消費3',.99)))
    result=hp_preparation_budget('ttmr',20,rules(replace(row,description_reads=(first,row.description_reads[1])),rule(18,'体力が50%以上の場合')))
    assert result.goals and result.recurring is None and not result.recurring_known


def test_conflicting_costs_thresholds_and_wrong_owner_are_not_selected():
    result=hp_preparation_budget('ttmr',20,rules(rule(1,'以降、ターン開始時、','体力消費3'),rule(1,'以降、ターン開始時、','体力消費4'),
                                               rule(18,'体力が50%以上の場合'),rule(18,'体力が60%以上の場合')))
    assert not result.recurring_known and not result.goals
    assert hp_preparation_budget('ttmr',19,rules()) is None
    assert hp_preparation_budget('ttmr',20,replace(rules(),identity_source='history')) is None


def budget():
    return HpPreparationBudget((HpThresholdGoal(18,50,('frame',)),),RecurringHpCost(1,3,('frame',)),True)


def state(hp=40):
    return BattleState(6,hp,10,max_hp=53)


@pytest.mark.parametrize('hp,status',[(36,'safe'),(35,'below_reserve'),(0,'below_reserve')])
def test_odd_maximum_rounds_up_and_reserves_all_supported_future_starts(hp,status):
    before=state()
    result=assess_hp_route(before,state(hp),PlanningTurn('absolute','battle',15,20),budget())
    assert (result.required_hp,result.reserve_cost)==(36,9)
    assert result.status==status and before.hp==40


@pytest.mark.parametrize('changes',[{'max_hp':None},{'max_hp':0},{'hp_known':False},{'hp':54}])
def test_unknown_maximum_or_hp_never_becomes_a_safe_route(changes):
    before=replace(state(),**changes)
    assert assess_hp_route(before,before,PlanningTurn('absolute','battle',15,20),budget()).status=='unknown'


def test_missing_cost_not_zero_and_unseen_added_turns_not_reserved():
    unknown=replace(budget(),recurring=None,recurring_known=False)
    assert assess_hp_route(state(),state(),PlanningTurn('absolute','battle',15,20),unknown).status=='unknown'
    assert assess_hp_route(state(),state(),PlanningTurn('absolute','battle',15,17),budget()).status=='not_applicable'


def test_score_override_excludes_unknown_or_unsafe_hp_route():
    from kaa.idol_road.score_priority import ScorePreparationPriority,ScorePreparationObjective
    from kaa.db.skill_card import SkillCard
    from kaa.battle_turn import AbsoluteTurnObservation
    from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation,EvaluationConfidence
    class Scores:
        def evaluate(self,card,state):
            return CardEvaluation(card._id,1,EvaluationConfidence.HIGH,'test',confirmed_output=200)
    card=SkillCard.from_id('p_card-01-act-1_001')
    assert card is not None
    card=replace(card,stamina=0,force_stamina=0,cost_value=0)
    current=replace(state(),is_exam=True,current_score=10900,absolute_turn_observation=AbsoluteTurnObservation('battle','frame',15,6))
    policy=lambda value:ScorePreparationPriority(20,(ScorePreparationObjective(16,11000,('frame',)),),evaluator=Scores(),hp_budget=value)
    assert policy(budget()).select(current,(card,)).preferred_indices==(0,)
    assert not policy(budget()).select(replace(current,max_hp=None),(card,)).preferred_indices
    assert not policy(budget()).select(replace(current,hp=35),(card,)).preferred_indices
    assert not policy(replace(budget(),recurring_known=False)).select(current,(card,)).preferred_indices


def test_context_maximum_requires_matching_ratio_evidence():
    import numpy as np
    from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame
    from kaa.idol_road.hp_ratio import HpRatioRead,confirm_hp_ratio
    from kaa.idol_road.planner_context import IdolRoadBattleContext
    session=BattleObservationSession('battle','ttmr',20,20,'confirmed_formation_start_request')
    ratio=confirm_hp_ratio('frame',(10,20,50,20),.99,tuple(HpRatioRead('frame',kind,'40/53',.99) for kind in ('original','grayscale')))
    frame=bind_battle_frame(session,BattleFrameValues('frame',True,6,40,hp_ratio=ratio))
    screen=np.zeros((1280,720,3),dtype='uint8')
    assert IdolRoadBattleContext(screen,frame).fetch_exam_max_hp()==53
    forged=replace(frame,observation=replace(frame.observation,max_hp=99))
    assert IdolRoadBattleContext(screen,forged).fetch_exam_max_hp() is None
    assert IdolRoadBattleContext(screen,replace(frame,values=replace(frame.values,hp_ratio=None))).fetch_exam_max_hp() is None
