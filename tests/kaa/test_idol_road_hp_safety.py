from dataclasses import replace
from types import SimpleNamespace
import pytest
from kaa.db.skill_card import SkillCard
from kaa.battle_turn import AbsoluteTurnObservation
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint,SelectionPolicyStop
from kaa.tasks.produce.new.play_cards.planner.strategy import PlannerStrategy
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation,EvaluationConfidence
from kaa.idol_road.hp_safety import HpSafetyPolicy
from kaa.idol_road.hp_preparation import HpPreparationBudget,HpThresholdGoal,RecurringHpCost
from kaa.idol_road.input_policy import InputBoundPolicy
from kaa.idol_road.battle_inputs import BattleInputs
from kaa.idol_road.battle_observation import BattleObservationSession,BattleFrameValues,bind_battle_frame


def budget():
    return HpPreparationBudget((HpThresholdGoal(18,50,('frame',)),),RecurringHpCost(1,3,('frame',)),True)


def test_current_only_floor_can_skip_without_inventing_unknown_recurring_charge():
    unknown=HpPreparationBudget((),None,False)
    assert not HpSafetyPolicy(20,unknown,reserve_mode='prefer').end_turn(state()).stop
    assert HpSafetyPolicy(20,unknown,reserve_mode='strict').end_turn(state()).stop
    assert not HpSafetyPolicy(20,replace(unknown,goals=budget().goals),reserve_mode='prefer').end_turn(state()).stop
    assert HpSafetyPolicy(20,replace(unknown,goals=budget().goals)).end_turn(state()).stop


def test_current_native_availability_does_not_invent_a_cost_or_future_hp_balance():
    current=replace(state(turn=2,hp=52),genki=5,observed_effects=frozenset({'stamina_consumption_modifier'}))
    value=SkillCard.from_id('p_card-01-men-2_035')
    guard=HpSafetyPolicy(20,HpPreparationBudget((),None,False),reserve_mode='prefer')
    assert guard.select(current,(value,)).allowed_indices==(0,)
    review=guard.review.candidates[0]
    assert review.affordable is None and review.projected_hp is None and not review.projected_hp_known
    assert review.reason=='native_available_current_only_no_hp_target'
    restricted=HpSafetyPolicy(20,budget())
    assert restricted.select(current,(value,)).stop
    assert restricted.select(current,(value,)).reason=='hp_safety:current_payment_unknown'
    advisory=HpSafetyPolicy(20,budget(),reserve_mode='prefer')
    assert advisory.select(current,(value,)).allowed_indices==(0,)
    assert advisory.review.candidates[0].projected_hp is None
    assert 'field_goal_unconfirmed' in advisory.review.candidates[0].reason


def test_native_legal_trigger_does_not_turn_unreadable_good_duration_into_a_number():
    from kaa.tasks.produce.new.play_cards.planner.trigger import evaluate_trigger_id,TriggerVerdict
    value=SkillCard.from_id('p_card-01-act-1_021')
    current=replace(state(turn=5,hp=47),genki=16,
        observed_effects=frozenset({'stamina_consumption_modifier','parameter_buff'}),
        observed_effect_values=(('parameter_buff',None),))
    assert evaluate_trigger_id(value._play_produce_exam_trigger_id,current) is TriggerVerdict.UNKNOWN
    guard=HpSafetyPolicy(20,HpPreparationBudget((),None,False),reserve_mode='prefer')
    assert guard.select(current,(value,)).allowed_indices==(0,)
    assert guard.review.candidates[0].projected_hp is None
    assert evaluate_trigger_id(value._play_produce_exam_trigger_id,current) is TriggerVerdict.UNKNOWN
    assert HpSafetyPolicy(20,budget()).select(current,(value,)).stop
    advisory=HpSafetyPolicy(20,budget(),reserve_mode='prefer')
    assert advisory.select(current,(value,)).allowed_indices==(0,)
    assert advisory.review.candidates[0].projected_hp is None


def test_native_available_special_payment_preserves_unknown_duration_and_disallows_future_growth():
    from kaa.tasks.produce.new.play_cards.planner.costs import special_cost_affordability
    value=SkillCard.from_id('p_card-01-men-2_037')
    current=replace(state(turn=5,hp=47),genki=16,
        observed_effects=frozenset({'parameter_buff'}),observed_effect_values=(('parameter_buff',None),))
    assert special_cost_affordability(current,value) is None
    guard=HpSafetyPolicy(20,HpPreparationBudget((),None,False),reserve_mode='prefer')
    assert guard.select(current,(value,)).allowed_indices==(0,)
    assert guard.review.candidates[0].affordable is None and guard.review.candidates[0].projected_hp is None
    assert guard.select(replace(current,unresolved_growth_card_ids=frozenset({value._id})),(value,)).stop


def state(turn=15,hp=40):
    return BattleState(20-turn+1,hp,0,max_hp=53,is_exam=True,
        absolute_turn_observation=AbsoluteTurnObservation('battle',f'frame-{turn}',turn,20-turn+1))


def card(key,cost):
    base=SkillCard.from_id('p_card-01-act-1_001')
    assert base is not None
    return replace(base,_id=key,stamina=0,force_stamina=cost,cost_value=0)


def input_policy(current,delegate,guard):
    """Pure bound-frame fixture; never device evidence or a live session."""
    absolute=current.absolute_turn_observation
    session=BattleObservationSession(absolute.battle_id,'ttmr',20,20,'confirmed_formation_start_request','variable_budget')
    frame=bind_battle_frame(session,BattleFrameValues(absolute.frame_id,True,current.remaining_turns,current.hp))
    frame=replace(frame,absolute_turn=absolute,observation=replace(frame.observation,
        turn=absolute.turn,turn_source='owned_card_log_return'))
    return InputBoundPolicy(BattleInputs(session),lambda:frame,delegate,guard)


def test_all_current_candidates_are_filtered_before_delegated_preferences():
    current=state()
    seen=[]
    delegate=SimpleNamespace(select=lambda state,cards:seen.append(tuple(card._id for card in cards)) or SelectionConstraint((0,),reason='ordinary'))
    policy=input_policy(current,delegate,HpSafetyPolicy(20,budget()))
    result=policy.select(current,(card('too-costly',8),card('safe',1)))
    assert seen==[('safe',)] and result.allowed_indices==(1,) and result.preferred_indices==(1,)


@pytest.mark.parametrize('changes',[{'hp_known':False},{'max_hp':None}])
def test_unknown_budget_stops_without_falling_back_to_other_goal(changes):
    delegate=SimpleNamespace(select=lambda *args:pytest.fail('must not fall back'))
    policy=input_policy(state(),delegate,HpSafetyPolicy(20,budget()))
    assert policy.select(replace(state(),**changes),(card('safe',0),)).stop


def test_existing_long_horizon_deficit_allows_HP_preservation_not_more_spending():
    result=HpSafetyPolicy(20,budget()).select(state(turn=1,hp=50),(card('free',0),card('blood',1)))
    assert result.allowed_indices==(0,) and not result.stop and 'below_reserve' in result.reason


def test_preferred_future_reserve_does_not_reject_known_current_payment():
    guard=HpSafetyPolicy(20,budget(),reserve_mode='prefer')
    result=guard.select(state(turn=1,hp=50),(card('free',0),card('blood',1)))
    assert result.allowed_indices==(0,1) and guard.review.current_route.status=='below_reserve'
    assert guard.review.candidates[1].projected_hp==49
    assert guard.select(state(turn=1,hp=0),(card('unpaid',1),)).stop


def test_imminent_goal_prefers_a_preserving_card_when_one_is_available():
    guard=HpSafetyPolicy(20,budget(),reserve_mode='prefer')
    result=guard.select(state(turn=17,hp=31),(card('spend',2),card('preserve',0)))
    assert result.allowed_indices==(1,)
    assert guard.review.candidates[0].reason=='imminent_field_hp_preferred_alternative'


def test_current_skip_estimate_can_preserve_imminent_goal_without_granting_a_reward():
    from kaa.idol_road.current_skip import CurrentSkipAssessment
    current=state(turn=17,hp=29)
    guard=HpSafetyPolicy(20,budget(),reserve_mode='prefer')
    guard.current_skip=CurrentSkipAssessment('frame-17',0,status='known_operations',
        declared_next_charge=3,known_operations_min_hp=28)
    delegate=SimpleNamespace(select=lambda *_:pytest.fail('must preserve the current SKIP alternative'))
    result=input_policy(current,delegate,guard).select(current,(card('spend',1),))
    assert result.end_turn and not result.stop and guard.current_skip.survival_guaranteed is False
    guard.current_skip=replace(guard.current_skip,frame_id='old')
    assert not guard.select(current,(card('spend',1),)).end_turn


def test_unsecured_bonus_does_not_abort_a_known_legal_skip_in_preference_mode():
    assert not HpSafetyPolicy(20,budget(),reserve_mode='prefer').end_turn(state(turn=17,hp=3)).stop


@pytest.mark.parametrize('hp,stopped',[(30,False),(29,True),(3,True)])
def test_imminent_percentage_goal_controls_end_turn(hp,stopped):
    assert HpSafetyPolicy(20,budget()).end_turn(state(turn=17,hp=hp)).stop is stopped


def test_zero_hp_keeps_affordable_free_or_genki_paid_cards_and_rejects_unpaid_hp_cost():
    current=replace(state(turn=1,hp=0),genki=3)
    genki_card=replace(card('genki-paid',0),stamina=3)
    guard=HpSafetyPolicy(20,budget())
    result=guard.select(current,(card('free',0),genki_card,card('hp-paid',1)))
    assert not result.stop and result.allowed_indices==(0,1)
    assert all(row.projected_hp==0 for row in guard.review.candidates[:2])
    assert guard.review.candidates[2].affordable is False


def test_paying_exact_current_hp_can_reach_zero_when_no_future_hp_goal_requires_a_reserve():
    no_goals=replace(budget(),goals=())
    guard=HpSafetyPolicy(20,no_goals)
    assert not guard.select(state(turn=1,hp=1),(card('one',1),)).stop
    assert guard.review.candidates[0].projected_hp==0


def test_zero_hp_does_not_stop_skip_but_does_not_satisfy_an_imminent_hp_threshold():
    assert not HpSafetyPolicy(20,budget()).end_turn(state(turn=1,hp=0)).stop
    assert HpSafetyPolicy(20,budget()).end_turn(state(turn=17,hp=0)).stop
    assert HpSafetyPolicy(20,budget()).select(state(turn=1,hp=-1),(card('free',0),)).stop


def test_early_end_turn_does_not_claim_all_future_healing_or_thresholds():
    result=HpSafetyPolicy(20,budget()).end_turn(state(turn=1,hp=50))
    assert not result.stop and 'future_goals_unconfirmed' in result.reason
    assert HpSafetyPolicy(20,replace(budget(),recurring_known=False)).end_turn(state()).stop


def test_end_of_observed_window_does_not_precharge_an_unobserved_extra_turn():
    result=HpSafetyPolicy(20,budget()).end_turn(state(turn=20,hp=1))
    assert not result.stop and 'extra_turn_unconfirmed' in result.reason


def test_preserving_last_card_cannot_issue_unsafe_end_turn():
    delegate=SimpleNamespace(select=lambda *args:SelectionConstraint(end_turn=True,reason='preserve_last'))
    policy=input_policy(state(turn=17,hp=29),delegate,HpSafetyPolicy(20,budget()))
    result=policy.select(state(turn=17,hp=29),(card('free',0),))
    assert result.stop and 'end_turn_threshold_unmet' in result.reason


def test_supported_projection_boundary_does_not_assume_unknown_side_effects_are_safe():
    altered=replace(card('unknown',0),_produce_card_status_enchant_id='unknown-side-effect')
    result=HpSafetyPolicy(20,budget()).select(state(),(altered,))
    assert result.stop


class Scores:
    def evaluate(self,card,state):
        return CardEvaluation(card._id,1000 if card._id=='unsafe' else 1,EvaluationConfidence.HIGH,'test')


@pytest.mark.parametrize('constraint,expected',[
    (SelectionConstraint(allowed_indices=(1,)), 'safe'),
    (SelectionConstraint((1,),allowed_indices=(1,)), 'safe'),
])
def test_shared_ranking_cannot_reintroduce_excluded_high_score_candidate(monkeypatch,constraint,expected):
    hands=[SimpleNamespace(card=card(key,0),available=True) for key in ('unsafe','safe')]
    selected=[]
    context=SimpleNamespace(fetch_hands=lambda:hands,commit=lambda hand:selected.append(hand.card._id))
    policy=SimpleNamespace(select=lambda *args:constraint,committed=lambda *args:None)
    strategy=PlannerStrategy(evaluator=Scores(),is_exam=True,selection_policy=policy)
    monkeypatch.setattr(strategy,'_build_observed_state',lambda *args:state())
    monkeypatch.setattr(strategy,'_candidate_state',lambda observed,hand:observed)
    monkeypatch.setattr(strategy,'_add_shallow_lookahead',lambda candidates,state:candidates)
    assert strategy.on_action(context) and selected==[expected]


@pytest.mark.parametrize('constraint',[
    SelectionConstraint(stop=True,reason='no-safe-action'),
    SelectionConstraint(allowed_indices=()),
    SelectionConstraint((0,),allowed_indices=(1,)),
    SelectionConstraint(allowed_indices=(999,)),
])
def test_shared_strategy_stops_before_commit_or_end_turn(monkeypatch,constraint):
    hands=[SimpleNamespace(card=card(key,0),available=True) for key in ('unsafe','safe')]
    context=SimpleNamespace(fetch_hands=lambda:hands,commit=lambda *args:pytest.fail('must not commit'))
    strategy=PlannerStrategy(evaluator=Scores(),is_exam=True,selection_policy=SimpleNamespace(select=lambda *args:constraint),
        end_turn=lambda *args:pytest.fail('must not skip'))
    monkeypatch.setattr(strategy,'_build_observed_state',lambda *args:state())
    monkeypatch.setattr(strategy,'_candidate_state',lambda observed,hand:observed)
    monkeypatch.setattr(strategy,'_add_shallow_lookahead',lambda candidates,state:candidates)
    with pytest.raises(SelectionPolicyStop):strategy.on_action(context)


@pytest.mark.parametrize('fallback',[True,False])
def test_hard_stop_cannot_turn_into_recommended_fallback(fallback):
    from kaa.tasks.produce.shared.cards import invoke_battle_strategy
    def stop(ctx):raise SelectionPolicyStop('hp-budget')
    with pytest.raises(SelectionPolicyStop):
        invoke_battle_strategy(SimpleNamespace(on_action=stop),None,allow_recommended_fallback=fallback)


def test_normal_non_policy_failure_keeps_legacy_fallback_behavior():
    from kaa.tasks.produce.shared.cards import invoke_battle_strategy
    def fail(ctx):raise RuntimeError('ordinary failure')
    assert invoke_battle_strategy(SimpleNamespace(on_action=fail),None,allow_recommended_fallback=True) is False


def test_review_explains_reserve_rejection_and_preserves_current_hand_indices():
    guard=HpSafetyPolicy(20,budget())
    result=guard.select(state(),(card('costly',8),card('safe',1)))
    assert result.allowed_indices==(1,)
    assert guard.review.frame_id=='frame-15' and guard.review.allowed_indices==(1,)
    assert [(row.index,row.card_id,row.reason,row.projected_hp) for row in guard.review.candidates]==[
        (0,'costly','reserve_spending_rejected',32),(1,'safe','allowed',39)]


def test_review_reports_unsupported_effect_instead_of_calling_it_unaffordable():
    from kaa.db.constants import ProduceExamEffectType
    base=card('draw',0)
    effect=replace(base.play_effects[0].produce_exam_effect,effect_type=ProduceExamEffectType.ExamCardDraw)
    altered=replace(base,play_effects=[replace(base.play_effects[0],produce_exam_effect=effect)])
    guard=HpSafetyPolicy(20,budget())
    assert guard.select(state(),(altered,)).stop
    row=guard.review.candidates[0]
    assert row.reason=='current_effects_unsupported' and row.affordable is True and row.projected_hp is None
    assert row.effect_types==('ProduceExamEffectType.ExamCardDraw',)


def test_invalid_current_observation_clears_previous_review_instead_of_reusing_allowed_cards():
    guard=HpSafetyPolicy(20,budget())
    assert not guard.select(state(),(card('safe',0),)).stop
    assert guard.select(replace(state(),hp_known=False),(card('safe',0),)).stop
    assert guard.review.candidates==() and guard.review.allowed_indices==() and guard.review.current_route is None


def test_pending_selection_preflight_clears_old_hp_review_without_touching_request():
    guard=HpSafetyPolicy(20,budget())
    policy=input_policy(state(),None,guard)
    assert not policy.select(state(),(card('safe',0),)).stop
    assert guard.review.allowed_indices==(0,)
    policy.inputs.begin('skip',policy.frame())
    before=policy.inputs.snapshot()
    assert policy.select(state(),(card('safe',0),)).stop
    assert guard.review is None and policy.inputs.snapshot()==before



def test_unknown_future_charge_keeps_current_payment_hard_and_route_unknown():
    unknown=replace(budget(),recurring=None,recurring_known=False)
    current=state(turn=10,hp=20)
    guard=HpSafetyPolicy(20,unknown,reserve_mode='prefer')
    result=guard.select(current,(card('legal',2),card('unaffordable',21)))
    assert result.allowed_indices==(0,)
    assert guard.review.current_route.status=='unknown'
    assert guard.review.candidates[0].route.status=='unknown'
    assert guard.review.candidates[0].projected_hp==18
    assert 'future_charge_unknown' in guard.review.candidates[0].reason
    assert guard.review.candidates[1].affordable is False
    assert HpSafetyPolicy(20,unknown).select(current,(card('legal',2),)).stop


def test_imminent_unknown_charge_prefers_current_threshold_without_proving_next_round():
    unknown=replace(budget(),recurring=None,recurring_known=False)
    guard=HpSafetyPolicy(20,unknown,reserve_mode='prefer')
    current=state(turn=17,hp=29)
    result=guard.select(current,(card('crosses',4),card('preserves',1)))
    assert result.allowed_indices==(1,)
    assert all(row.route.status=='unknown' for row in guard.review.candidates)
    assert guard.review.candidates[0].reason=='imminent_field_current_hp_preferred_future_charge_unknown'
    assert not guard.end_turn(current).stop
    assert 'field_goal_unconfirmed' in guard.end_turn(current).reason


def test_deliberate_hold_can_end_turn_without_authorizing_an_unsupported_card():
    altered=replace(card('unknown',0),_produce_card_status_enchant_id='unknown-side-effect')
    current=state(turn=1,hp=40)
    guard=HpSafetyPolicy(20,budget(),reserve_mode='prefer')
    assert guard.select(current,(altered,)).stop
    hold=SimpleNamespace(select=lambda *args:SelectionConstraint(end_turn=True,reason='preserve_pure_scoring'))
    result=input_policy(current,hold,guard).select(current,(altered,))
    assert result.end_turn and not result.stop and result.allowed_indices==() and not result.preferred_indices
    attempt=SimpleNamespace(select=lambda *args:SelectionConstraint((0,),reason='play_it'))
    assert input_policy(current,attempt,guard).select(current,(altered,)).stop
    strict=state(turn=17,hp=3)
    assert input_policy(strict,hold,HpSafetyPolicy(20,budget())).select(strict,(altered,)).stop
