from dataclasses import replace
import pytest
from kaa.idol_road.score_priority import ScorePreparationObjective,ScorePreparationPriority,score_preparation_objectives
from kaa.idol_road.rules import RuleTextRead,RuleTextLine,VisibleStageRule,RuleViewport,StageRules
from kaa.idol_road.rule_symbols import RuleSymbolEvidence
from kaa.tasks.produce.new.play_cards.planner.state import BattleState
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation,EvaluationConfidence
from kaa.battle_turn import AbsoluteTurnObservation
from kaa.db.skill_card import SkillCard


def reads(text,confidence=.99,frame='frame'):
    return tuple(RuleTextRead(frame,kind,(RuleTextLine(text,confidence),)) for kind in ('original','grayscale'))


def row(threshold=11000):
    return VisibleStageRule((0,0,600,190),16,reads('16ターン目'),
        reads(f'スコアが{threshold}以上の場合、以降の2ターンの'),
        turn_addition=RuleSymbolEvidence('frame','turn_addition',(500,80,40,40),.99))


def rules(*rows):
    return StageRules('ttmr',20,(RuleViewport('viewport',rows or (row(),)),))


def test_goal_comes_from_complete_condition_line_without_claiming_reward():
    goals=score_preparation_objectives('ttmr',20,rules())
    assert len(goals)==1 and goals[0].score_at_least==11000 and goals[0].deadline_turn==15
    assert not goals[0].reward_confirmed and 'phase_unconfirmed' in goals[0].timing
    assert not score_preparation_objectives('ttmr',19,rules())
    assert not score_preparation_objectives('ttmr',20,replace(rules(),identity_source='history'))


@pytest.mark.parametrize('changes',[
    {'description_reads':reads('スコアが11000以上の場合、以降の2ターンの',.97)},
    {'description_reads':reads('スコアが11000以下の場合、以降の2ターンの')},
    {'description_reads':reads('スコアが11000以上')},
    {'turn_addition':None}, {'turn_addition':replace(row().turn_addition,score=.96)},
    {'turn_reads':reads('15ターン目')},
    {'description_reads':reads('スコアが11000以上の場合、以降の2ターンの',frame='old')},
])
def test_unknown_stale_partial_or_unsupported_declaration_is_not_a_goal(changes):
    assert not score_preparation_objectives('ttmr',20,rules(replace(row(),**changes)))


def test_duplicates_merge_and_conflicting_thresholds_block_instead_of_picking_one():
    assert len(score_preparation_objectives('ttmr',20,rules(row(),row())))==1
    assert not score_preparation_objectives('ttmr',20,rules(row(),row(12000)))
    invalid=replace(row(12000),turn_addition=replace(row().turn_addition,score=.9))
    assert score_preparation_objectives('ttmr',20,rules(row(),invalid))[0].score_at_least==11000


def state(turn=15,score=10900):
    return BattleState(remaining_turns=6,hp=50,genki=10,is_exam=True,current_score=score,
        absolute_turn_observation=AbsoluteTurnObservation('battle',f'frame-{turn}',turn,6))


class Scores:
    def evaluate(self,card,state):
        return CardEvaluation(card._id,1000 if card._id=='setup' else 1,EvaluationConfidence.HIGH,'test',
                              confirmed_output=0 if card._id=='setup' else 120)


def card(card_id):
    base=SkillCard.from_id('p_card-01-act-1_001')
    assert base is not None
    return replace(base,_id=card_id,stamina=0,force_stamina=0,cost_value=0)


def priority():
    return ScorePreparationPriority(20,(ScorePreparationObjective(16,11000,('frame',)),),evaluator=Scores())


def test_deadline_prefers_supported_score_progress_without_changing_observed_score():
    current=state()
    choice=priority().select(current,(card('setup'),card('attack')))
    assert choice.preferred_indices==(1,) and 'estimated_progress=100' in choice.reason
    assert current.current_score==10900 and current.absolute_turn_observation.turn==15
    assert not choice.end_turn and 'reward_unconfirmed' in choice.reason


def test_current_pure_score_primitive_can_prepare_goal_without_becoming_a_future_prefix():
    current=replace(state(),playable_count=2)
    attack=SkillCard.from_id('p_card-01-act-1_020')
    result=priority().select(current,(attack,card('setup')))
    assert result.preferred_indices==(0,)
    assert "('setup', 'p_card-01-act-1_020')" not in result.reason


def test_preferred_hp_reserve_does_not_disable_current_score_deadline():
    from tests.kaa.test_idol_road_hp_safety import budget
    current=replace(state(),hp=30,max_hp=53)
    policy=ScorePreparationPriority(20,(ScorePreparationObjective(16,11000,('frame',)),),
        evaluator=Scores(),hp_budget=budget(),strict_hp_reserve=False)
    assert policy.select(current,(card('attack'),)).preferred_indices==(0,)


@pytest.mark.parametrize('turn,score,reason',[(14,10900,'setup_window'),(15,None,'observation_unknown'),
    (15,11000,'prepared_reward_unconfirmed'),(16,10900,'')])
def test_early_unknown_prepared_and_late_states_do_not_force_a_score_card(turn,score,reason):
    result=priority().select(state(turn,score),(card('attack'),))
    assert not result.preferred_indices
    assert reason in result.reason


def test_turn_rollback_or_battle_change_does_not_reuse_old_goal_state():
    policy=priority()
    policy.select(state(),(card('attack'),))
    assert 'observation_unknown' in policy.select(state(14),(card('attack'),)).reason
    changed=replace(state(),absolute_turn_observation=AbsoluteTurnObservation('other','new',15,6))
    assert 'observation_unknown' in policy.select(changed,(card('attack'),)).reason


def test_field_current_last_card_constraint_stays_above_score_deadline(monkeypatch):
    from kaa.idol_road.field_priority import FieldPriority
    from kaa.tasks.produce.new.play_cards.planner.selection_policy import SelectionConstraint
    field=FieldPriority(20,(),(),score=(ScorePreparationObjective(16,11000,('frame',)),))
    monkeypatch.setattr(field.category,'select',lambda *args,**kwargs:SelectionConstraint(end_turn=True,reason='last_card'))
    monkeypatch.setattr(field.score,'select',lambda *args,**kwargs:pytest.fail('must preserve current last card'))
    assert field.select(state(),()).end_turn


def test_two_visible_cards_compare_sequence_progress_but_not_future_draws():
    class SequenceScores:
        def evaluate(self,card,state):
            output=0 if card._id=='setup' else 220 if state.playable_count==1 else 120
            return CardEvaluation(card._id,1,EvaluationConfidence.HIGH,'test',confirmed_output=output)
    policy=ScorePreparationPriority(20,(ScorePreparationObjective(16,11000,('frame',)),),evaluator=SequenceScores())
    result=policy.select(replace(state(score=10700),playable_count=2),(card('setup'),card('attack')))
    assert result.preferred_indices==(0,) and "('setup', 'attack')" in result.reason


def test_future_cost_needs_model_proof_even_with_current_available_cards(monkeypatch):
    from kaa.idol_road import score_priority
    monkeypatch.setattr(score_priority,'can_afford_play',lambda *args:None)
    result=priority().select(replace(state(),playable_count=2),(card('setup'),card('attack')),current_cards_available=True)
    assert result.preferred_indices==(1,)
    assert "('setup', 'attack')" not in result.reason


def test_current_score_cannot_be_projected_into_a_next_card_trigger():
    triggered=replace(card('attack'),_play_produce_exam_trigger_id='unmodeled_score_condition')
    result=priority().select(replace(state(),playable_count=2),(card('setup'),triggered))
    assert not result.preferred_indices


@pytest.mark.parametrize('turn',[1,15])
def test_unknown_active_score_goal_stops_before_ordinary_ranking(turn):
    from kaa.idol_road.field_priority import FieldPriority
    from kaa.idol_road.ranked_field_policy import RankedFieldPolicy
    from types import SimpleNamespace
    field=FieldPriority(20,(),(),score=(ScorePreparationObjective(16,11000,('frame',)),))
    evaluator=SimpleNamespace(evaluate=lambda *args:pytest.fail('unknown score must not fall through to ordinary ranking'))
    policy=RankedFieldPolicy(field,evaluator=evaluator)
    current=replace(state(turn,None),remaining_turns=21-turn,
        absolute_turn_observation=AbsoluteTurnObservation('battle',f'frame-{turn}',turn,21-turn))
    result=policy.select(current,(card('attack'),))
    assert result.stop and result.allowed_indices==() and not result.end_turn
    assert result.reason=='field_score:observation_unknown'
    assert policy.trace.evaluations==()


def test_unknown_score_without_an_active_goal_does_not_create_a_new_restriction():
    assert not priority().select(state(16,None),(card('attack'),)).stop
    assert not ScorePreparationPriority(20,()).select(state(1,None),(card('attack'),)).stop
