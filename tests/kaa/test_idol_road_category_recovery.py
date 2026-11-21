from dataclasses import replace
import pytest
from kaa.idol_road.category_rule import CategoryConditionEvidence,read_category_condition,category_objectives
from kaa.idol_road.rule_symbols import RuleSymbolEvidence
from kaa.idol_road.category_priority import CategoryPriority
from tests.kaa.test_idol_road_category_priority import reads,rule,objectives,cards,state,Scores,MENTAL,ACTIVE

M_RECOVERY='直前にメンタルスキルカードを使用していた場合、低下状態回復1'
A_DRAW='直前にアクティブスキルカードを使用していた場合、以降、ターン開始後、スキルカードを引く'


def evidence(text=A_DRAW):
    return CategoryConditionEvidence('frame',(100,100,500,100),RuleSymbolEvidence('frame','active',(80,0,38,38),.99),reads(text))


def test_recovery_and_draw_keep_their_distinct_reward_metadata():
    recovery=read_category_condition(2,reads(M_RECOVERY))
    draw=read_category_condition(4,reads(A_DRAW))
    assert (recovery.deadline_turn,recovery.category,recovery.reward_kind,recovery.reward_count)==(1,MENTAL,'debuff_recovery',1)
    assert (draw.deadline_turn,draw.category,draw.reward_kind,draw.reward_count)==(3,ACTIVE,'subsequent_start_draw',None)


@pytest.mark.parametrize('text',[M_RECOVERY.replace('回復1','回復2'),A_DRAW+'2枚',
    A_DRAW.replace('以降、',''),A_DRAW.replace('直前に','次に'),M_RECOVERY.replace('場合、','場合以外、')])
def test_other_reward_shapes_are_not_silently_generalized(text):
    assert read_category_condition(4,reads(text)) is None


def test_only_verified_same_body_icon_mask_can_recover_low_confidence_text():
    raw=reads(A_DRAW.replace('直前にア','直前にAア'),confidence=.97)
    assert read_category_condition(4,raw) is None
    assert read_category_condition(4,raw,evidence()).category==ACTIVE
    for cue in (replace(evidence(),frame_id='old'),replace(evidence(),body_rect=(700,100,500,100)),
        replace(evidence(),icon=replace(evidence().icon,kind='mental'))):
        assert read_category_condition(4,raw,cue) is None


def test_high_confidence_original_reward_conflict_blocks_masked_recovery():
    assert read_category_condition(4,reads(A_DRAW.replace('以降、ターン開始後、スキルカードを引く','低下状態回復1')),evidence()) is None
    assert read_category_condition(4,reads(A_DRAW.replace('スキルカードを引く','スキルカードを引かない')),evidence()) is None
    mixed=(reads(M_RECOVERY)[0],reads(A_DRAW,confidence=.97)[1])
    assert read_category_condition(4,mixed,evidence()) is None


def test_same_category_different_rewards_and_misplaced_cue_do_not_merge():
    assert not category_objectives('ttmr',18,objectives(rule(4,A_DRAW),rule(4,A_DRAW.replace('以降、ターン開始後、スキルカードを引く','低下状態回復1'))))
    low=replace(rule(4,A_DRAW,confidence=.97),category_condition=evidence())
    assert not category_objectives('ttmr',18,objectives(low))


def test_draw_precondition_prefers_last_active_without_crediting_future_cards():
    active,mental,_=cards()
    goal=read_category_condition(4,reads(A_DRAW))
    chosen=CategoryPriority(18,(goal,),evaluator=Scores({active._id:1,mental._id:1000}))
    value=state(remaining_turns=16)
    result=chosen.select(value,(active,mental))
    assert result.preferred_indices==(0,) and 'reward_unconfirmed' in result.reason
    assert value.hand_card_ids==() and value.draw_pile_card_ids==()
