from dataclasses import replace
import json

import pytest

from kaa.idol_road.items import ItemCandidate, ItemNameRead, bind_item_reference
from kaa.idol_road.item_description import (
    ItemDescriptionCandidate, ItemDescriptionRead, ItemDescriptionLine, bind_item_description,
)
from kaa.idol_road.item_recovery_rule import interpret_item_recovery


TEXT='アクティブスキルカード使用時、直前にメンタルスキルカードを使用した状態の場合、体力回復6'


def bound(text=TEXT):
    reference=bind_item_reference(tuple(ItemNameRead('frame',kind,'道具',.999)
        for kind in ('original','grayscale')),(ItemCandidate('recovery','道具'),ItemCandidate('other','道具')))
    candidates=tuple(ItemDescriptionCandidate(identity,'道具',json.dumps([{'text':body}]))
        for identity,body in (('recovery',text),('other','別の説明')))
    reads=tuple(ItemDescriptionRead('frame',kind,(ItemDescriptionLine(text,.999),))
        for kind in ('original','grayscale'))
    return bind_item_description(reference,reads,candidates),candidates


def test_previous_card_recovery_retains_user_confirmed_timing_and_unverified_effect_order():
    binding,candidates=bound()
    analysis=interpret_item_recovery(binding,candidates,previous_turn_timing_confirmed=True)
    rule=analysis.declaration
    assert analysis.status=='confirmed_visible_declaration'
    assert (rule.current_category,rule.previous_category,rule.recovery)==(
        'ProduceCardCategory_ActiveSkill','ProduceCardCategory_MentalSkill',6)
    assert rule.timing=='card_use' and rule.executable is False
    assert rule.effect_order=='unverified'
    assert rule.cross_round_semantics=='previous_turn_last_card'
    assert rule.timing_source=='user_confirmed_previous_turn_last_card'
    assert rule.remaining_uses is None


def test_timing_knowledge_is_not_generalized_to_other_same_worded_items():
    binding,candidates=bound()
    rule=interpret_item_recovery(binding,candidates).declaration
    assert rule and rule.cross_round_semantics==rule.timing_source=='unverified'


@pytest.mark.parametrize('change',[{'item_id':'other'},{'confirmed_description':TEXT.replace('6','9')},
                                  {'status':'resolved','candidate_ids':('other',)}])
def test_altered_binding_is_rejected_before_interpreting_resources(change):
    binding,candidates=bound()
    assert interpret_item_recovery(replace(binding,**change),candidates).status=='identity_unconfirmed'


@pytest.mark.parametrize('text',[TEXT+'（1回）',TEXT.replace('回復6','回復0'),TEXT.replace('回復6','回復-6'),
                                TEXT.replace('直前に','前のターンに'),TEXT+'追加効果'])
def test_other_limits_or_timing_require_a_separate_contract(text):
    binding,candidates=bound(text)
    result=interpret_item_recovery(binding,candidates)
    assert result.status=='unsupported_description' and result.declaration is None


def test_unavailable_or_changed_candidate_definitions_keep_identity_unknown():
    binding,candidates=bound()
    assert interpret_item_recovery(binding,None).declaration is None
    changed=(replace(candidates[0],description_json='[]'),candidates[1])
    assert interpret_item_recovery(binding,changed).status=='identity_unconfirmed'
