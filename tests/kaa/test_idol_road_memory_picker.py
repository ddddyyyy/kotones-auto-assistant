from dataclasses import replace
import pytest
from kaa.idol_road.formation import FormationSlot,FormationSlots
from kaa.idol_road.memory_picker import MemoryCandidate,SubMemoryViewport,choose_sub_alternative,replacement_matches_observed_powers
from kaa.idol_road.rules import RuleTextRead,RuleTextLine


def confirm_reads(frame='picker',number=1):
    return tuple(RuleTextRead(frame,kind,(RuleTextLine(f'サブ{number}決定',.99),)) for kind in ('original','grayscale'))


def lineup(frame='before',powers=(10,20,30)):
    return FormationSlots(frame,tuple(FormationSlot('main' if i==0 else 'sub',(i*100,700,90,90),
        'occupied',power,'a'*64) for i,power in enumerate(powers)))


def candidate(power=40,index=0,**changes):
    return replace(MemoryCandidate('picker',(20+index*100,800,90,90),power,'b'*64,False,False),**changes)


def viewport(*items):
    return SubMemoryViewport('picker',(candidate(20,index=0,assigned=True,highlighted=True),*items),True,
        confirm_rect=(300,1050,190,70),confirm_reads=confirm_reads(),confirm_control_score=.99)


def test_visible_alternative_is_sampled_without_full_roster_or_memory_id_inference():
    first,second=candidate(40,1),candidate(50,2)
    seen=[]
    chosen=choose_sub_alternative(viewport(first,second),lineup(),1,
        choose_index=lambda size:seen.append(size) or size-1)
    assert chosen==second and seen==[2]


@pytest.mark.parametrize('changes',[{'assigned':True},{'assigned':None},{'highlighted':True},
    {'highlighted':None},{'frame_id':'stale'},{'power':30},{'power':None},{'presentation_sha256':'bad'}])
def test_current_assigned_ambiguous_stale_or_unchanged_candidates_are_not_chosen(changes):
    assert choose_sub_alternative(viewport(candidate(**changes)),lineup(),1) is None


def test_missing_native_group_binding_or_wrong_slot_is_not_a_random_change():
    view=viewport(candidate())
    assert choose_sub_alternative(replace(view,editing_group_confirmed=False),lineup(),1) is None
    assert choose_sub_alternative(view,lineup(),0) is None
    assert choose_sub_alternative(replace(view,candidates=(candidate(),)),lineup(),1) is None


def test_post_selection_check_accepts_sub_reordering_but_not_capacity_main_or_other_changes():
    selected=candidate()
    assert replacement_matches_observed_powers(lineup(),lineup('after',(10,30,40)),1,selected)
    for after in (lineup('after'),lineup('after',(11,40,30)),lineup('after',(10,40)),
                  lineup('before',(10,40,30)),lineup('after',(10,40,31))):
        assert not replacement_matches_observed_powers(lineup(),after,1,selected)


@pytest.mark.parametrize('number',[1,2,3])
def test_native_sub_number_comes_from_complete_bound_label(number):
    assert replace(viewport(candidate()),confirm_reads=confirm_reads(number=number)).sub_number==number


@pytest.mark.parametrize('changes',[{'confirm_reads':()}, {'confirm_reads':confirm_reads('old')},
    {'confirm_control_score':.96},{'confirm_reads':confirm_reads(number=2)}])
def test_wrong_unknown_or_unbound_sub_number_cannot_select_an_alternative(changes):
    assert choose_sub_alternative(replace(viewport(candidate()),**changes),lineup(),1) is None


def test_replacement_binds_the_actual_sub_number_before_picking():
    from kaa.idol_road.memory_replacement import MemoryReplacement
    change=MemoryReplacement(lineup(),slot_index=1);change.open()
    wrong=replace(viewport(candidate()),confirm_reads=confirm_reads(number=3))
    with pytest.raises(ValueError,match='Wrong native Sub number'):
        change.observe_selector(wrong,change.token)
    with pytest.raises(ValueError,match='Invalid native Sub slot'):
        MemoryReplacement(lineup(),slot_index=0)
