from dataclasses import replace
import pytest

from kaa.idol_road.good_condition_rule import read_good_condition_requirement, good_condition_requirements
from kaa.idol_road.rules import RuleTextLine, RuleTextRead, RuleViewport, VisibleStageRule, StageRules
from kaa.idol_road.models import Observation, Scene
from kaa.idol_road.preparation import prepare_battle
from kaa.idol_road.good_condition_rule import good_condition_planning_requirements
from kaa.idol_road.rule_symbols import RuleSymbolEvidence
from kaa.idol_road.good_condition_rule import GoodConditionLexicalEvidence


def lexical(body=None, **changes):
    return replace(GoodConditionLexicalEvidence('frame',(10,10,500,90),(170,12,190,30),.999,True,
        () if body is None else body),**changes)


def test_complete_current_suffix_disambiguates_only_fixed_kana_word():
    raw=reads('好調が19夕一ン以上の場合、効果')
    assert read_good_condition_requirement(11,raw).requirement is None
    result=read_good_condition_requirement(11,raw,lexical())
    assert result.requirement.minimum_turns==19 and result.reads==raw
    assert not result.requirement.executable


@pytest.mark.parametrize('change',[
    {'frame_id':'old'},{'unique':False},{'suffix_score':.969},
    {'suffix_rect':(170,80,190,30)},{'body_rect':(600,10,500,90)},
])
def test_suffix_evidence_must_belong_to_complete_current_body(change):
    assert read_good_condition_requirement(11,reads('好調が19夕一ン以上の場合、効果'),lexical(**change)).requirement is None


@pytest.mark.parametrize('kind',['number','comparator'])
def test_scaled_pair_cannot_override_strong_native_number_or_comparator(kind):
    native=(reads('好調が18夕一ン以上の場合、効果')[0],reads('好調が19夕一ン以上の場合、効果',confidence=.97)[1]) if kind=='number' else reads('好調が19夕一ン以下の場合、効果')
    result=read_good_condition_requirement(11,native,lexical(reads('好調が19夕一ン以上の場合、効果')))
    assert result.requirement is None and result.status=='high_confidence_conflict'


def test_complete_recycle_reward_can_prepare_condition_without_predicting_cards():
    body=reads('好調が19夕一ン以上の場合、除外にある消費体カが4以上のスキルカードを山札のランダムな位置に移動')
    item=VisibleStageRule((0,0,600,200),11,reads('11ターン目'),body,good_condition_lexical=lexical())
    goals=good_condition_planning_requirements('ttmr',19,rules(item))
    assert len(goals)==1 and goals[0].minimum_turns==19 and not goals[0].reward_confirmed
    for changed in (replace(item,card_rect=(0,0,100,100)),replace(item,description_reads=reads('好調が19夕一ン以上の場合、除外にある消費体カが4以下のスキルカードを山札のランダムな位置に移動'))):
        assert not good_condition_planning_requirements('ttmr',19,rules(changed))


def reads(text='好調が4ターン以上の場合、以降、ターン', *, frame='frame', confidence=.99):
    return tuple(RuleTextRead(frame, kind, (RuleTextLine(text, confidence),))
                 for kind in ('original', 'grayscale'))


def row(*, turn=6, body=None, frame='frame'):
    body = reads(frame=frame) if body is None else body
    return VisibleStageRule((1, 1, 100, 100), turn, reads(f'{turn}ターン目', frame=frame), body)


def rules(*rows):
    return StageRules('ttmr', 19, (RuleViewport('page', rows or (row(),)),))


@pytest.mark.parametrize('text,minimum', [
    ('好調状態の場合、絶好調7ターン', 1),
    ('好調が４ターン以上の場合、以降、ターン', 4),
    ('好調が12ターン以上の場合、未知の効果', 12),
])
def test_only_requirement_is_known_without_reward_or_phase(text, minimum):
    result = read_good_condition_requirement(6, reads(text))
    assert result.status == 'confirmed_condition'
    goal = result.requirement
    assert goal is not None and goal.minimum_turns == minimum
    assert goal.scope == 'condition_only'
    assert not goal.reward_confirmed and not goal.trigger_phase_confirmed and not goal.executable


def test_unreadable_reward_does_not_erase_separately_readable_condition():
    body = tuple(replace(read, lines=read.lines + (RuleTextLine(
        '終了時、スコア+10' if read.representation == 'original' else '終了時、スコア-10', .91),))
        for read in reads())
    result = read_good_condition_requirement(6, body)
    assert result.requirement is not None and result.requirement.minimum_turns == 4
    assert not result.requirement.reward_confirmed


@pytest.mark.parametrize('text', [
    '好調が4ターン以下の場合、以降、ターン',
    '好調が4ターン未満の場合、以降、ターン',
    '絶好調が4ターン以上の場合、以降、ターン',
    '好調が0ターン以上の場合、以降、ターン',
    '好調が04ターン以上の場合、以降、ターン',
    '以降、好調が4ターン以上の場合、',
    '好調状態ではない場合、絶好調7ターン',
    '好調が4ターン以上の場合かつ集中が10以上の場合、効果',
    '好調が4ターン以上の場合',
])
def test_different_comparator_resource_or_incomplete_clause_is_not_target(text):
    assert read_good_condition_requirement(6, reads(text)).requirement is None


@pytest.mark.parametrize('body', [
    reads(confidence=.979), reads()[:1],
    (reads()[0], replace(reads()[1], frame_id='old')),
    (reads()[0], reads('好調が5ターン以上の場合、以降')[1]),
    reads('好調が4ターン以上の場合、', frame=''),
])
def test_low_confidence_conflicting_or_stale_evidence_cannot_authorize_target(body):
    assert read_good_condition_requirement(6, body).requirement is None


def test_wrapped_clause_requires_every_line_up_to_boundary():
    body = tuple(replace(read, lines=(RuleTextLine('好調が4ターン', .99),
                                     RuleTextLine('以上の場合、', .99))) for read in reads())
    assert read_good_condition_requirement(6, body).requirement is not None
    bad = (body[0], replace(body[1], lines=(body[1].lines[0], RuleTextLine('以上の場合、', .97))))
    assert read_good_condition_requirement(6, bad).requirement is None


def test_scope_and_independent_turn_evidence_are_required():
    for change in ({'stage': 18}, {'source': 'database'}, {'identity_source': 'history'}):
        assert not good_condition_requirements('ttmr', 19, replace(rules(), **change))
    stale = row(body=reads(frame='old'))
    assert not good_condition_requirements('ttmr', 19, rules(stale))
    unconfirmed = replace(row(), turn_reads=reads('6ターン目', confidence=.97))
    assert not good_condition_requirements('ttmr', 19, rules(unconfirmed))


def test_deduplication_preserves_frames_without_adding_turns():
    result = good_condition_requirements('ttmr', 19, rules(row(), row(frame='second'), row()))
    assert len(result) == 1 and result[0].minimum_turns == 4
    assert result[0].evidence_frames == ('frame', 'second')


@pytest.mark.parametrize('conflicting', [
    row(body=reads('好調が5ターン以上の場合、効果')),
    row(body=reads('好調が4ターン以下の場合、効果')),
    row(body=(reads()[0], reads('好調が5ターン以上の場合、効果')[1])),
])
def test_other_viewport_conflicts_are_not_dropped(conflicting):
    assert not good_condition_requirements('ttmr', 19, rules(row(), conflicting))


def test_preparation_exports_partial_condition_but_cannot_authorize_battle():
    formation = Observation(Scene.CONFIRM, 'ttmr', 19, stars=0, formation_ready=True)
    result = prepare_battle('ttmr', 19, formation, stage_rules=rules())
    assert result.good_condition_requirements[0].minimum_turns == 4
    assert not result.automatic_battle_ready
    # Reading only the condition never claims to understand the whole rule.
    assert result.rule_declarations is not None and result.rule_declarations.unknown_sightings
    stale = prepare_battle('ttmr', 18, formation, stage_rules=rules())
    assert not stale.good_condition_requirements


@pytest.mark.parametrize('kind',['positive_turn_end_support','positive_debuff_recovery'])
def test_condition_without_confirmed_beneficial_support_is_not_a_planning_goal(kind):
    assert good_condition_requirements('ttmr',19,rules())
    assert not good_condition_planning_requirements('ttmr',19,rules())
    symbol = RuleSymbolEvidence('frame',kind,(5,5,30,30),.99)
    supported = replace(row(),good_condition_support=symbol)
    assert good_condition_planning_requirements('ttmr',19,rules(supported))
    for changes in ({'frame_id':'old'}, {'score':.969}, {'kind':'mental'}, {'rect':(105,5,30,30)}):
        bad = replace(supported,good_condition_support=replace(symbol,**changes))
        assert not good_condition_planning_requirements('ttmr',19,rules(bad))
