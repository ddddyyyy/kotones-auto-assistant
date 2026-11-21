from types import SimpleNamespace
from typing import Any, cast

import pytest

from kaa.db.constants import ProduceExamEffectType as Effect
from kaa.tasks.produce.new.play_cards.planner import strategy as module
from kaa.tasks.produce.new.play_cards.planner.evaluator import CardEvaluation, EvaluationConfidence
from kaa.tasks.produce.new.play_cards.planner.state import BattleState


def effect(kind, ident='root', chains=()):
    return SimpleNamespace(_id=ident,effect_type=kind,chain_effect_ids=chains,
                           effect_value1=1,effect_value2=0,effect_count=1,effect_turn=0)


def hand(ident, *effects):
    return SimpleNamespace(card=SimpleNamespace(_id=ident,name=ident,stamina=0,force_stamina=0,
        cost_type=None,cost_value=0,play_move_position_type='ProduceCardMovePositionType_Grave',
        play_effects=[SimpleNamespace(produce_exam_effect=e,_produce_exam_trigger_id='') for e in effects]))


class Evaluator:
    def evaluate(self, card, state):
        return CardEvaluation(card._id,10,EvaluationConfidence.MEDIUM,'first-score')


def ranked(first, *, depth=2):
    second=hand('second')
    evaluator=Evaluator()
    state=BattleState(3,50,8,playable_count=3)
    candidates=[(h,evaluator.evaluate(h.card,state)) for h in (first,second)]
    return module.PlannerStrategy(evaluator)._add_shallow_lookahead(cast(Any,candidates),state,depth=depth)


@pytest.mark.parametrize('depth',[2,3])
def test_redraw_keeps_first_score_without_old_hand_followup(depth):
    first=hand('redraw',effect(Effect.ExamHandGraveCountCardDraw))
    result=ranked(first,depth=depth)
    assert result[0][1].score==10 and result[0][1].reason=='first-score'


@pytest.mark.parametrize('kind',[Effect.ExamCardDraw,Effect.ExamPlayableValueAdd,Effect.ExamLessonBuff])
def test_ordinary_draw_and_setup_keep_existing_visible_followup(kind):
    assert ranked(hand('setup',effect(kind)))[0][1].score==19


def test_chain_redraw_cannot_keep_old_visible_hand(monkeypatch):
    monkeypatch.setattr(module,'load_exam_effects',lambda ids:{'child':effect(Effect.ExamHandGraveCountCardDraw,'child')})
    first=hand('chain',effect(Effect.ExamLessonBuff,chains=('child',)))
    assert ranked(first)[0][1].score==10


def test_known_plain_chain_does_not_disable_followup(monkeypatch):
    monkeypatch.setattr(module,'load_exam_effects',lambda ids:{'child':effect(Effect.ExamCardDraw,'child')})
    first=hand('chain',effect(Effect.ExamLessonBuff,chains=('child',)))
    assert ranked(first)[0][1].score==19


@pytest.mark.parametrize('case',['missing','cycle','depth','too_wide'])
def test_uncertain_chain_has_finite_search_and_no_old_hand_bonus(monkeypatch,case):
    calls=[]
    def load(ids):
        calls.append(ids)
        if case=='missing':return {}
        if case=='cycle':return {'child':effect(Effect.ExamLessonBuff,'child',('root',))}
        return {ident:effect(Effect.ExamLessonBuff,ident,(str(int(ident)+1),)) for ident in ids}
    monkeypatch.setattr(module,'load_exam_effects',load)
    refs=tuple(str(i) for i in range(33)) if case=='too_wide' else ('0',) if case=='depth' else ('child',)
    first=hand('uncertain',effect(Effect.ExamLessonBuff,chains=refs))
    assert ranked(first)[0][1].score==10
    assert len(calls)<=9


def test_unknown_trigger_redraw_is_not_assumed_inactive():
    first=hand('conditional',effect(Effect.ExamHandGraveCountCardDraw))
    first.card.play_effects[0]._produce_exam_trigger_id='unknown-event'
    assert ranked(first)[0][1].score==10
