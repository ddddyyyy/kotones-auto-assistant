from dataclasses import replace
from kaa.config.idol_road import IdolRoadConfig
from kaa.idol_road.controller import IdolRoadController
from kaa.idol_road.models import Observation,Scene,Action
from kaa.idol_road.rules import VisibleStageRule,RuleViewport
from tests.kaa.test_idol_road_recurring_hp_cost import reads,timing,PARENT


def flow_and_row():
    flow=IdolRoadController(IdolRoadConfig(characters=['hski'],mode='auto'))
    flow._stage=1;flow._support_requested.add(('hski',1))
    weak=tuple(replace(read,lines=(replace(read.lines[0],confidence=.95),read.lines[1])) for read in reads())
    row=VisibleStageRule(PARENT,1,(),weak,recurring_hp_timing=timing())
    return flow,row


def observation(row):
    return Observation(Scene.SUPPORT_DETAILS,character_id='hski',stage=1,rule_viewport=RuleViewport('viewport',(row,)))


def test_two_rereads_are_bounded_and_do_not_repair_unknown_cost():
    flow,row=flow_and_row()
    for _ in range(2):
        decision=flow.step(observation(row))
        assert decision.action==Action.WAIT and decision.reason=='await_current_turn_start_cost_rule'
        assert flow.stage_rules is None
    assert flow.step(observation(row)).action==Action.SCROLL_SUPPORT
    assert flow.stage_rules.viewports[0].visible_rules==(row,)


def test_a_stronger_current_viewport_is_collected_without_merging_old_reads():
    flow,row=flow_and_row()
    assert flow.step(observation(row)).action==Action.WAIT
    current=replace(row,description_reads=reads())
    assert flow.step(observation(current)).action==Action.SCROLL_SUPPORT
    assert flow.stage_rules.viewports[0].visible_rules==(current,)
