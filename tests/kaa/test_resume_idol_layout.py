"""Resume-dialog crops must follow the detected dialog layout."""

import cv2

from kaa.config.const import HajimeScenario
from kaa.db.constants import ProduceExamEffectType
from kaa.tasks.produce.session import ProduceSession
from kaa.tasks.produce.shared.common import identify_resume_idol_card


def test_saving_layout_avoids_false_beginner_idol_match():
    image = cv2.imread(
        'kotonebot-resource/sprites/jp/produce/produce_resume_4.png'
    )
    assert image is not None

    # The old first-crop-wins path matched this unrelated empty area.
    assert identify_resume_idol_card(image, saving_layout=False) == (
        'i_card-skin-kllj-1-001'
    )
    assert identify_resume_idol_card(image, saving_layout=True) == (
        'i_card-skin-hrnm-3-002'
    )


def test_normal_layout_reads_its_own_card_position():
    image = cv2.imread(
        'kotonebot-resource/sprites/jp/produce/produce_resume.png'
    )
    assert image is not None

    assert identify_resume_idol_card(image, saving_layout=False) == (
        'i_card-skin-hski-3-000'
    )


def test_resumed_idol_id_determines_planner_archetype():
    full_power = ProduceSession(
        idol_card='i_card-skin-shro-3-008',
        scenario=HajimeScenario.MASTER,
        is_resumed=True,
    )
    false_match = ProduceSession(
        idol_card='i_card-skin-kllj-1-001',
        scenario=HajimeScenario.MASTER,
        is_resumed=True,
    )

    assert full_power.archetype == ProduceExamEffectType.ExamFullPower
    assert false_match.archetype == ProduceExamEffectType.ExamParameterBuff
