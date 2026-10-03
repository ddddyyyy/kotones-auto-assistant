from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2

from kaa.tasks import R
from kaa.tasks.produce.new import controller as controller_module
from kaa.tasks.produce.new import page as page_module
from kaa.tasks.produce.new.consts import Scene, SceneType
from kaa.tasks.produce.shared import common as common_module
from kaa.tasks.produce.shared.common import ProduceInterrupt


_ROOT = Path(__file__).parents[2]
_OUTING = _ROOT / 'kotonebot-resource' / 'sprites' / 'jp' / 'in_produce'


def _match_score(path: Path) -> float:
    screen = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    template_path = R.InProduce.TextOutingStaminaMax.template.file_path
    assert template_path is not None
    template = cv2.imread(
        template_path,
        cv2.IMREAD_GRAYSCALE,
    )
    assert screen is not None and template is not None
    return float(cv2.minMaxLoc(cv2.matchTemplate(
        screen, template, cv2.TM_CCOEFF_NORMED,
    ))[1])


def test_outing_confirmation_template_matches_popup_not_normal_outing():
    assert _match_score(_OUTING / 'screenshot_outing_2.png') >= 0.8
    assert _match_score(_OUTING / 'screenshot_outing.png') < 0.8


def test_outing_popup_is_dispatched_as_a_specific_scene():
    with patch.object(
        page_module.R.InProduce.TextOutingStaminaMax,
        'exists',
        return_value=True,
    ):
        scene = page_module._SceneCheckMixin()._check_interrupt_dialogs()

    assert scene is not None
    assert scene.type is SceneType.OUTING_STAMINA_MAX


def test_outing_popup_scene_calls_confirmation_handler():
    controller = controller_module.ProduceController.__new__(
        controller_module.ProduceController
    )
    fake_device = SimpleNamespace(screenshot=Mock())
    with (
        patch.object(controller_module, 'device', new=fake_device),
        patch.object(ProduceInterrupt, '_check_outing_stamina_max') as handle,
    ):
        assert controller._handle_interrupts(Scene(SceneType.OUTING_STAMINA_MAX))

    fake_device.screenshot.assert_called_once_with()
    handle.assert_called_once_with(fake_device.screenshot.return_value)


def test_outing_confirmation_handler_clicks_continue_when_popup_persists():
    body = SimpleNamespace(exists=Mock(return_value=True))
    continue_button = SimpleNamespace(try_click=Mock(return_value=True))
    fake_r = SimpleNamespace(
        InProduce=SimpleNamespace(TextOutingStaminaMax=body),
        Common=SimpleNamespace(ButtonSelect2=continue_button),
    )
    fake_device = SimpleNamespace(screenshot=Mock())
    with (
        patch.object(common_module, 'R', new=fake_r),
        patch.object(common_module, 'device', new=fake_device),
    ):
        result = ProduceInterrupt._check_outing_stamina_max(
            fake_device.screenshot.return_value
        )

    assert result == 'OutingStaminaMax'
    assert body.exists.call_count == 2
    continue_button.try_click.assert_called_once_with()
