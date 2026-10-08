import json
from unittest.mock import Mock

import pytest
from kotonebot.client.host.mumu12_host import Mumu12V5Host, Mumu12V5Instance
from kotonebot.errors import UserFriendlyError

from kaa.config.base_config import BackendConfig, MuMu12V5Device
from kaa.config.schema import KaaConfig
from kaa.main.mumu_adb import resolve_mumu_adb_address


def instance():
    return Mumu12V5Instance(id='0', name='game', adb_ip='10.0.2.15',
                           adb_port=16384, adb_name='127.0.0.1:16384')


@pytest.mark.parametrize('host,port', [('192.168.31.8', 5555), ('127.0.0.1', 16384)])
def test_connect_endpoint_replaces_info_and_stale_serial(monkeypatch, host, port):
    invoke = Mock(return_value=json.dumps({'adb_host': host, 'adb_port': port, 'cmd_output': 'connected'}))
    monkeypatch.setattr(Mumu12V5Host, '_invoke_manager', invoke)
    target = instance()
    resolve_mumu_adb_address(target)
    assert (target.adb_ip, target.adb_port, target.adb_name) == (host, port, None)
    invoke.assert_called_once_with(['adb', '-v', '0', '-c', 'connect'])


@pytest.mark.parametrize('output', [
    'not json', '[]', '{}',
    '{"adb_host":"", "adb_port":5555}',
    '{"adb_host":"bad host", "adb_port":5555}',
    '{"adb_host":"127.0.0.1", "adb_port":true}',
    '{"adb_host":"127.0.0.1", "adb_port":0}',
    '{"adb_host":"127.0.0.1", "adb_port":65536}',
    '{"adb_host":"127.0.0.1", "adb_port":"5555"}',
])
def test_invalid_endpoint_fails_without_partial_binding(monkeypatch, output):
    monkeypatch.setattr(Mumu12V5Host, '_invoke_manager', Mock(return_value=output))
    target = instance()
    with pytest.raises(UserFriendlyError):
        resolve_mumu_adb_address(target)
    assert (target.adb_ip, target.adb_port, target.adb_name) == ('10.0.2.15', 16384, '127.0.0.1:16384')


@pytest.mark.parametrize('recipe', ['nemu_ipc', 'adb', 'uiautomator2'])
def test_factory_binds_endpoint_before_creating_device(monkeypatch, recipe):
    from kaa.main.kaa import KaaDeviceFactory

    monkeypatch.setattr(Mumu12V5Host, '_invoke_manager', Mock(return_value=json.dumps(
        {'adb_host': '192.168.31.8', 'adb_port': 5555})))
    target = instance()
    marker = object()

    def create_device(actual_recipe, host_config):
        assert actual_recipe == recipe
        assert (target.adb_ip, target.adb_port, target.adb_name) == ('192.168.31.8', 5555, None)
        return marker

    monkeypatch.setattr(target, 'create_device', create_device)
    config = KaaConfig(backend=BackendConfig(lifecycle=MuMu12V5Device(type='mumu12v5', instance_id='0'), screenshot_impl=recipe))
    assert KaaDeviceFactory()._create_device_impl(target, config) is marker
