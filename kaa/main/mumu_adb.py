"""Resolve the actual ADB endpoint reported by MuMu's connection command."""
import json
from typing import TYPE_CHECKING

from kotonebot.errors import UserFriendlyError

if TYPE_CHECKING:
    from kotonebot.client.host import Mumu12Instance


def resolve_mumu_adb_address(instance: 'Mumu12Instance') -> None:
    """Bridge mode can expose a different endpoint than the info command."""
    from kotonebot.client.host.mumu12_host import Mumu12V5Host

    output = Mumu12V5Host._invoke_manager(['adb', '-v', instance.id, '-c', 'connect'])
    try:
        address = json.loads(output)
        if not isinstance(address, dict):
            raise ValueError('Expected an ADB address object')
        host, port = address.get('adb_host'), address.get('adb_port')
        if not isinstance(host, str) or not host or any(c.isspace() for c in host):
            raise ValueError('Invalid ADB host')
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid ADB port')
    except (ValueError, TypeError) as error:
        raise UserFriendlyError('无法读取 MuMu 实例的实际 ADB 地址，请检查模拟器连接状态。') from error

    # Publish both fields only after validation; a stale serial must not select
    # another device when the endpoint changed after switching network modes.
    instance.adb_ip = host
    instance.adb_port = port
    instance.adb_name = None
