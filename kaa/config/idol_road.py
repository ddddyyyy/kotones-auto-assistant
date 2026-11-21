"""偶像之路Profile配置；默认预览，自动爬塔须显式选择实验模式。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IdolRoadConfig(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    enabled: bool = False
    """是否启用偶像之路任务，默认禁用。"""
    characters: list[str] = Field(default_factory=list)
    """按顺序挑战的角色 ID；使用游戏数据 ID，不绑定育成角色枚举。"""
    mode: Literal['preview', 'manual', 'planner_trial', 'auto'] = 'preview'
    """auto为变量回合Sense实验爬塔；其余保留预览、人工与固定回合单场试跑。"""
    max_battles: int = Field(default=20, ge=1, le=100)
    """单次任务挑战场数上限。"""
    timeout_seconds: int = Field(default=600, ge=1, le=86400)
    """单次任务总时限，包含人工战斗等待。"""
    stop_on_failure: bool = True
    """失败时停止整个任务；关闭时跳过当前角色，不重试同一关。"""
    retry_failed_stage: bool = False
    """手动模式启用同组多次尝试；auto总是按同组次数重试。"""
    attempts_per_lineup: int = Field(default=3, ge=2, le=10)
    """同一编组确认失败达到此次数后才更换，至少尝试两次。"""
    max_lineup_changes: int = Field(default=2, ge=0, le=10)
    """每个关卡随机更换Sub记忆的次数上限。"""
    auto_fill_missing_formation: bool = True
    """非预览模式发现明确空槽时使用游戏自动编排；会重选整组记忆。"""

    @field_validator('characters')
    @classmethod
    def validate_characters(cls, values: list[str]) -> list[str]:
        if any(len(value) != 4 or not value.isascii() or not value.isalpha()
               or not value.islower() for value in values):
            raise ValueError('角色 ID 必须为四位小写英文字母')
        if len(set(values)) != len(values):
            raise ValueError('角色不能重复')
        return values
