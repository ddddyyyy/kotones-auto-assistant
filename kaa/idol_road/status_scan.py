"""状态详情的局部分屏键；只确认已支持标题的位置变化，不证明列表完整。"""
from .status_details import BattleStatusDetails,confirm_status_row


def status_viewport_key(details: BattleStatusDetails) -> str | None:
    if (not details.frame_id or not details.rows
        or any(row.evidence.frame_id!=details.frame_id or confirm_status_row(row.evidence)!=row for row in details.rows)):
        return None
    # 不让新帧ID／OCR置信度微动冒充滚动；分屏仍各自保留，不合并效果实体。
    return repr(tuple(sorted(row.evidence.title_rect for row in details.rows)))
