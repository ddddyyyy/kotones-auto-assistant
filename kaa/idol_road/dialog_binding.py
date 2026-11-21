"""只读编成弹窗的请求来源；不从模糊背景补读当前身份。"""


class FormationDialogBinding:
    def __init__(self) -> None:
        self.identity: tuple[str, int] | None = None
        self._closing = False

    def open(self, character_id: str, stage: int) -> None:
        self.identity = (character_id, stage)
        self._closing = False

    def close(self) -> None:
        self._closing = True

    def clear(self) -> None:
        self.identity = None
        self._closing = False

    def on_formation(self, character_id: str, stage: int) -> None:
        if self._closing or self.identity != (character_id, stage):
            self.clear()
