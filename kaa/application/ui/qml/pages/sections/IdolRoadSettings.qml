import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../../components/controls"
import "../../components/form"

FormGroupBox {
    id: root
    title: "偶像之路（实验性）"
    property var settingsCtrl
    property var errors: ({})
    property var characterModel: []
    readonly property var roadConfig: settingsCtrl?.config?.profile?.tasks?.idol_road ?? {}
    binder: roadBinder

    Component.onCompleted: {
        if (settingsCtrl && typeof settingsCtrl.idolRoadCharactersJson === "function")
            characterModel = JSON.parse(settingsCtrl.idolRoadCharactersJson())
    }
    FormBinder {
        id: roadBinder
        data: root.roadConfig
        prefix: "tasks.idol_road"
        errors: root.errors
        onCommitted: function(key, value) { root.settingsCtrl.setField("tasks.idol_road." + key, value) }
    }
    FormCheckBox { field: "enabled"; label: "启用偶像之路" }
    ColumnLayout {
        width: parent.width
        spacing: 8
        visible: root.roadConfig.enabled ?? false

        Label {
            Layout.fillWidth: true
            text: "至少一星后进入下一关，三星关卡直接跳过。完整编组保留，缺位可按设置自动编排。支持从其他角色地图／编成页开始；地图焦点丢失时最多重进一次。预览不会开战；手动模式需要人工打牌。星级或下一关无法确认时停止。请单独运行此任务。"
            wrapMode: Text.WordWrap
        }
        FieldRegistrar {
            startParent: parent
            binder: roadBinder
            field: "characters"
            label: "角色（按选择顺序）"
        }
        MultiSelect {
            objectName: "idolRoadCharacters"
            Layout.fillWidth: true
            label: "角色（按选择顺序）"
            model: root.characterModel
            selectedValues: root.roadConfig.characters ?? []
            onSelectionChanged: root.settingsCtrl.setListField("tasks.idol_road.characters", selectedValues)
        }
        FormSegmentedButton {
            objectName: "idolRoadMode"
            field: "mode"
            label: "运行模式"
            options: [
                {label: "预览：停在编成", value: "preview"},
                {label: "手动：人工打牌", value: "manual"},
                {label: "AI单场试跑：手毬集中／好调", value: "planner_trial"},
                {label: "AI爬塔：变量回合Sense（实验）", value: "auto"}
            ]
        }
        Label {
            Layout.fillWidth: true
            visible: root.roadConfig.mode === "auto"
            text: "实验模式支持推荐集中／好调的追加回合场地。开战后读取手牌；没有合适的牌时考虑SKIP。确认失败后在同编组重试，达到次数后随机替换一个Sub。画面、场地预算或回合结算无法确认时停止，该停止不累计挑战失败。暂不支持温存／全力／热情。"
            wrapMode: Text.WordWrap
        }
        Label {
            Layout.fillWidth: true
            visible: root.roadConfig.mode === "planner_trial"
            text: "AI试跑仅支持一名手毬、推荐效果为集中或好调的完整编组，按当前推荐选择策略，打一场后停止。暂不支持已识别的追加回合场地。尚未完整支持关卡和道具效果；不使用饮料、不自动重试，识别或决策失败时停止。"
            wrapMode: Text.WordWrap
        }
        FormSpinBox { field: "max_battles"; label: "最大挑战次数"; from: 1; to: 100 }
        FormCheckBox { objectName: "idolRoadAutoFill"; field: "auto_fill_missing_formation"; label: "缺位时自动编排（预览模式只报告）" }
        Label {
            Layout.fillWidth: true
            text: "自动编排会重新选择整组记忆。每层重新检查槽位，完整编组保留；重排后重新读取目标和道具。"
            wrapMode: Text.WordWrap
        }
        FormSpinBox { field: "timeout_seconds"; label: "总超时（秒）"; from: 1; to: 86400 }
        FormCheckBox {
            objectName: "idolRoadRetryFailed"
            field: "retry_failed_stage"
            label: "失败后重试同一关，多次失败再随机换Sub"
            visible: root.roadConfig.mode === "manual"
        }
        FormSpinBox {
            objectName: "idolRoadAttemptsPerLineup"
            field: "attempts_per_lineup"; label: "同编组尝试次数"; from: 2; to: 10
            visible: root.roadConfig.mode === "auto" || (root.roadConfig.mode === "manual" && (root.roadConfig.retry_failed_stage ?? false))
        }
        FormSpinBox {
            field: "max_lineup_changes"; label: "每关最多换Sub次数"; from: 0; to: 10
            visible: root.roadConfig.mode === "auto" || (root.roadConfig.mode === "manual" && (root.roadConfig.retry_failed_stage ?? false))
        }
        Label {
            Layout.fillWidth: true
            visible: root.roadConfig.mode === "manual" && (root.roadConfig.retry_failed_stage ?? false)
            text: "打牌仍由人工操作。确认失败后返回同关，达到尝试次数才随机替换一个Sub；更换后重新计次。总挑战次数和超时仍生效。"
            wrapMode: Text.WordWrap
        }
        FormCheckBox {
            field: "stop_on_failure"; label: "失败后停止任务（关闭则跳过当前角色）"
            visible: root.roadConfig.mode !== "auto" && !(root.roadConfig.mode === "manual" && (root.roadConfig.retry_failed_stage ?? false))
        }
    }
}
