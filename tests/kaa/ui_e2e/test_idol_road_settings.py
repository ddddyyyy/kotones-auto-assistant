import json

from PySide6.QtCore import QMetaObject, Qt, Slot,Property,QObject,Signal
from PySide6.QtQml import QQmlApplicationEngine

from .conftest import FakeSettingsController, find, load_qml


class RoadSettings(FakeSettingsController):
    @Slot(result=str)
    def idolRoadCharactersJson(self) -> str:
        return json.dumps([{'text': '咲季', 'value': 'hski'}, {'text': '美鈴', 'value': 'hmsz'}])


class InitialRoadSettings(QObject):
    configChanged=Signal()

    def __init__(self,config):
        super().__init__()
        self._config=config

    config=Property('QVariantMap',lambda self:self._config,notify=configChanged)

    @Slot(result=str)
    def idolRoadCharactersJson(self):
        return json.dumps([{'text':'手毬','value':'ttmr'}])

    @Slot(str,'QVariant')
    def setField(self,path,value):
        pass

    @Slot(str,'QVariant')
    def setListField(self,path,value):
        pass


def test_missing_slot_auto_setting_loads_from_profile(qml_engine: QQmlApplicationEngine):
    settings=InitialRoadSettings(config={'profile':{'tasks':{'idol_road':{
        'enabled':True,'mode':'preview','auto_fill_missing_formation':True}}}})
    settings.setParent(qml_engine)
    page=load_qml(qml_engine,'pages/sections/IdolRoadSettings.qml',{'settingsCtrl':settings})
    assert find(page,'idolRoadAutoFill').property('_val') is True


def test_bounded_retry_settings_load_from_profile(qml_engine: QQmlApplicationEngine):
    settings=InitialRoadSettings(config={'profile':{'tasks':{'idol_road':{
        'enabled':True,'mode':'manual','retry_failed_stage':True,
        'attempts_per_lineup':3,'max_lineup_changes':2}}}})
    settings.setParent(qml_engine)
    page=load_qml(qml_engine,'pages/sections/IdolRoadSettings.qml',{'settingsCtrl':settings})
    assert find(page,'idolRoadRetryFailed').property('_val') is True
    assert find(page,'idolRoadAttemptsPerLineup').property('_val')==3


def test_single_ai_trial_mode_is_explicit_and_loads_from_profile(qml_engine: QQmlApplicationEngine):
    settings=InitialRoadSettings(config={'profile':{'tasks':{'idol_road':{
        'enabled':True,'characters':['ttmr'],'mode':'planner_trial',
    }}}})
    settings.setParent(qml_engine)
    page=load_qml(qml_engine,'pages/sections/IdolRoadSettings.qml',{'settingsCtrl':settings})
    mode=find(page,'idolRoadMode')
    options=mode.property('options')
    if hasattr(options,'toVariant'):
        options=options.toVariant()
    assert [option['value'] for option in options]==['preview','manual','planner_trial','auto']
    assert mode.property('_val')=='planner_trial'


def test_road_selection_writes_ordered_profile_list(qml_engine: QQmlApplicationEngine):
    settings = RoadSettings(config={'profile': {'tasks': {'idol_road': {
        'enabled': True, 'characters': ['hski'], 'mode': 'preview',
    }}}})
    page = load_qml(qml_engine, 'pages/sections/IdolRoadSettings.qml', {'settingsCtrl': settings})
    selector = find(page, 'idolRoadCharacters')
    selector.setProperty('selectedValues', ['hmsz', 'hski'])
    assert QMetaObject.invokeMethod(selector, 'selectionChanged', Qt.ConnectionType.DirectConnection)
    values = settings.fields['tasks.idol_road.characters']
    if hasattr(values, 'toVariant'):
        values = values.toVariant()
    assert values == ['hmsz', 'hski']
    assert settings.dirty
