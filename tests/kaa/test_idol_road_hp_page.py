from types import SimpleNamespace
import numpy as np
import pytest
from kaa.idol_road import hp_page


def install(monkeypatch,text,reads):
    monkeypatch.setattr(hp_page,'digit_classifier',lambda:SimpleNamespace(classification=lambda _:text))
    stream=iter(reads)
    monkeypatch.setattr(hp_page,'en',lambda:SimpleNamespace(ocr=lambda _:next(stream)))


def rows(text='48',confidence=.99):
    return [SimpleNamespace(text=text,confidence=confidence)]


@pytest.mark.parametrize('text',['0','48','248'])
def test_accepts_complete_same_crop_consensus_without_prefix_repair(monkeypatch,text):
    install(monkeypatch,text,[rows(text),rows(text),rows(text)])
    assert hp_page.read_masked_hp(np.zeros((1280,720,3),dtype='uint8'))==int(text)


@pytest.mark.parametrize('reads',[
    [rows('49'),rows(),rows()],
    [rows(),rows('49'),rows()],
    [rows(),rows(),rows('49')],
    [[],[],rows()],
    [rows(),rows(),rows(confidence=.97)],
    [rows()+rows('49'),rows(),rows()],
])
def test_unknown_or_conflicting_evidence_cannot_release_hp(monkeypatch,reads):
    install(monkeypatch,'48',reads)
    assert hp_page.read_masked_hp(np.zeros((1280,720,3),dtype='uint8')) is None


@pytest.mark.parametrize('text',['x48','048','48 ','','48/53'])
def test_classifier_text_is_not_trimmed_or_repaired(monkeypatch,text):
    install(monkeypatch,text,[])
    assert hp_page.read_masked_hp(np.zeros((1280,720,3),dtype='uint8')) is None
