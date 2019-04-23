import pandas as pd
import numpy as np
from lightkurve import KeplerTargetPixelFile

from ..photutilspsf import select_members_on_tpf


def test_select_members_on_tpf():
    members = pd.read_csv('photutilspsf/tests/testmembers.csv')
    tpf = KeplerTargetPixelFile('photutilspsf/tests/ktwo200062544-c07_0256.fits')
    rselect = select_members_on_tpf(tpf, members)
    assert 'xcentroid' in rselect.columns
    assert 'ycentroid' in rselect.columns
    assert (rselect['xcentroid'] >= 0).all()
    assert (rselect['ycentroid'] >= 0).all()
    assert (rselect['xcentroid'] <= tpf.shape[1]-1).all()
    assert (rselect['ycentroid'] <= tpf.shape[2]-1).all()
    assert rselect.shape[0] == 1