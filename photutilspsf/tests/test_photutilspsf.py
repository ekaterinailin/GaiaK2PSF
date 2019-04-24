import pandas as pd
import numpy as np
from lightkurve import KeplerTargetPixelFile

from ..photutilspsf import (select_members_on_tpf,
                            get_gaia_astrometry,
                            merge_olivares_members_with_gaia_coordinates,
                           )

def test_merge_olivares_members_with_gaia_coordinates():
    fake_gaia = pd.DataFrame({
     'dec': {0: -15.5948144537334, 1:-15.9},
     'dec_error': {0: 0.058149423141046395,1:0.05},
     'ra': {0: 286.2414094445609,1:286.2},
     'ra_dec_corr': {0: -0.30148670077323914, 1:-0.2},
     'ra_error': {0: 0.05757310655164684, 1:0.05},
     'source_id': {0:4089095052377089280,1:625453654702751878},
     'visibility_periods_used': {0: 8, 1:5}})

    fake_members = pd.DataFrame({'EPIC': {0: -2147483648,
      1: -2147483648,
      2: -2147483648,
      3: 220070265,
      4: 219856935,
      5: 219635263,
      6: 218118532},
     'dec': {0: -13.8008297875,
      1: -13.6497964942,
      2: -14.0692542115,
      3: -14.7439040201,
      4: -15.5948140305,
      5: -16.2444658246,
      6: -19.3783871022},
     'gaia_id': {0: '4197452198170458624',
      1: '4197424194982998144',
      2: '4184845605842332928',
      3: '4101219779417245696',
      4: '4089095052377089280',
      5: '4088801723287036800',
      6: '4083839867817876352'},
     'pmem': {0: 0.736601194479747,
      1: 0.951864566515844,
      2: 0.932025480041679,
      3: 0.799267305859579,
      4: 0.872384297670303,
      5: 0.98561588347909,
      6: 0.79549065315158},
     'ra': {0: 286.577835918,
      1: 287.164554916,
      2: 291.326538493,
      3: 285.240526061,
      4: 286.241408758,
      5: 286.347896457,
      6: 289.223175241}})

    res = merge_olivares_members_with_gaia_coordinates(fake_members, fake_gaia)
    assert res.shape[0] == fake_gaia.shape[0] + fake_members.shape[0]
    assert res[res.EPIC == 0].shape[0] == 2
    assert res[res.pmem == 0].shape[0] == 2
    assert 'gaia_id' in res.columns
    
def test_get_gaia_astrometry():
    res = get_gaia_astrometry((154.89888734446,19.86981583640),1/3600.)#AD Leo
    assert res.source_id.iloc[0] == 625453654702751872
    assert isinstance(res,pd.DataFrame)

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