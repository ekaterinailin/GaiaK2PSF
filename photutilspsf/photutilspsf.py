from lightkurve import KeplerTargetPixelFile
from astropy.io import fits
import numpy as np
import pandas as pd

def select_members_on_tpf(tpf, members):
    '''
    Use the TargetPixelFile WCS solution to map
    sky coordinates to 0-based pixel coordinates 
    '''
    radec = np.array(list(zip(members.ra,members.dec)), np.float_)
    cx, cy = tpf.wcs.wcs_world2pix(radec, 0).T #0 indicates we start pixel counting at 0
    members['xcentroid'] = cx
    members['ycentroid'] = cy
    return members.loc[(members.xcentroid >= 0) & 
                       (members.xcentroid <= tpf.shape[1]-1) &
                       (members.ycentroid >= 0) &
                       (members.ycentroid <= tpf.shape[2]-1), :]

def read_fits_table(path):
    '''
    Reads a fits table to pandas DataFrame.
    NO TESTS.
    
    Parameters:
    ------------
    path : str
        Path to table
    
    Return:
    -------
    DataFrame
    '''
    data = fits.open(path)[1].data
    return pd.DataFrame({'ra':data['RAJ2000'].byteswap().newbyteorder(),
                  'dec':data['DEJ2000'].byteswap().newbyteorder(),
                  'gaia_id':data['source_id'].byteswap().newbyteorder().astype(str),
                         'EPIC':data['EPIC'].byteswap().newbyteorder().astype(int),
                        'pmem':data['probability_DANCe'].byteswap().newbyteorder()})