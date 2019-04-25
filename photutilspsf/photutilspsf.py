import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from photutils.psf import (IntegratedGaussianPRF,
                           DAOGroup,
                           IterativelySubtractedPSFPhotometry,
                           BasicPSFPhotometry)

from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.detection import IRAFStarFinder

from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm

from lightkurve import KeplerTargetPixelFile

def do_photometry_with_fixed_centroids(photometry, memberspath, TPF, index, **kwargs):
    '''
    Do photometry with fixed centroids.
    
    Parameters:
    ------------
    
    kwargs : dict
        Keyword arguments to  pass to 
        wrapper_supplement_olivares_with_gaia_coordinates_on_tpf
    '''
    image = TPF.flux[index,:,:]
    rselect = wrapper_supplement_olivares_with_gaia_coordinates_on_tpf(TPF,
                                                                       memberspath,
                                                                       **kwargs)

    pos = Table(names=['EPIC','x_0', 'y_0'], data=[rselect['EPIC'],
                                                   rselect['xcentroid'],
                                                   rselect['ycentroid']])
    result_table =  photometry(image=image, init_guesses=pos)
    del image
    return result_table

def define_basic_photometry_with_fixed_centroids(sigma_psf=0.5, group_psf=1.5,
                                                 fitshape=(11,11), psf_model=None):
    '''
    Define photometry with centroid given by members and Gaia
    centroids.
    
    Parameters:
    ------------
    sigma_psf : float
        1 sigma size in pixel
    group_psf : float
        maximum distance between stars for grouping
    fitshape : tuple of float
        Rectangular shape around the center of a star
        which will be used to collect the data to do
        the fitting. 
    psf_model :  astropy.modeling.Fittable2DModel instance
        PSF or PRF model to fit the data. Could be one of 
        the models in this package like DiscretePRF, 
        IntegratedGaussianPRF, or any other suitable 
        2D model.
    Returns:
    ---------
    photutils.psf.BasicPSFPhotometry object
    '''
    daogroup = DAOGroup(group_psf *
                        sigma_psf * 
                        gaussian_sigma_to_fwhm)

    bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    if psf_model is None:
        psf_model = IntegratedGaussianPRF(sigma=sigma_psf)

    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    
    return BasicPSFPhotometry(group_maker=daogroup,
                              bkg_estimator=bkg,
                              psf_model=psf_model,
                              fitter=LevMarLSQFitter(),
                              fitshape=fitshape,
                              aperture_radius=5)

def define_iterative_photometry_with_finder(image, sigma_psf=0.5, group_psf=1.5,
                                            grouper=None,bkg=MMMBackground(),
                                            fitter=LevMarLSQFitter(),psf_model=None,
                                            fitshape=11,niters=10, finder=None):
    '''
    Defines a photutils photometry routine with
    an iterative source finder.
    
    Parameters:
    -----------
    image : ndarray
        photometric image
    sigma_psf :
    
    Returns:
    ---------
    IterativelySubtractedPSFPhotometry object
    '''
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image)
    if grouper is None:
        grouper = DAOGroup(group_psf *
                           sigma_psf * 
                           gaussian_sigma_to_fwhm)

    if psf_model is None:
        psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    if finder is None:
        finder = IRAFStarFinder(threshold=3*std,
                          fwhm=2*sigma_psf*gaussian_sigma_to_fwhm,
                          minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                          sharplo=0.0, sharphi=8., brightest=10)
    
    return IterativelySubtractedPSFPhotometry(finder=finder,
                                                group_maker=grouper,
                                                bkg_estimator=bkg,
                                                psf_model=psf_model,
                                                fitter=fitter,
                                                niters=niters,
                                                fitshape=fitshape,
                                                aperture_radius=3)

def wrapper_supplement_olivares_with_gaia_coordinates_on_tpf(TPF, path, get_saved=False, brightest=1.):
    '''
    Supplement olivares table with gaia coordinates.
    Save the intermediate resulst
    
    Parameters:
    ------------
    TPF : KeplerTargetPixelFile
    
    path : str
        path to Olivares members table
    get_saved : bool
        if True will fetch a stored table
     
    Return:
    -------
    DataFrame with Olivares members and Gaia solution
    ''' 
    EPIC = TPF.targetid
    C = TPF.campaign
    datapath = 'data/{}_{:02d}.csv'.format(EPIC, C)
    if get_saved==True:
        return pd.read_csv(datapath, skiprows=4)
    elif get_saved==False:
        members = read_fits_table(path)
        center, radius = find_center_and_radius(TPF)
        others = get_gaia_astrometry(center, radius)
        bright_part = (others.sort_values(by='phot_g_mean_flux')
                             .iloc[:int(np.rint(brightest * others.shape[0])),:])
        ontpf = merge_olivares_members_with_gaia_coordinates(members, bright_part)
        res = select_members_on_tpf(TPF, ontpf)
       
        #save intermediate to file
        with open(datapath, 'w') as f:
            f.write('# This is a concatenated table of Olivares+2019\n'
                    '# and Gaia DR2 coordinates on the TargetPixelFile\n'
                    '# of EPIC {} in campaign {}\n#\n'.format(EPIC, C))
            res.to_csv(f,index=False)
        return res

def merge_olivares_members_with_gaia_coordinates(members, gaia):
    '''
    Take Olivares' membership table and 
    concatenate with Gaia query results.
    
    Parameters:
    ------------
    members : DataFrame
        Olivares membership table
    gaia : DataFrame
        Gaia database query results
    '''
    gaia = gaia[['source_id','ra','dec']]
    gaia = gaia.rename(index=str, columns={'source_id':'gaia_id'})
    gaia['EPIC'] = 0
    gaia['pmem'] = 0.
    ontpf = pd.concat((members,gaia),ignore_index=True, sort=True)
    assert ontpf.shape[0] == members.shape[0] + gaia.shape[0]
    return ontpf

def find_center_and_radius(tpf):
    '''
    Find center and radius of a given
    KeplerTargetPixelFile.
    NOT TESTED
    
    Parameters:
    -----------
    tpf : KeplerTargetPixelFile
    
    Returns:
    ---------
    (RA, Dec), radius in deg
    '''
    center = (np.array(tpf.flux[0,:,:].shape)-1)/2
    lo, la = tpf.wcs.wcs_pix2world(*center, 0) #0 indicates we start pixel counting at 0
    lom, lam = tpf.wcs.wcs_pix2world(0, 0, 0)
    radius = np.sqrt((lom - lo)**2 + (lam - la)**2)
    return (lo, la), radius

def get_gaia_astrometry(center, radius):
    '''
    Query Gaia archive in radius (deg) around
    a sky coordinate (deg, deg).
    
    Parameters:
    ------------
    center : tuple of floats
        RA, Dec
    radius : float
        search radius
    
    Returns:
    --------
    DataFrame with Gaia solution within radius
    '''
    from astroquery.gaia import Gaia
    coord = SkyCoord(ra=center[0], dec=center[1], unit=(u.degree, u.degree), frame='icrs')
    r = Gaia.query_object_async(coordinate=coord, radius=radius*u.degree)
    return r.to_pandas()

def select_members_on_tpf(TPF, members):
    '''
    Use the TargetPixelFile WCS solution to map
    sky coordinates to 0-based pixel coordinates
    
    Parameters:
    ------------
    TPF : KeplerTargetPixelFile
    
    members : DataFrame
        table with coordinates that shall
        be checked for location on TPF
        
    Return:
    --------
    Subset of the members table where the entries are located
    on the TPF. Added x any centroid columns.
    '''
    radec = np.array(list(zip(members.ra,members.dec)), np.float_)
    cx, cy = TPF.wcs.wcs_world2pix(radec, 0).T #0 indicates we start pixel counting at 0
    members['xcentroid'] = cx
    members['ycentroid'] = cy
    return members.loc[(members.xcentroid >= 0) & 
                       (members.xcentroid <= TPF.shape[1]-1) &
                       (members.ycentroid >= 0) &
                       (members.ycentroid <= TPF.shape[2]-1), :]

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