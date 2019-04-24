import matplotlib.pyplot as plt
import numpy as np

from photutils import aperture_photometry, CircularAperture

from astropy.visualization import simple_norm

def plot_psfmodel(psfmodel):
    '''
    Display the PSF model.
    
    Parameters:
    -----------
    psfmodel: EPSFModel or the like
    '''
    norm = simple_norm(psfmodel.data, 'log', percent=99.)
    plt.imshow(psfmodel.data, norm=norm, origin='lower', cmap='viridis')
    plt.colorbar()
    return

def plot_finder_results(photometry, image):
    '''
    Circle 4 pixel radii around found targets.
    
    Parameters:
    ------------
    photometry : photutils.psf Photometry object
    
    image : ndarray
        photometric image
    '''
    sources = photometry.finder(image)
    positions = (sources['xcentroid'], sources['ycentroid'])    
    apertures = CircularAperture(positions, r=4.)      
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray_r', origin='lower')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    return

def plot_psf_photometry_results(image, residual_image):
    '''
    Plot image, residuals and model.
    
    Parameters:
    ------------
    
    '''
    plt.figure(figsize=(20,7))

    plt.subplot(1, 3, 1,)

    plt.imshow(image, cmap='viridis', aspect=1, interpolation='nearest',)
    plt.ylim(0,image.shape[1]-1)
    plt.title('Original Data')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    plt.subplot(1 ,3, 2)
    #plt.scatter(result_tab['x_fit'],result_tab['y_fit'],c='r', marker='x')
    plt.imshow(image-residual_image, cmap='viridis', aspect=1,
               interpolation='nearest', origin='lower')
    plt.ylim(0,image.shape[1]-1)
    plt.title('Model Image')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    plt.subplot(1 ,3, 3)
    plt.imshow(residual_image, cmap='viridis', aspect=1,
               interpolation='nearest', origin='lower')
    plt.title('Residual Image')
    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
    plt.ylim(0,image.shape[1]-1)
    return