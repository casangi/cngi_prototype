#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
this module will be included in the api
"""

import numpy as np
from numba import jit

########################
def fit_gaussian(xds,dv='PSF',beam_set_name='RESTORE_PARMS',npix_window=[9,9],sampling=[9,9],cutoff=0.35):
    """
    Fit one or more elliptical gaussian components on an image region

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image xarray dataset
    dv : str
        image data variable to fit. Default is 'PSF'
        
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    import xarray as xr
    import matplotlib.pyplot as plt
    import dask.array as da
    from scipy.interpolate import interpn
    
    _xds = xds.copy(deep=True)
    
    sampling = np.array(sampling)
    npix_window = np.array(npix_window)
    delta = np.array([_xds[dv].l[1] - _xds[dv].l[0], _xds[dv].m[1] - _xds[dv].m[0]])*3600*180/np.pi
    chunks = _xds[dv].data.chunks[2:] + (3,)
    
    ellipse_parms = da.map_blocks(casa_fit, _xds[dv].data, npix_window, sampling,cutoff,
                                  delta, dtype=np.double, drop_axis=[0,1], new_axis=[3], chunks=chunks)
        
    _xds[beam_set_name] = xr.DataArray(ellipse_parms, dims=['time','chan','pol','elps_index'])
    
    return _xds
    


################################################################
########################### casa_fit ###########################
@jit(nopython=True,cache=True,nogil=True)
def gaussian2D(width_x, width_y, rotation, sampling):
    rotation = 90-rotation
    rotation = np.deg2rad(rotation)
    #x, y = np.indices((sampling[0]*2+1,sampling[1]*2+1))
    x_size = sampling[0]*2+1
    y_size = sampling[1]*2+1
    
    x = np.repeat(np.arange(x_size),y_size).reshape((x_size,y_size))
    y = np.repeat(np.arange(y_size),x_size).reshape((y_size,x_size)).T
    
    x = x - sampling[0]
    y = y - sampling[1]
    xp = x * np.cos(rotation) - y * np.sin(rotation)
    yp = x * np.sin(rotation) + y * np.cos(rotation)
    g = 1.*np.exp(-(((xp)/width_x)**2+((yp)/width_y)**2)/2.)
    return g

def beam_chi2(params, psf, sampling):
    psf_ravel = psf[~np.isnan(psf)]
    width_x, width_y, rotation = params
    gaussian = gaussian2D(width_x, width_y, rotation, sampling)[~np.isnan(psf)]
    chi2 = np.sum((gaussian-psf_ravel)**2)
    return chi2

def casa_fit(img_to_fit,npix_window,sampling,cutoff,delta):
    import numpy.linalg as linalg
    import scipy.optimize as optimize
    from scipy.interpolate import interpn
    
    #ellipse_parms = np.zeros(img_to_fit.shape[2:4] + (3,),dtype=numba.double)
    ellipse_parms = np.zeros(img_to_fit.shape[2:5] + (3,))
    
    img_size = np.array(img_to_fit.shape[0:2])
    img_center = img_size//2
    start_window = img_center - npix_window//2
    end_window = img_center + npix_window//2 + 1
    img_to_fit = img_to_fit[start_window[0]:end_window[0],start_window[1]:end_window[1],:,:]
    
    d0 = np.arange(0, npix_window[0])*np.abs(delta[0])
    d1 = np.arange(0, npix_window[1])*np.abs(delta[1])
    interp_d0 = np.linspace(0, npix_window[0]-1, sampling[0])*np.abs(delta[0])
    interp_d1 = np.linspace(0, npix_window[1]-1, sampling[1])*np.abs(delta[1])
    xp, yp = np.meshgrid(interp_d0, interp_d1,indexing='ij')
    points = np.vstack((np.ravel(xp), np.ravel(yp))).T
    
    #img_to_fit = np.reshape(interpn((d0,d1),img_to_fit[:,:,0,0],points,method="splinef2d"),[sampling[1],sampling[0]]).T
    
    for time in range(img_to_fit.shape[2]):
        for chan in range(img_to_fit.shape[3]):
            for pol in range(img_to_fit.shape[4]):
            
                interp_img_to_fit = np.reshape(interpn((d0,d1),img_to_fit[:,:,time,chan,pol],points,method="splinef2d"),[sampling[1],sampling[0]]).T
                
                interp_img_to_fit[interp_img_to_fit<cutoff] = np.nan

                #print(interp_img_to_fit.shape)
                # Fit a gaussian to the thresholded points
                p0 = [2.5, 2.5, 0.]
                res = optimize.minimize(beam_chi2, p0, args=(interp_img_to_fit, sampling//2))

                # convert to useful units
                phi = res.x[2] - 90.
                if phi < -90.:
                    phi += 180.

                ellipse_parms[time,chan,pol,0] = np.max(np.abs(res.x[0:2]))*np.abs(delta[0])*2.355/(sampling[0]/npix_window[0])
                ellipse_parms[time,chan,pol,1] = np.min(np.abs(res.x[0:2]))*np.abs(delta[1])*2.355/(sampling[1]/npix_window[1])
                ellipse_parms[time,chan,pol,2] = -phi
    return ellipse_parms
