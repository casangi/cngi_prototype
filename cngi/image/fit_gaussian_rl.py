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

########################
def fit_gaussian_rl(xds,dv='PSF',beam_set_name='RESTORE_PARMS',fit_method='rm_fit',npix_window=[21,21],sampling=[401,401],cutoff=0.5,cutoff_sensitivity=0.003):
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
    import dask.array as da
    
    _xds = xds.copy(deep=True)
    
    sampling = np.array(sampling)
    npix_window = np.array(npix_window)
    delta = np.array([_xds[dv].l[1] - _xds[dv].l[0], _xds[dv].m[1] - _xds[dv].m[0]])*3600*180/np.pi
    chunks = _xds[dv].data.chunks[2:] + (3,)
    ellipse_parms  = da.map_blocks(rm_fit, _xds[dv].data, npix_window, sampling, cutoff, cutoff_sensitivity, delta,
                                   dtype=np.double, drop_axis=[0,1], new_axis=[3], chunks=chunks)

    _xds[beam_set_name] = xr.DataArray(ellipse_parms, dims=['time','chan','pol','elps_index'])
    return _xds
    
    
##############################################################
########################### rm_fit ###########################
def fit_ellipse(x,y):
    import numpy.linalg as linalg
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  linalg.eig(np.dot(linalg.inv(S), C))
    n =  np.argmax(E)
    a = V[:,n]
    if a[0] < 0:
        a = -a
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a < c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def ellipse_axis_length(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*(np.sqrt((a-c)**2 + 4*b*b)-(a+c))
    down2=(b*b-a*c)*(-np.sqrt((a-c)**2 + 4*b*b)-(a+c))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def rm_fit(img_to_fit,npix_window,sampling,cutoff,cutoff_sensitivity,delta):
    import numpy.linalg as linalg
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
            
                ellipse_points = np.argwhere(np.abs(interp_img_to_fit-cutoff) < cutoff_sensitivity)
                x_mean = np.mean(ellipse_points[:,0])
                y_mean = np.mean(ellipse_points[:,1])
                x =  ellipse_points[:,0] - x_mean
                y =  ellipse_points[:,1] - y_mean
                
                try:
                    a = fit_ellipse(x,y)
                    center = ellipse_center(a)
                    center[0] += x_mean
                    center[1] += y_mean
                    phi = ellipse_angle_of_rotation(a)
                    axes = ellipse_axis_length(a)
                    center, phi, axes

                    # Should we change this to radians?
                    # convert to useful units
                    phi = np.degrees(phi) - 90. # Astronomers use east of north
                    if phi < -90.:
                        phi += 180.
                        
                    ellipse_parms[time,chan,pol,0] = axes[0]/(sampling[0]-1)*(npix_window[0]-1)*np.abs(delta[0])*2
                    ellipse_parms[time,chan,pol,1] = axes[1]/(sampling[1]-1)*(npix_window[1]-1)*np.abs(delta[1])*2
                    ellipse_parms[time,chan,pol,2] = phi
                except:
                    print('Could not fit psf for time', time, ' chan ', chan, '  pol ', pol)
                    

                
    return ellipse_parms
    
############################################################################################################################



