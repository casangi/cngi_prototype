#   Copyright 2020 AUI, Inc. Washington DC, USA
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numba
from numba import jit
import numpy as np
import numpy.linalg as linalg


########################
def fit_gaussian(img_dataset,image_data_variable_to_fit='PSF',npix_window=[21,21],sampling=[401,401],cutoff=0.5,cutoff_sensitivity=0.003):
    """
    Fit one or more elliptical gaussian components on an image region

    Parameters
    ----------
    img_dataset : xarray.core.dataset.Dataset
        input Image

    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    import xarray as xr
    import matplotlib.pyplot as plt
    import dask.array as da
    from scipy.interpolate import interpn
    
    print('Hallo')
    
    print(img_dataset[image_data_variable_to_fit])
    
    sampling = np.array(sampling)
    npix_window = np.array(npix_window)
    
    
    # Window out the central npix_window pixels
    img_to_fit = img_dataset[image_data_variable_to_fit]
    img_size = np.array([img_to_fit.sizes['d0'],img_to_fit.sizes['d1']])
    img_center = img_size//2
    start_window = img_center - npix_window//2
    end_window = img_center + npix_window//2 + 1
    img_to_fit = img_to_fit[start_window[0]:end_window[0],start_window[1]:end_window[1],:,:]

    interp_d0 = np.linspace(0, npix_window[0]-1, sampling[0])
    interp_d1 = np.linspace(0, npix_window[1]-1, sampling[1])
    
    img_to_fit = img_to_fit.interp(d0=interp_d0,d1=interp_d1,method="linear") #change method="splinef2d" when available
    
    delta = np.array(img_dataset.incr[0:2])*3600*180/np.pi
    new_chunksize = 0
    a = da.map_blocks(find_ellipse,img_to_fit.data,npix_window,sampling,cutoff,cutoff_sensitivity,delta,dtype=np.double,drop_axis=[0,1],new_axis=[2])
    
    
    
    print(a)
    a = a.compute()
    print(a)
    
    
    
    #print(a)
    #a.values
    
    
    return True
    
########################################################################################################################################################################
def fit_ellipse(x,y):
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


def find_ellipse(x, y):
    xmean = x.mean()
    ymean = y.mean()
    x = x - xmean
    y = y - ymean
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    x += xmean
    y += ymean
    return center, phi, axes
    
#@jit(nopython=True, cache=True, nogil=True)
def find_ellipse(img_to_fit,npix_window,sampling,cutoff,cutoff_sensitivity,delta):
    import numpy.linalg as linalg
    
    #ellipse_parms = np.zeros(img_to_fit.shape[2:4] + (3,),dtype=numba.double)
    ellipse_parms = np.zeros(img_to_fit.shape[2:4] + (3,))

    for chan in range(img_to_fit.shape[2]):
        for pol in range(img_to_fit.shape[3]):
            ellipse_points = np.argwhere(np.abs(img_to_fit[:,:,chan,pol]-cutoff) < cutoff_sensitivity)
            x_mean = np.mean(ellipse_points[:,0])
            y_mean = np.mean(ellipse_points[:,1])
            x =  ellipse_points[:,0] - x_mean
            y =  ellipse_points[:,1] - y_mean
            
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
                
            ellipse_parms[chan,pol,0] = axes[0]/(sampling[0]-1)*(npix_window[0]-1)*delta[0]*2
            ellipse_parms[chan,pol,1] = axes[1]/(sampling[1]-1)*(npix_window[1]-1)*delta[1]*2
            ellipse_parms[chan,pol,2] = phi
            
    return ellipse_parms
            
            


    '''
    ellipse_points = np.array(ellipse_points.compute())

    print(ellipse_points)
    print(ellipse_points.shape)

    one_plane = ellipse_points[ellipse_points[:,2]==2]

    print(one_plane.shape)

    plt.figure()
    plt.plot(one_plane[:,0],one_plane[:,1],'.')
    '''
    
    #return np.zeros((1,1,3))


'''
print('*******')





print(img_to_fit)

ellipse_points = da.argwhere(abs(img_to_fit.data-cutoff) < cutoff_sensitivity) #Xarray does not have argwhere

ellipse_points = np.array(ellipse_points.compute())

print(ellipse_points)
print(ellipse_points.shape)

one_plane = ellipse_points[ellipse_points[:,2]==2]

print(one_plane.shape)

plt.figure()
plt.plot(one_plane[:,0],one_plane[:,1],'.')





plt.show()
print('*******')
'''
