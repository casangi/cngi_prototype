#   Copyright 2019 AUI, Inc. Washington DC, USA
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

#ducting - code is complex and might fail after some time if parameters is wrong (time waisting). Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has x parametrs.

#Clustering
#https://stackoverflow.com/questions/11513484/1d-number-array-clustering
#https://stackoverflow.com/questions/7869609/cluster-one-dimensional-data-optimally
#https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit/35151947#35151947
#https://www.astro.rug.nl/~yatawatta/eusipco2.pdf optimal w
#https://arxiv.org/pdf/1807.09239.pdf


import numpy as np
import scipy.constants
import math
import matplotlib.pyplot as plt
import xarray as xr
import dask.array as da

def _create_w_map(mxds,gcf_parms,grid_parms,sel_parms):
    vis_dataset = mxds.attrs[sel_parms['xds']]
    lambda_min = scipy.constants.c/np.max(np.abs(vis_dataset.chan)).data
    lambda_max = scipy.constants.c/np.min(np.abs(vis_dataset.chan)).data
    w_max = np.nanmax(np.abs(vis_dataset.UVW.data.compute()[:,:,2]))/lambda_min
    
    #alternative
    #_find_optimal_w_set(vis_dataset.UVW.data.compute()[:,:,2],gcf_parms['w_step'],gcf_parms['w_hist_cutoff'],lambda_min,lambda_max)
    w_list = _calculate_w_list(gcf_parms,grid_parms,w_max)
    
    w_list = xr.DataArray(da.from_array(w_list,chunks=(1)), dims=('cf_w'))
    return w_list


#Still in development
def _find_optimal_w_set(vals_2d,val_step,val_hist_cutoff,lambda_min,lambda_max):
    val_max = np.nanmax(np.abs(vals_2d))/lambda_min
    val_min = np.nanmin(np.abs(vals_2d))/lambda_max
    
    n_val_cells =  (np.int(np.ceil(val_max/val_step)) + 1)
    val_grid = np.zeros(n_val_cells,np.int)
    
    val_shape = vals_2d.shape
    
    for ii in range(val_shape[0]):
        for jj in range(val_shape[1]):
            if ~np.isnan(vals_2d[ii,jj]):
                val_min = np.abs(vals_2d[ii,jj])/lambda_max
                val_max = np.abs(vals_2d[ii,jj])/lambda_min
                
                dis = val_max - val_min
                n_grid_vals = int(np.ceil(dis/val_step))
                if n_grid_vals > 1:
                    grid_vals = np.zeros(n_grid_vals)
                    #print('n_grid_vals  triggered',n_grid_vals)
                    for kk in range(n_grid_vals):
                        grid_vals[kk] = val_min + kk*dis/(n_grid_vals-1)
                else:
                    grid_vals = [(val_max + val_min)/2]
                    
                for val in grid_vals:
                    grid_indx = int(math.floor(val/val_step + 0.5))
                    val_grid[grid_indx] = val_grid[grid_indx] + 1
                    
    print(val_grid.shape)
    val_centers = np.where(val_grid > val_hist_cutoff)[0]
    
    print(val_centers)
    
    plt.figure()
    plt.plot(val_grid,'.')
    
    #plt.figure()
    #plt.plot(val_centers,'.')
    
    #print(val_centers.shape)
    print('cutoff',val_hist_cutoff)
    
    plt.show()
    
            
    
    
    print(n_val_cells)
    print(val_max)
    print(val_min)
    print(lambda_max,lambda_min)
    print(lambda_min,lambda_max)

   
def _calculate_w_list(gcf_parms,grid_parms,w_max):
    maxUVW = 1/np.abs(np.min(np.abs(grid_parms['cell_size']))*gcf_parms['w_fudge'])
    print(maxUVW)
    if w_max < maxUVW:
        maxUVW = w_max
    
    w_values = maxUVW/(np.arange(gcf_parms['w_projection_planes'],0,-1)**2)
    w_values[0] = 0
    return w_values
    
def _calc_w_sky(w_indx,w_values,gcf_parms,grid_parms):
    w_values = w_values[w_indx[:,0]]
    
    print('In _calc_w_sky')
    w_sky_image_size = gcf_parms['conv_size']
    w_sky_image_center = w_sky_image_size//2
    #Calculate the cell size for w Term
    w_sky_cell_size = (grid_parms['image_size']*grid_parms['cell_size']*gcf_parms['oversampling'])/(gcf_parms['conv_size'])
    
    x = np.arange(-w_sky_image_center[0], w_sky_image_size[0]-w_sky_image_center[0])*w_sky_cell_size[0]
    y = np.arange(-w_sky_image_center[1], w_sky_image_size[1]-w_sky_image_center[1])*w_sky_cell_size[1]
    
    #print('x.shape',x.shape)
    #print('w_values',w_values)

    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
    
    #w_sky = np.exp((2*np.pi*1j*(np.sqrt(1 - x_grid**2 - y_grid**2) - 1))[:,:,None]*w_values[None,None,:])
    #print('w_sky.shape',w_sky.shape)
    w_sky = np.moveaxis(np.exp((2*np.pi*1j*(np.sqrt(1 - x_grid**2 - y_grid**2) - 1))[:,:,None]*w_values[None,None,:]),-1,0)
    print('w_sky.shape',w_sky.shape)
    return w_sky
    
def _calc_w_sky_approx(w_values,gcf_parms,grid_parms):
    w_sky_image_size = gcf_parms['conv_size']
    w_sky_image_center = w_sky_image_size//2
    #Calculate the cell size for w Term
    w_sky_cell_size = (grid_parms['image_size']*grid_parms['cell_size']*gcf_parms['oversampling'])/(gcf_parms['conv_size'])
    
    x = np.arange(-w_sky_image_center[0], w_sky_image_size[0]-w_sky_image_center[0])*w_sky_cell_size[0]
    y = np.arange(-w_sky_image_center[1], w_sky_image_size[1]-w_sky_image_center[1])*w_sky_cell_size[1]
    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
    
    #w_sky = np.exp(   (-np.pi*1j*(x_grid**2 + y_grid**2))[:,:,None]*w_values[None,None,:]    )
    w_sky = 0
    return w_sky
    
def _calc_w_uv_approx(w_values,gcf_parms,grid_parms):
    w_uv_image_size = gcf_parms['conv_size']
    w_uv_image_center = w_uv_image_size//2
    #Calculate the cell size for w Term
    w_uv_cell_size = 1/(grid_parms['image_size']*grid_parms['cell_size']*gcf_parms['oversampling'])
    
    x = np.arange(-w_uv_image_center[0], w_uv_image_size[0]-w_uv_image_center[0])*w_uv_cell_size[0]
    y = np.arange(-w_uv_image_center[1], w_uv_image_size[1]-w_uv_image_center[1])*w_uv_cell_size[1]

    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
    
    print(grid_parms['image_size'],grid_parms['cell_size'],gcf_parms['oversampling'],gcf_parms['conv_size'])
    print(w_values,w_uv_cell_size)
    
    w_uv = (-1j/w_values[None,None,:])*np.exp((np.pi*1j*(x_grid**2 + y_grid**2))[:,:,None]/w_values[None,None,:])
    return w_uv
