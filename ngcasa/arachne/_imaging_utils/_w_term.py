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

import numpy as np

   
def _calculate_w_list(gcf_parms,grid_parms):
    maxUVW = 1/np.abs(np.min(np.abs(grid_parms['cell_size']))*gcf_parms['w_fudge'])
    w_values = maxUVW/(np.arange(gcf_parms['w_projection_planes'],0,-1)**2)
    w_values[0] = 0
    return w_values
    
def _calc_w_sky(w_values,gcf_parms,grid_parms):
    w_sky_image_size = gcf_parms['conv_size']
    w_sky_image_center = w_sky_image_size//2
    #Calculate the cell size for w Term
    w_sky_cell_size = (grid_parms['image_size']*grid_parms['cell_size']*gcf_parms['oversampling'])/(gcf_parms['conv_size'])
    
    x = np.arange(-w_sky_image_center[0], w_sky_image_size[0]-w_sky_image_center[0])*w_sky_cell_size[0]
    y = np.arange(-w_sky_image_center[1], w_sky_image_size[1]-w_sky_image_center[1])*w_sky_cell_size[1]

    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')
    
    w_sky = np.exp((2*np.pi*1j*(np.sqrt(1 - x_grid**2 - y_grid**2) - 1))[:,:,None]*w_values[None,None,:])
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
