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
import numba as nb
import itertools
from copy import deepcopy
import time

def _ndim_list(shape):
    return [_ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]

def deconvolve_point_clean(img_xds, deconvolve_parms, sel_parms):
    """
    .. todo::
        This function is not yet implemented
    
    An iterative solver to construct a model from an observed image(set) and psf(set).
    
    Sky model : Point source
    
    Algorithm : CLEAN (a greedy algorithm for chi-square minimization)
       
    Options : Hogbom, Clark
    
    Input : Requires an input cube (mfs is a cube with nchan=1)
    
    Output : Cube model image
    
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """
    print('######################### Start deconvolve_point_clean #########################')
    from numba import jit
    import time
    import math
    import dask.array.fft as dafft
    import xarray as xr
    import dask.array as da
    import matplotlib.pylab as plt
    import dask
    import copy, os
    from numcodecs import Blosc
    from itertools import cycle
    
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms
 
    
    #print('****',sel_parms,'****')
    _img_xds =  img_xds.copy(deep=True)
    _sel_parms = copy.deepcopy(sel_parms)
    _deconvolve_parms = copy.deepcopy(deconvolve_parms)

    ##############Parameter Checking and Set Defaults##############


    #Check img data_group
    #_check_sel_parms(_img_xds,_sel_parms,new_or_modified_data_variables={'psf_sum_weight':'SUM_WEIGHT','psf':'PSF'},append_to_in_id=True)

    ##################################################################################
    
    
    

    #chan_chunk_size = img_xds[_sel_parms["residual"]].chunks[2][0]

    #freq_chan = da.from_array(vis_dataset.coords['chan'].values, chunks=(chan_chunk_size))

    n_chunks_in_each_dim = _img_xds[sel_parms['data_group_in']["image"]].data.numblocks
    chunk_sizes = _img_xds[sel_parms['data_group_in']["image"]].chunks
    image_shape = _img_xds[sel_parms['data_group_in']["image"]].shape
    
    #print('n_chunks_in_each_dim',n_chunks_in_each_dim)
    
    # l,m,time,chan,pol
    #only allow chunking on time and chan
    
    iter_chunks_indx = itertools.product(np.arange(1),np.arange(1),np.arange(n_chunks_in_each_dim[2]), np.arange(n_chunks_in_each_dim[3]),np.arange(1))
    
    
    model_list = _ndim_list((1,1,n_chunks_in_each_dim[2],n_chunks_in_each_dim[3],1))
    residual_list = _ndim_list((1,1,n_chunks_in_each_dim[2],n_chunks_in_each_dim[3],1))
    #clean_list = _ndim_list((1,1,n_chunks_in_each_dim[2],n_chunks_in_each_dim[3],1))
    
    #c_l, c_m, c_pol chunking is ignored
    for c_l, c_m, c_time, c_chan, c_pol in iter_chunks_indx:
        #print(c_time,c_chan)
        #There are two diffrent gridder wrapped functions _standard_grid_psf_numpy_wrap and _standard_grid_numpy_wrap.
        #This is done to simplify the psf and weight gridding graphs so that the vis_dataset is not loaded.
        
        deconv_graph_node = dask.delayed(_clean_wrap)(
            img_xds[_sel_parms['data_group_in']["image"]].data.partitions[:, :, c_time, c_chan, :],
            img_xds[_sel_parms['data_group_in']["psf"]].data.partitions[:, :, c_time, c_chan, :],
             dask.delayed(deconvolve_parms))
             
        model_list[c_l][c_m][c_time][c_chan][c_pol] = da.from_delayed(deconv_graph_node[0],(image_shape[0],image_shape[1],chunk_sizes[2][c_time],chunk_sizes[3][c_chan],image_shape[4]),dtype=np.double)
        residual_list[c_l][c_m][c_time][c_chan][c_pol] = da.from_delayed(deconv_graph_node[1],(image_shape[0],image_shape[1],chunk_sizes[2][c_time],chunk_sizes[3][c_chan],image_shape[4]),dtype=np.double)
        
    model = da.block(model_list)
    residual = da.block(residual_list)
    
    _img_xds[sel_parms['data_group_out']["residual"]] = xr.DataArray(residual, dims=['l','m','time','chan','pol'])
    _img_xds[sel_parms['data_group_out']["model"]] = xr.DataArray(model, dims=['l','m','time','chan','pol'])
    
    print('######################### Created graph for deconvolve_point_clean #########################')
    return _img_xds
    
def _clean_wrap(dirty, psf, deconvolve_parms):
    """
    gamma, threshold, niter
    Performs Hogbom Clean on the  ``dirty`` image given the ``psf``.
    Parameters
    ----------
    dirty : np.ndarray
        float64 dirty image of shape (ny, nx)
    psf : np.ndarray
        float64 Point Spread Function of shape (2*ny, 2*nx)
    gamma (optional) float
        the gain factor (must be less than one)
    threshold (optional) : float or str
        the threshold to clean to
    niter (optional : integer
        the maximum number of iterations allowed
    Returns
    -------
    np.ndarray
        float64 clean image of shape (ny, nx)
    np.ndarray
        float64 residual image of shape (ny, nx)
    """
    
    # l,m,time,chan,pol
    # deep copy dirties to first residuals,
    # want to keep the original dirty maps
    residual = deepcopy(dirty)
    #print(residual.shape)
    
    model = np.zeros(residual.shape)
    threshold = deconvolve_parms['threshold']
    niter = deconvolve_parms['n_iter']
    gain = deconvolve_parms['gain']
    
    image_shape = dirty.shape
    
    start_time = time.time()
    if deconvolve_parms['decon_kernel'] == 0:
        _clean_jit(residual, model, psf, gain, threshold,niter)
    else:
        _clean_jit_vec(residual, model, psf, gain, threshold,niter)
    print('Time ', time.time()-start_time)
    
    return model, residual
    
    
    
@jit(nopython=True,nogil=True,cache=True)
def _clean_jit(residual, model, psf, gain, threshold,niter):
    peak_pos = np.zeros((2,),dtype=nb.u4)
    #peak_pos = np.zeros((2,),dtype=int)
    
    psf_shape = np.array(psf.shape)
    psf_center = psf_shape//2
    res_shape = np.array(residual.shape)
    res_center = res_shape//2
    
    #print(psf_shape,psf_center)
    #print(res_shape,res_center)

    for i_time in range(res_shape[2]):
        for i_chan in range(res_shape[3]):
            for i_pol in range(res_shape[4]):
                i = 0
                _abs_image_peaks(residual[:,:,i_time,i_chan,i_pol],peak_pos)
                peak = residual[peak_pos[0],peak_pos[1],i_time,i_chan,i_pol]
                if np.isnan(peak) or (peak==0.0):
                    i = niter
                peak_abs = np.abs(peak)
                scaled_threshold = threshold*peak_abs
                
                #print(scaled_threshold,peak_abs)
                
                while peak_abs > scaled_threshold and i < niter:
                    model[peak_pos[0],peak_pos[1],i_time,i_chan,i_pol] += gain*peak
                    
                    res_start_indx_x = peak_pos[0] - psf_center[0]
                    if res_start_indx_x < 0:
                        res_start_indx_x = 0
                    
                    res_start_indx_y = peak_pos[1] - psf_center[1]
                    if res_start_indx_y < 0:
                        res_start_indx_y = 0
                    
                    res_end_indx_x = peak_pos[0] + (psf_shape[0] - psf_center[0]) #Is actauly end index plus 1, because np.arange is [...)
                    if res_end_indx_x >= res_shape[0]:
                        res_end_indx_x = res_shape[0]
                    
                    res_end_indx_y = peak_pos[1] + (psf_shape[1] - psf_center[1])
                    if res_end_indx_y >= res_shape[1]:
                        res_end_indx_y = res_shape[1]
                    
                    #print('peak_pos',peak_pos)
                    #print('res_start_indx_x,res_end_indx_x',res_start_indx_x,res_end_indx_x)
                    #print('res_start_indx_y,res_end_indx_y',res_start_indx_y,res_end_indx_y)
                    
                    for i_x in np.arange(res_start_indx_x,res_end_indx_x):
                        for i_y in np.arange(res_start_indx_y,res_end_indx_y):
                            psf_i_x = psf_center[0] - (peak_pos[0] - res_start_indx_x) + (i_x - res_start_indx_x)
                            psf_i_y = psf_center[1] - (peak_pos[1] - res_start_indx_y) + (i_y - res_start_indx_y)
                            residual[i_x,i_y,i_time,i_chan,i_pol] -= gain*psf[psf_i_x,psf_i_y,i_time,i_chan,i_pol]
                    
                    _abs_image_peaks(residual[:,:,i_time,i_chan,i_pol],peak_pos)
                    peak = residual[peak_pos[0],peak_pos[1],i_time,i_chan,i_pol]
                    if np.isnan(peak) or (peak==0.0):
                        i = niter
                    peak_abs = np.abs(peak)
                    i += 1
                    
                #print(i_time,i_chan,i_pol,scaled_threshold,peak_abs,peak_pos,i,niter,gain)
                    
#Partially vectorized
def _clean_jit_vec(residual, model, psf, gain, threshold,niter):
    peak_pos = np.zeros((2,),dtype=int)
    
    psf_shape = np.array(psf.shape)
    psf_center = psf_shape//2
    res_shape = np.array(residual.shape)
    res_center = res_shape//2
    
    #print(psf_shape,psf_center)
    #print(res_shape,res_center)

    for i_time in range(res_shape[2]):
        for i_chan in range(res_shape[3]):
            for i_pol in range(res_shape[4]):
                i = 0
                
                peak_pos = np.array(np.unravel_index(np.nanargmax(np.abs(residual[:,:,i_time,i_chan,i_pol])),res_shape[0:2]))
                peak = residual[peak_pos[0],peak_pos[1],i_time,i_chan,i_pol]
                
                if np.isnan(peak) or (peak==0.0):
                    i = niter
                peak_abs = np.abs(peak)
                scaled_threshold = threshold*peak_abs
                
                #print(scaled_threshold,peak_abs)
                
                while peak_abs > scaled_threshold and i < niter:
                    model[peak_pos[0],peak_pos[1],i_time,i_chan,i_pol] += gain*peak
                    
                    res_start_indx_x = peak_pos[0] - psf_center[0]
                    if res_start_indx_x < 0:
                        res_start_indx_x = 0
                    
                    res_start_indx_y = peak_pos[1] - psf_center[1]
                    if res_start_indx_y < 0:
                        res_start_indx_y = 0
                    
                    res_end_indx_x = peak_pos[0] + (psf_shape[0] - psf_center[0]) #Is actauly end index plus 1, because np.arange is [...)
                    if res_end_indx_x >= res_shape[0]:
                        res_end_indx_x = res_shape[0]
                    
                    res_end_indx_y = peak_pos[1] + (psf_shape[1] - psf_center[1])
                    if res_end_indx_y >= res_shape[1]:
                        res_end_indx_y = res_shape[1]
                    
                    psf_start_indx_x = psf_center[0] - (peak_pos[0] - res_start_indx_x)
                    psf_start_indx_y = psf_center[1] - (peak_pos[1] - res_start_indx_y)
                    psf_end_indx_x = psf_center[0] + (res_end_indx_x - peak_pos[0])
                    psf_end_indx_y = psf_center[1] + (res_end_indx_y - peak_pos[1])
                    
                    residual[res_start_indx_x:res_end_indx_x,res_start_indx_y:res_end_indx_y,i_time,i_chan,i_pol] -= gain*psf[psf_start_indx_x:psf_end_indx_x,psf_start_indx_y:psf_end_indx_y,i_time,i_chan,i_pol]
                    
                    peak_pos = np.array(np.unravel_index(np.nanargmax(np.abs(residual[:,:,i_time,i_chan,i_pol])),res_shape[0:2]))
                    peak = residual[peak_pos[0],peak_pos[1],i_time,i_chan,i_pol]
                    if np.isnan(peak) or (peak==0.0):
                        i = niter
                    peak_abs = np.abs(peak)
                    i += 1
                    
                #print(i_time,i_chan,i_pol,scaled_threshold,peak_abs,peak_pos,i,niter,gain)
                


    
@jit(nopython=True,nogil=True,cache=True)
def _abs_image_peaks(image,peak_pos):
    #min = image[0,0]
    
    max = 0.0
    #min_x = 0
    #min_y = 0
    max_x = 0
    max_y = 0
    
    for i_x in range(image.shape[0]):
        for i_y in range(image.shape[1]):
            if np.abs(image[i_x,i_y]) > max:
                max = np.abs(image[i_x,i_y])
                peak_pos[0] = i_x
                peak_pos[1] = i_y
            
#            if image[i_x,i_y] < min:
#                min = image[i_x,i_y]
#                peak_pos[2] = i_x
#                peak_pos[3] = i_y
                




    
   
    

