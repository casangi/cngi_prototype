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
from numba import jit
import numpy as np
import math

def ndim_list(shape):
    return [ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]

def _graph_standard_grid(vis_dataset, cgk_1D, grid_parms, sel_parms):
    import dask
    import dask.array as da
    import xarray as xr
    import time
    import itertools

    # Getting data for gridding
    chan_chunk_size = vis_dataset[sel_parms["imaging_weight"]].chunks[2][0]

    freq_chan = da.from_array(vis_dataset.coords['chan'].values, chunks=(chan_chunk_size))

    n_chunks_in_each_dim = vis_dataset[sel_parms["imaging_weight"]].data.numblocks
    chunk_indx = []

    iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]),
                                         np.arange(n_chunks_in_each_dim[2]), np.arange(n_chunks_in_each_dim[3]))
                                         
    if grid_parms['chan_mode'] == 'continuum':
        n_chan_chunks_img = 1
        n_other_chunks = n_chunks_in_each_dim[0]*n_chunks_in_each_dim[1]*n_chunks_in_each_dim[2]*n_chunks_in_each_dim[3]
    elif grid_parms['chan_mode'] == 'cube':
        n_chan_chunks_img = n_chunks_in_each_dim[2]
        n_other_chunks = n_chunks_in_each_dim[0]*n_chunks_in_each_dim[1]*n_chunks_in_each_dim[3]
    
    #n_delayed = np.prod(n_chunks_in_each_dim)
    chunk_sizes = vis_dataset[sel_parms["imaging_weight"]].chunks

    list_of_grids = ndim_list((n_chan_chunks_img,n_other_chunks))
    list_of_sum_weights = ndim_list((n_chan_chunks_img,n_other_chunks))
    
  
    # Build graph
    for c_time, c_baseline, c_chan, c_pol in iter_chunks_indx:
        #There are two diffrent gridder wrapped functions _standard_grid_psf_numpy_wrap and _standard_grid_numpy_wrap.
        #This is done to simplify the psf and weight gridding graphs so that the vis_dataset is not loaded.
        if grid_parms['do_psf']:
            sub_grid_and_sum_weights = dask.delayed(_standard_grid_psf_numpy_wrap)(
            vis_dataset[sel_parms["uvw"]].data.partitions[c_time, c_baseline, 0],
            vis_dataset[sel_parms["imaging_weight"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
             freq_chan.partitions[c_chan],
            dask.delayed(cgk_1D), dask.delayed(grid_parms))
            grid_dtype = np.double
        else:
            sub_grid_and_sum_weights = dask.delayed(_standard_grid_numpy_wrap)(
            vis_dataset[sel_parms["data"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
            vis_dataset[sel_parms["uvw"]].data.partitions[c_time, c_baseline, 0],
            vis_dataset[sel_parms["imaging_weight"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
            freq_chan.partitions[c_chan],
            dask.delayed(cgk_1D), dask.delayed(grid_parms))
            grid_dtype = np.complex128
        
        
        if grid_parms['chan_mode'] == 'continuum':
            c_time_baseline_chan_pol = c_pol + c_chan*n_chunks_in_each_dim[3] + c_baseline*n_chunks_in_each_dim[3]*n_chunks_in_each_dim[2] + c_time*n_chunks_in_each_dim[3]*n_chunks_in_each_dim[2]*n_chunks_in_each_dim[1]
            list_of_grids[0][c_time_baseline_chan_pol] = da.from_delayed(sub_grid_and_sum_weights[0], (1, chunk_sizes[3][c_pol], grid_parms['image_size_padded'][0], grid_parms['image_size_padded'][1]),dtype=grid_dtype)
            list_of_sum_weights[0][c_time_baseline_chan_pol] = da.from_delayed(sub_grid_and_sum_weights[1],(1, chunk_sizes[3][c_pol]),dtype=np.float64)
            
        elif grid_parms['chan_mode'] == 'cube':
            c_time_baseline_pol = c_pol + c_baseline*n_chunks_in_each_dim[3] + c_time*n_chunks_in_each_dim[1]*n_chunks_in_each_dim[3]
            list_of_grids[c_chan][c_time_baseline_pol] = da.from_delayed(sub_grid_and_sum_weights[0], (chunk_sizes[2][c_chan], chunk_sizes[3][c_pol], grid_parms['image_size_padded'][0], grid_parms['image_size_padded'][1]),dtype=grid_dtype)
            list_of_sum_weights[c_chan][c_time_baseline_pol]  = da.from_delayed(sub_grid_and_sum_weights[1],(chunk_sizes[2][c_chan], chunk_sizes[3][c_pol]),dtype=np.float64)
            
            
    # Sum grids
    for c_chan in range(n_chan_chunks_img):
        list_of_grids[c_chan] = _tree_sum_list(list_of_grids[c_chan])
        list_of_sum_weights[c_chan] = _tree_sum_list(list_of_sum_weights[c_chan])

    # Concatenate Cube
    if grid_parms['chan_mode'] == 'cube':
        list_of_grids_and_sum_weights = [da.concatenate(list_of_grids,axis=1)[0],da.concatenate(list_of_sum_weights,axis=1)[0]]
    else:
        list_of_grids_and_sum_weights = [list_of_grids[0][0],list_of_sum_weights[0][0]]
    
    # Put axes in image orientation. How much does this add to compute?
    list_of_grids_and_sum_weights[0] = da.moveaxis(list_of_grids_and_sum_weights[0], [0, 1],
                                                                [-2, -1])
    
    list_of_grids_and_sum_weights[1] = da.moveaxis(list_of_grids_and_sum_weights[1],[0, 1], [-2, -1])
    
    return list_of_grids_and_sum_weights
    

def _tree_sum_list(list_to_sum):
    import dask.array as da
    while len(list_to_sum) > 1:
        new_list_to_sum = []
        for i in range(0, len(list_to_sum), 2):
            if i < len(list_to_sum) - 1:
                lazy = da.add(list_to_sum[i],list_to_sum[i+1])
            else:
                lazy = list_to_sum[i]
            new_list_to_sum.append(lazy)
        list_to_sum = new_list_to_sum
    return list_to_sum
    
    
def _standard_grid_numpy_wrap(vis_data, uvw, weight, freq_chan, cgk_1D, grid_parms):
    """
      Wraps the jit gridder code.
      
      Parameters
      ----------
      grid : complex array
          (n_chan, n_pol, n_u, n_v)
      sum_weight : float array
          (n_chan, n_pol) 
      uvw  : float array
          (n_time, n_baseline, 3)
      freq_chan : float array
          (n_chan)
      weight : float array
          (n_time, n_baseline, n_vis_chan)
      cgk_1D : float array
          (oversampling*(support//2 + 1))
      grid_parms : dictionary
          keys ('image_size','cell','oversampling','support')

      Returns
      -------
      grid : complex array
          (1,n_imag_chan,n_imag_pol,n_u,n_v)
      """

    n_chan = weight.shape[2]
    if grid_parms['chan_mode'] == 'cube':
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(np.int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(np.int)

    n_imag_pol = weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(np.int)

    n_uv = grid_parms['image_size_padded']
    delta_lm = grid_parms['cell_size']
    oversampling = grid_parms['oversampling']
    support = grid_parms['support']
    
    if grid_parms['complex_grid']:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128)
    else:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)
    
    do_psf = grid_parms['do_psf']
    _standard_grid_jit(grid, sum_weight, do_psf, vis_data, uvw, freq_chan, chan_map, pol_map, weight, cgk_1D, n_uv, delta_lm, support, oversampling)
     

    return grid, sum_weight


def _standard_grid_psf_numpy_wrap(uvw, weight, freq_chan, cgk_1D, grid_parms):
    """
      Wraps the jit gridder code.
      
      Parameters
      ----------
      grid : complex array
          (n_chan, n_pol, n_u, n_v)
      sum_weight : float array
          (n_chan, n_pol)
      uvw  : float array
          (n_time, n_baseline, 3)
      freq_chan : float array
          (n_chan)
      weight : float array
          (n_time, n_baseline, n_vis_chan)
      cgk_1D : float array
          (oversampling*(support//2 + 1))
      grid_parms : dictionary
          keys ('image_size','cell','oversampling','support')

      Returns
      -------
      grid : complex array
          (1,n_imag_chan,n_imag_pol,n_u,n_v)
      """
    

    
    n_chan = weight.shape[2]
    if grid_parms['chan_mode'] == 'cube':
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(np.int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(np.int)
        

    n_imag_pol = weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(np.int)

    n_uv = grid_parms['image_size_padded']
    delta_lm = grid_parms['cell_size']
    oversampling = grid_parms['oversampling']
    support = grid_parms['support']
    
    grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)
    
    do_psf = grid_parms['do_psf']
    vis_data = np.zeros((1, 1, 1, 1), dtype=np.bool) #This 0 bool array is needed to pass to _standard_grid_jit so that the code can be resued and to keep numba happy.

    _standard_grid_jit(grid, sum_weight, do_psf, vis_data, uvw, freq_chan, chan_map, pol_map, weight, cgk_1D, n_uv, delta_lm, support, oversampling)
    
    return grid, sum_weight


import numpy as np

#When jit is used round is repolaced by standard c++ round that is different to python round
@jit(nopython=True, cache=True, nogil=True)
def _standard_grid_jit(grid, sum_weight, do_psf, vis_data, uvw, freq_chan, chan_map, pol_map, weight, cgk_1D,
                       n_uv, delta_lm, support, oversampling):
    """
      Parameters
      ----------
      grid : complex array 
          (n_chan, n_pol, n_u, n_v)
      sum_weight : float array 
          (n_chan, n_pol) 
      vis_data : complex array 
          (n_time, n_baseline, n_vis_chan, n_pol)
      uvw  : float array 
          (n_time, n_baseline, 3)
      freq_chan : float array 
          (n_chan)
      chan_map : int array 
          (n_chan)
      pol_map : int array 
          (n_pol)
      weight : float array 
          (n_time, n_baseline, n_vis_chan)
      cgk_1D : float array 
          (oversampling*(support//2 + 1))
      grid_parms : dictionary 
          keys ('n_imag_chan','n_imag_pol','n_uv','delta_lm','oversampling','support')

      Returns
      -------
      """
      
    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c

    #oversampling_center = int(oversampling // 2)
    support_center = int(support // 2)
    uv_center = n_uv // 2

    start_support = - support_center
    end_support = support - support_center # end_support is larger by 1 so that python range() gives correct indices
    
    
    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(chan_map)
    n_pol = len(pol_map)
    
    n_u = n_uv[0]
    n_v = n_uv[1]
    
    
    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                
                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]
                    
                    #Doing round as int(x+0.5) since u_pos/v_pos should always positive and this matices fortran and gives consistant rounding.
                    #u_center_indx = int(u_pos + 0.5)
                    #v_center_indx = int(v_pos + 0.5)
                    
                    #Do not use numpy round
                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)
                    
                    if (u_center_indx+support_center < n_u) and (v_center_indx+support_center < n_v) and (u_center_indx-support_center >= 0) and (v_center_indx-support_center >= 0):
                        u_offset = u_center_indx - u_pos
                        u_center_offset_indx = math.floor(u_offset * oversampling + 0.5)
                        v_offset = v_center_indx - v_pos
                        v_center_offset_indx = math.floor(v_offset * oversampling + 0.5)
                        
                        for i_pol in range(n_pol):
                            if do_psf:
                                weighted_data = weight[i_time, i_baseline, i_chan, i_pol]
                            else:
                                weighted_data = vis_data[i_time, i_baseline, i_chan, i_pol] * weight[i_time, i_baseline, i_chan, i_pol]
                                
                            #print('1. u_center_indx, v_center_indx', u_center_indx, v_center_indx, vis_data[i_time, i_baseline, i_chan, i_pol], weight[i_time, i_baseline, i_chan, i_pol])
                            
                            if ~np.isnan(weighted_data) and (weighted_data != 0.0):
                                a_pol = pol_map[i_pol]
                                norm = 0.0
                                
                                for i_v in range(start_support,end_support):
                                    v_indx = v_center_indx + i_v
                                    v_offset_indx = np.abs(oversampling * i_v + v_center_offset_indx)
                                    conv_v = cgk_1D[v_offset_indx]
                                        

                                    for i_u in range(start_support,end_support):
                                        u_indx = u_center_indx + i_u
                                        u_offset_indx = np.abs(oversampling * i_u + u_center_offset_indx)
                                        conv_u = cgk_1D[u_offset_indx]
                                        conv = conv_u * conv_v
                                            
                                        grid[a_chan, a_pol, u_indx, v_indx] = grid[a_chan, a_pol, u_indx, v_indx] + conv * weighted_data
                                        norm = norm + conv
                                        
                                        
                                sum_weight[a_chan, a_pol] = sum_weight[a_chan, a_pol] + weight[i_time, i_baseline, i_chan, i_pol] * norm

    return

############################################################################################################################################################################################################################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################################################################################################################################################################################################################



def _graph_standard_degrid(vis_dataset, grid, briggs_factors, cgk_1D, grid_parms, sel_parms):
   import dask
   import dask.array as da
   import xarray as xr
   import time
   import itertools
   
   # Getting data for gridding
   chan_chunk_size = vis_dataset[sel_parms["imaging_weight"]].chunks[2][0]

   freq_chan = da.from_array(vis_dataset.coords['chan'].values, chunks=(chan_chunk_size))

   n_chunks_in_each_dim = vis_dataset[sel_parms["imaging_weight"]].data.numblocks
   chunk_indx = []

   iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]),
                                        np.arange(n_chunks_in_each_dim[2]), np.arange(n_chunks_in_each_dim[3]))

   #n_delayed = np.prod(n_chunks_in_each_dim)
   chunk_sizes = vis_dataset[sel_parms["imaging_weight"]].chunks

   n_chan_chunks_img = n_chunks_in_each_dim[2]
   list_of_degrids = []
   list_of_sum_weights = []
   
   list_of_degrids = ndim_list(n_chunks_in_each_dim)
   
   
   # Build graph
   for c_time, c_baseline, c_chan, c_pol in iter_chunks_indx:
       if grid_parms['chan_mode'] == 'cube':
            a_c_chan = c_chan
       else:
            a_c_chan = 0
       
       if grid_parms['do_imaging_weight']:
           sub_degrid = dask.delayed(_standard_imaging_weight_degrid_numpy_wrap)(
                grid.partitions[0,0,a_c_chan,c_pol],
                vis_dataset[sel_parms["uvw"]].data.partitions[c_time, c_baseline, 0],
                vis_dataset[sel_parms["imaging_weight"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
                briggs_factors.partitions[:,a_c_chan,c_pol],
                freq_chan.partitions[c_chan],
                dask.delayed(grid_parms))
                
           single_chunk_size = (chunk_sizes[0][c_time], chunk_sizes[1][c_baseline],chunk_sizes[2][c_chan], chunk_sizes[3][c_pol])
           list_of_degrids[c_time][c_baseline][c_chan][c_pol] = da.from_delayed(sub_degrid, single_chunk_size,dtype=np.double)
       else:
           print('Degridding of visibilities and psf still needs to be implemented')
           
           #sub_grid_and_sum_weights = dask.delayed(_standard_grid_numpy_wrap)(
           #vis_dataset[vis_dataset[grid_parms["data"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
           #vis_dataset[grid_parms["uvw"]].data.partitions[c_time, c_baseline, 0],
           #vis_dataset[grid_parms["imaging_weight"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
           #freq_chan.partitions[c_chan],
           #dask.delayed(cgk_1D), dask.delayed(grid_parms))
       
   degrid = da.block(list_of_degrids)
   return degrid


def _standard_imaging_weight_degrid_numpy_wrap(grid_imaging_weight, uvw, natural_imaging_weight, briggs_factors, freq_chan, grid_parms):
    n_chan = natural_imaging_weight.shape[2]
    n_imag_chan = n_chan
    
    if grid_parms['chan_mode'] == 'cube':
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(np.int)
    else:  # continuum
        n_imag_chan = 1
        chan_map = (np.zeros(n_chan)).astype(np.int)
        
    n_imag_pol = natural_imaging_weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(np.int)

    n_uv = grid_parms['image_size_padded']
    delta_lm = grid_parms['cell_size']
    
    imaging_weight = np.zeros(natural_imaging_weight.shape, dtype=np.double)
                       
    _standard_imaging_weight_degrid_jit(imaging_weight, grid_imaging_weight, briggs_factors, uvw, freq_chan, chan_map, pol_map, natural_imaging_weight,n_uv, delta_lm)
    
    return imaging_weight

@jit(nopython=True, cache=True, nogil=True)
def _standard_imaging_weight_degrid_jit(imaging_weight, grid_imaging_weight, briggs_factors, uvw, freq_chan, chan_map, pol_map, natural_imaging_weight, n_uv, delta_lm):
    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c
    
    uv_center = n_uv // 2
    
    
    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(chan_map)
    n_pol = len(pol_map)
    n_imag_chan = chan_map.shape[0]

    n_u = n_uv[0]
    n_v = n_uv[1]
    
    #print('Degrid operation')
    
    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]
                    
                    #Doing round as int(x+0.5) since u_pos/v_pos should always be positive and  fortran and gives consistant rounding.
                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)
                    
                    #print('f uv', freq_chan[i_chan], uvw[i_time, i_baseline, 0],uvw[i_time, i_baseline, 1])
                    if (u_center_indx < n_u) and (v_center_indx < n_v) and (u_center_indx >= 0) and (v_center_indx >= 0):
                        #print('u_center_indx, v_center_indx',  u_center_indx, v_center_indx)
                        for i_pol in range(n_pol):
                            a_pol = pol_map[i_pol]
                            
                            
                            imaging_weight[i_time, i_baseline, i_chan, i_pol] = natural_imaging_weight[i_time, i_baseline, i_chan, i_pol]
                            if ~np.isnan(natural_imaging_weight[i_time, i_baseline, i_chan, i_pol]) and (natural_imaging_weight[i_time, i_baseline, i_chan, i_pol] != 0.0):
                                if ~np.isnan(grid_imaging_weight[u_center_indx, v_center_indx, a_chan, a_pol]) and (grid_imaging_weight[u_center_indx, v_center_indx, a_chan, a_pol] != 0.0):
                                    briggs_grid_imaging_weight = briggs_factors[0,a_chan,a_pol]*grid_imaging_weight[u_center_indx, v_center_indx, a_chan, a_pol] + briggs_factors[1,a_chan,a_pol]
                                    imaging_weight[i_time, i_baseline, i_chan, i_pol] = imaging_weight[i_time, i_baseline, i_chan, i_pol] / briggs_grid_imaging_weight
                                
    return




