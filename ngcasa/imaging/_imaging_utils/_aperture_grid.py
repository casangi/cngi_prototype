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
#from numba import gdb

def ndim_list(shape):
    return [ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]

def _graph_aperture_grid(vis_dataset,gcf_dataset,grid_parms,sel_parms):
    import dask
    import dask.array as da
    import xarray as xr
    import time
    import itertools
    import matplotlib.pyplot as plt
    
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
    
    
    #print(cf_dataset)
    
    grid_parms['complex_grid'] = True
    # Build graph
    
    for c_time, c_baseline, c_chan, c_pol in iter_chunks_indx:
        
        if grid_parms['grid_weights']:
            sub_grid_and_sum_weights = dask.delayed(_aperture_weight_grid_numpy_wrap)(
            vis_dataset[sel_parms["uvw"]].data.partitions[c_time, c_baseline, 0],
            vis_dataset[sel_parms["imaging_weight"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
            vis_dataset["field_id"].data.partitions[c_time],
            gcf_dataset["CF_BASELINE_MAP"].data.partitions[c_baseline],
            gcf_dataset["CF_CHAN_MAP"].data.partitions[c_chan],
            gcf_dataset["CF_POL_MAP"].data.partitions[c_pol],
            gcf_dataset["WEIGHT_CONV_KERNEL"].data,
            gcf_dataset["SUPPORT"].data,
            gcf_dataset["PHASE_GRADIENT"].data,
            freq_chan.partitions[c_chan],
            dask.delayed(grid_parms))
            grid_dtype = np.complex128
        else:
            sub_grid_and_sum_weights = dask.delayed(_aperture_grid_numpy_wrap)(
            vis_dataset[sel_parms["data"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
            vis_dataset[sel_parms["uvw"]].data.partitions[c_time, c_baseline, 0],
            vis_dataset[sel_parms["imaging_weight"]].data.partitions[c_time, c_baseline, c_chan, c_pol],
            vis_dataset["field_id"].data.partitions[c_time],
            gcf_dataset["CF_BASELINE_MAP"].data.partitions[c_baseline],
            gcf_dataset["CF_CHAN_MAP"].data.partitions[c_chan],
            gcf_dataset["CF_POL_MAP"].data.partitions[c_pol],
            gcf_dataset["CONV_KERNEL"].data,
            gcf_dataset["SUPPORT"].data,
            gcf_dataset["PHASE_GRADIENT"].data,
            freq_chan.partitions[c_chan],
            dask.delayed(grid_parms))
            grid_dtype = np.complex128
            

        if grid_parms['chan_mode'] == 'continuum':
            c_time_baseline_chan_pol = c_pol + c_chan*n_chunks_in_each_dim[3] + c_baseline*n_chunks_in_each_dim[3]*n_chunks_in_each_dim[2] + c_time*n_chunks_in_each_dim[3]*n_chunks_in_each_dim[2]*n_chunks_in_each_dim[1]
            list_of_grids[0][c_time_baseline_chan_pol] = da.from_delayed(sub_grid_and_sum_weights[0], (1, chunk_sizes[3][c_pol], grid_parms['image_size_padded'][0], grid_parms['image_size_padded'][1]),dtype=grid_dtype)
            list_of_sum_weights[0][c_time_baseline_chan_pol] = da.from_delayed(sub_grid_and_sum_weights[1],(1, chunk_sizes[3][c_pol]),dtype=np.double)
            
        elif grid_parms['chan_mode'] == 'cube':
            c_time_baseline_pol = c_pol + c_baseline*n_chunks_in_each_dim[3] + c_time*n_chunks_in_each_dim[1]*n_chunks_in_each_dim[3]
            list_of_grids[c_chan][c_time_baseline_pol] = da.from_delayed(sub_grid_and_sum_weights[0], (chunk_sizes[2][c_chan], chunk_sizes[3][c_pol], grid_parms['image_size_padded'][0], grid_parms['image_size_padded'][1]),dtype=grid_dtype)
            list_of_sum_weights[c_chan][c_time_baseline_pol]  = da.from_delayed(sub_grid_and_sum_weights[1],(chunk_sizes[2][c_chan], chunk_sizes[3][c_pol]),dtype=np.double)
        
    
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
    


def _aperture_weight_grid_numpy_wrap(uvw,imaging_weight,field,cf_baseline_map,cf_chan_map,cf_pol_map,weight_conv_kernel,weight_support,phase_gradient,freq_chan,grid_parms):
    #print('imaging_weight ', imaging_weight.shape)
    #print('cf_chan_map ', cf_chan_map.shape, ' cf_baseline_map', cf_baseline_map.shape, 'cf_pol_map', cf_pol_map.shape )
    
    n_chan = imaging_weight.shape[2]
    if grid_parms['chan_mode'] == 'cube':
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(np.int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(np.int)

    n_imag_pol = imaging_weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(np.int)

    n_uv = grid_parms['image_size_padded']
    delta_lm = grid_parms['cell_size']
    oversampling = grid_parms['oversampling']
    
    
    if grid_parms['complex_grid']:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128)
    else:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)
    
    _aperture_weight_grid_jit(grid, sum_weight, uvw, freq_chan, chan_map, pol_map, cf_baseline_map, cf_chan_map, cf_pol_map, imaging_weight, weight_conv_kernel, n_uv, delta_lm, weight_support, oversampling, field, phase_gradient)


    return grid, sum_weight
    
@jit(nopython=True, cache=True, nogil=True)
def _aperture_weight_grid_jit(grid, sum_weight, uvw, freq_chan, chan_map, pol_map, cf_baseline_map, cf_chan_map, cf_pol_map, imaging_weight, weight_conv_kernel, n_uv, delta_lm, weight_support, oversampling, field, phase_gradient):
    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c
    
    uv_center = n_uv // 2
    
    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(chan_map)
    n_pol = len(pol_map)
    
    n_u = n_uv[0]
    n_v = n_uv[1]
    
    u_center = uv_center[0]
    v_center = uv_center[1]
    
    max_support_center = np.max(weight_support)
    
    conv_v_center = weight_conv_kernel.shape[-1]//2
    conv_u_center = weight_conv_kernel.shape[-2]//2
    
    #print(phase_gradient.shape)
    #print(weight_conv_kernel.shape)
    
    prev_field = -1
    
    for i_time in range(n_time):
        if prev_field != field[i_time]:
            weight_conv_kernel_phase_gradient = weight_conv_kernel*phase_gradient[field[i_time],:,:]
            prev_field = field[i_time]
    
        for i_baseline in range(n_baseline):
            cf_baseline = cf_baseline_map[i_baseline]
            for i_chan in range(n_chan):
                cf_chan = cf_chan_map[i_chan]
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                
                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]
                    
                    #Do not use numpy round
                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)
                    
                    if (u_center_indx+max_support_center < n_u) and (v_center_indx+max_support_center < n_v) and (u_center_indx-max_support_center >= 0) and (v_center_indx-max_support_center >= 0):
                        #u_offset = u_center_indx - u_pos
                        #u_center_offset_indx = math.floor(u_offset * oversampling[0] + 0.5)
                        #v_offset = v_center_indx - v_pos
                        #v_center_offset_indx = math.floor(v_offset * oversampling[1] + 0.5)
                        
                        for i_pol in range(n_pol):
                            weighted_data = imaging_weight[i_time, i_baseline, i_chan, i_pol]
                                
                            if ~np.isnan(weighted_data) and (weighted_data != 0.0):
                                cf_pol = cf_pol_map[i_pol]
                                a_pol = pol_map[i_pol]
                                norm = 0.0
                                
                                '''
                                support = weight_support[cf_baseline,cf_chan,cf_pol,:]
                                #support = np.array([13,13])
                                support_center = support // 2
                                start_support = - support_center
                                end_support = support - support_center # end_support is larger by 1 so that python range() gives correct indices
                                '''
                                
                                
                                support_u = weight_support[cf_baseline,cf_chan,cf_pol,0]
                                support_v = weight_support[cf_baseline,cf_chan,cf_pol,1]
                                
                                support_center_u = support_u // 2
                                support_center_v = support_v // 2
                                
                                start_support_u = - support_center_u
                                start_support_v = - support_center_v
                                
                                end_support_u = support_u - support_center_u
                                end_support_v = support_v - support_center_v
                                
                                
                                for i_v in range(start_support_v,end_support_v):
                                    v_indx = v_center + i_v
                                    cf_v_indx = oversampling[1]*i_v + conv_v_center

                                    for i_u in range(start_support_u,end_support_u):
                                        u_indx = u_center + i_u
                                        cf_u_indx = oversampling[0]*i_u + conv_u_center
                                        
                                        conv = weight_conv_kernel_phase_gradient[cf_baseline,cf_chan,cf_pol,cf_u_indx,cf_v_indx]
                                        
                                        grid[a_chan, a_pol, u_indx, v_indx] = grid[a_chan, a_pol, u_indx, v_indx] +   conv * weighted_data
                                        norm = norm + conv
                            
                                sum_weight[a_chan, a_pol] = sum_weight[a_chan, a_pol] + weighted_data * np.real(norm)

    return


def _aperture_grid_numpy_wrap(vis_data,uvw,imaging_weight,field,cf_baseline_map,cf_chan_map,cf_pol_map,conv_kernel,weight_support,phase_gradient,freq_chan,grid_parms):
    #print('imaging_weight ', imaging_weight.shape)
    import time
    
    n_chan = imaging_weight.shape[2]
    if grid_parms['chan_mode'] == 'cube':
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(np.int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(np.int)

    n_imag_pol = imaging_weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(np.int)

    n_uv = grid_parms['image_size_padded']
    delta_lm = grid_parms['cell_size']
    oversampling = grid_parms['oversampling']
    
    
    if grid_parms['complex_grid']:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128)
    else:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)
    
    do_psf = grid_parms['do_psf']
    
    #print('vis_data', vis_data.shape , 'grid ', grid.shape, 'sum_weight', sum_weight.shape, 'cf_chan_map ', cf_chan_map.shape, ' cf_baseline_map', cf_baseline_map.shape, 'cf_pol_map', cf_pol_map.shape, ' conv_kernel',  conv_kernel.shape, 'phase_gradient', phase_gradient.shape, 'field', field.shape,  )
    
    #start = time.time()
    _aperture_grid_jit(grid, sum_weight, do_psf, vis_data, uvw, freq_chan, chan_map, pol_map, cf_baseline_map, cf_chan_map, cf_pol_map, imaging_weight, conv_kernel, n_uv, delta_lm, weight_support, oversampling, field, phase_gradient)
    #time_to_grid = time.time() - start
    #print("time to grid ", time_to_grid)


    return grid, sum_weight
    
# Important changes to be made https://github.com/numba/numba/issues/4261
# debug=True and gdb()
@jit(nopython=True, cache=True, nogil=True)
def _aperture_grid_jit(grid, sum_weight, do_psf, vis_data, uvw, freq_chan, chan_map, pol_map, cf_baseline_map, cf_chan_map, cf_pol_map, imaging_weight, conv_kernel, n_uv, delta_lm, weight_support, oversampling, field, phase_gradient):

    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c
    
    uv_center = n_uv // 2
    
    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(chan_map)
    n_pol = len(pol_map)
    
    n_u = n_uv[0]
    n_v = n_uv[1]
    
    u_center = uv_center[0]
    v_center = uv_center[1]
    
    max_support_center = np.max(weight_support)
    
    
    conv_v_center = conv_kernel.shape[-1]//2
    conv_u_center = conv_kernel.shape[-2]//2
    
    conv_size = np.array(conv_kernel.shape[-2:])
    
    #print('conv_size',conv_size)
    
    #print('sizes ',conv_kernel.shape, conv_u_center, conv_v_center)
    
    #print(phase_gradient.shape)
    #print(weight_conv_kernel.shape)
    prev_field = -1
    
    for i_time in range(n_time):
        if prev_field != field[i_time]:
            conv_kernel_phase_gradient = conv_kernel*phase_gradient[field[i_time],:,:]
            prev_field = field[i_time]
        #conv_kernel_phase_gradient = conv_kernel
        
        for i_baseline in range(n_baseline):
            cf_baseline = cf_baseline_map[i_baseline]
            for i_chan in range(n_chan):
                cf_chan = cf_chan_map[i_chan]
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                
                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]
                    
                    #Do not use numpy round
                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)
                    
                    if (u_center_indx+max_support_center < n_u) and (v_center_indx+max_support_center < n_v) and (u_center_indx-max_support_center >= 0) and (v_center_indx-max_support_center >= 0):
                        u_offset = u_center_indx - u_pos
                        u_center_offset_indx = math.floor(u_offset * oversampling[0] + 0.5) + conv_u_center
                        v_offset = v_center_indx - v_pos
                        v_center_offset_indx = math.floor(v_offset * oversampling[1] + 0.5) + conv_v_center
                        
                        for i_pol in range(n_pol):
                            if do_psf:
                                weighted_data = imaging_weight[i_time, i_baseline, i_chan, i_pol]
                            else:
                                weighted_data = vis_data[i_time, i_baseline, i_chan, i_pol] * imaging_weight[i_time, i_baseline, i_chan, i_pol]
                                
                            if ~np.isnan(weighted_data) and (weighted_data != 0.0):
                                cf_pol = cf_pol_map[i_pol]
                                a_pol = pol_map[i_pol]
                                norm = 0.0
                                
                                '''
                                support = weight_support[cf_baseline,cf_chan,cf_pol,:]
                                #support = np.array([13,13])
                                support_center = support // 2
                                start_support = - support_center
                                end_support = support - support_center # end_support is larger by 1 so that python range() gives correct indices
                                '''
                                
                                support_u = weight_support[cf_baseline,cf_chan,cf_pol,0]
                                support_v = weight_support[cf_baseline,cf_chan,cf_pol,1]
                                
                                support_center_u = support_u // 2
                                support_center_v = support_v // 2
                                
                                start_support_u = - support_center_u
                                start_support_v = - support_center_v
                                
                                end_support_u = support_u - support_center_u
                                end_support_v = support_v - support_center_v
                                
                                #print(support)
                                ###############
#                                resized_conv_size = (support  + 1)*oversampling
#                                start_indx = conv_size//2 - resized_conv_size//2
#                                end_indx = start_indx + resized_conv_size
#                                normalize_factor = np.real(np.sum(conv_kernel[cf_baseline,cf_chan,cf_pol,start_indx[0]:end_indx[0],start_indx[1]:end_indx[1]])/(oversampling[0]*oversampling[1]))
#
#                                conv_kernel_phase_gradient = conv_kernel*phase_gradient[field[i_time],:,:]/normalize_factor
#                                print(normalize_factor)
                                ##############
                                
                                for i_v in range(start_support_v,end_support_v):
                                    v_indx = v_center_indx + i_v
                                    cf_v_indx = oversampling[1]*i_v + v_center_offset_indx

                                    for i_u in range(start_support_u,end_support_u):
                                        u_indx = u_center_indx + i_u
                                        cf_u_indx = oversampling[0]*i_u + u_center_offset_indx
                                        
                                        conv = conv_kernel_phase_gradient[cf_baseline,cf_chan,cf_pol,cf_u_indx,cf_v_indx]
                                        
                                        grid[a_chan, a_pol, u_indx, v_indx] = grid[a_chan, a_pol, u_indx, v_indx] +   conv * weighted_data
                                        norm = norm + conv
                            
                                sum_weight[a_chan, a_pol] = sum_weight[a_chan, a_pol] + imaging_weight[i_time, i_baseline, i_chan, i_pol]*np.real(norm**2)#*np.real(norm**2)#* np.real(norm) #np.abs(norm**2) #**2 term is needed since the pb is in the image twice (one naturally and another from the gcf)

    return



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
