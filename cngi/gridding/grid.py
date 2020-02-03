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


def grid(vis_dataset, grid_parms):
    """
    Grids visibilities from Visibility Dataset.
    If to_disk is set to true the data is saved to disk.
    Parameters
    ----------
    vis_xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    grid_parms : dictionary
          keys ('chan_mode','imsize','cell','oversampling','support','to_disk','outfile')
    Returns
    -------
    grid_xds : xarray.core.dataset.Dataset

    """
    from numcodecs import Blosc
    import os
    from itertools import cycle
    import dask.array as da
    import xarray as xr
    import dask
    import time
    import copy

    # Parameter adjustments
    grid_parms_copy = copy.deepcopy(grid_parms)
    padding = 1.2  # Padding factor
    dtr = np.pi / (3600 * 180)
    grid_parms_copy['imsize'] = (padding * np.array(grid_parms_copy['imsize'])).astype(int)  # Add padding
    grid_parms_copy['cell'] = np.array(grid_parms_copy['cell']) * dtr
    grid_parms_copy['cell'][0] = -grid_parms_copy['cell'][0]

    assert grid_parms_copy['chan_mode'] == 'continuum' or grid_parms_copy[
        'chan_mode'] == 'cube', 'The chan_mode parameter in grid_parms can only be \'continuum\' or \'cube\'.'

    if grid_parms_copy['chan_mode'] == 'continuum':
        freq_coords = [da.mean(vis_dataset.coords['chan'].values)]
        imag_chan_chunk_size = 1
    elif grid_parms_copy['chan_mode'] == 'cube':
        freq_coords = vis_dataset.coords['chan'].values
        imag_chan_chunk_size = vis_dataset.DATA.chunks[2][0]

    # Create dask graph of gridding
    grids_and_sum_weights, correcting_cgk_image = _graph_grid(vis_dataset, grid_parms_copy)

    # Create delayed xarray dataset
    chunks = vis_dataset.DATA.chunks
    n_imag_pol = chunks[3][0]
    grid_dict = {}
    coords = {'chan': freq_coords, 'pol': np.arange(n_imag_pol), 'u': np.arange(grid_parms_copy['imsize'][0]),
              'v': np.arange(grid_parms_copy['imsize'][1])}
    grid_dict['CORRECTING_CGK'] = xr.DataArray(da.array(correcting_cgk_image), dims=['u', 'v'])
    # grid_dict['VIS_GRID'] = xr.DataArray(grids_and_sum_weights[0], dims=['chan','pol','u', 'v'])
    grid_dict['VIS_GRID'] = xr.DataArray(grids_and_sum_weights[0], dims=['u', 'v', 'chan', 'pol'])
    grid_dict['SUM_WEIGHT'] = xr.DataArray(grids_and_sum_weights[1], dims=['chan', 'pol'])
    grid_xds = xr.Dataset(grid_dict, coords=coords)

    if grid_parms['to_disk'] == True:
        outfile = grid_parms['outfile']
        tmp = os.system("rm -fr " + outfile)
        tmp = os.system("mkdir " + outfile)

        compressor = Blosc(cname='zstd', clevel=2, shuffle=0)
        encoding = dict(zip(list(grid_xds.data_vars), cycle([{'compressor': compressor}])))
        start = time.time()
        xr.Dataset.to_zarr(grid_xds, store=outfile, mode='w', encoding=encoding)
        grid_time = time.time() - start
        print('Grid time ', time.time() - start)

        grid_xds = xr.open_zarr(outfile)
        grid_xds.attrs['grid_time'] = grid_time
        return grid_xds

    else:
        return grid_xds


def _graph_grid(vis_dataset, grid_parms):
    import dask
    import dask.array as da
    import xarray as xr
    import time
    import itertools
    from cngi.gridding import gridding_convolutional_kernels as gck

    ##############################
    # Chunking on Polarization not supported add assert statement
    # Smeered poinst outside grid are ignored
    # Remove pol chunking option (pols are inhereantly connected). Should always be in same block
    ##############################

    # Creating gridding kernel
    # The support in CASA is defined as the half support, which is 3
    cgk, correcting_cgk_image = gck.create_prolate_spheroidal_kernel(grid_parms['oversampling'], grid_parms['support'],
                                                                     grid_parms['imsize'])
    cgk_1D = gck.create_prolate_spheroidal_kernel_1D(grid_parms['oversampling'], grid_parms['support'])

    # Getting data for gridding
    chan_chunk_size = vis_dataset.DATA.chunks[2][0]

    freq_chan = da.from_array(vis_dataset.coords['chan'].values, chunks=(chan_chunk_size))

    n_chunks_in_each_dim = vis_dataset.DATA.data.numblocks
    chunk_indx = []

    iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]),
                                         np.arange(n_chunks_in_each_dim[2]), np.arange(n_chunks_in_each_dim[3]))

    assert grid_parms['chan_mode'] == 'continuum' or grid_parms[
        'chan_mode'] == 'cube', 'The chan_mode parameter in grid_parms can only be \'continuum\' or \'cube\'.'

    if grid_parms['chan_mode'] == 'continuum':
        n_chan_chunks_img = 1
    elif grid_parms['chan_mode'] == 'cube':
        n_chan_chunks_img = n_chunks_in_each_dim[2]

    list_of_sub_grids_and_sum_weights = [[] for i in range(n_chan_chunks_img)]
    n_delayed = np.prod(n_chunks_in_each_dim)
    chunk_sizes = vis_dataset.DATA.chunks

    print(vis_dataset.UVW)
    # Build graph
    for c_time, c_baseline, c_chan, c_pol in iter_chunks_indx:
        sub_grid_and_sum_weights = dask.delayed(_standard_grid_dask)(
            vis_dataset.DATA.data.partitions[c_time, c_baseline, c_chan, c_pol],
            vis_dataset.UVW.data.partitions[c_time, c_baseline, 0],
            vis_dataset.WEIGHT.data.partitions[c_time, c_baseline, c_pol],
            vis_dataset.FLAG_ROW.data.partitions[c_time, c_baseline],
            vis_dataset.FLAG.data.partitions[c_time, c_baseline, c_chan, c_pol],
            freq_chan.partitions[c_chan],
            dask.delayed(cgk_1D), dask.delayed(grid_parms))
        if grid_parms['chan_mode'] == 'continuum':
            sub_grid_and_sum_weights = [da.from_delayed(sub_grid_and_sum_weights[0], (
            1, chunk_sizes[3][c_pol], grid_parms['imsize'][0], grid_parms['imsize'][1]), dtype=np.complex128),
                                        da.from_delayed(sub_grid_and_sum_weights[1], (1, chunk_sizes[3][c_pol]),
                                                        dtype=np.float64)]
            list_of_sub_grids_and_sum_weights[0].append(sub_grid_and_sum_weights)
        elif grid_parms['chan_mode'] == 'cube':
            sub_grid_and_sum_weights = [da.from_delayed(sub_grid_and_sum_weights[0], (
            chunk_sizes[2][c_chan], chunk_sizes[3][c_pol], grid_parms['imsize'][0], grid_parms['imsize'][1]),
                                                        dtype=np.complex128),
                                        da.from_delayed(sub_grid_and_sum_weights[1],
                                                        (chunk_sizes[2][c_chan], chunk_sizes[3][c_pol]),
                                                        dtype=np.float64)]
            list_of_sub_grids_and_sum_weights[c_chan].append(sub_grid_and_sum_weights)

    # Sum grids
    for c_chan in range(n_chan_chunks_img):
        while len(list_of_sub_grids_and_sum_weights[c_chan]) > 1:
            new_list_of_sub_grids_and_sum_weights = []
            for i in range(0, len(list_of_sub_grids_and_sum_weights[c_chan]), 2):
                if i < len(list_of_sub_grids_and_sum_weights[c_chan]) - 1:
                    lazy = [da.add(list_of_sub_grids_and_sum_weights[c_chan][i][0],
                                   list_of_sub_grids_and_sum_weights[c_chan][i + 1][0]),
                            da.add(list_of_sub_grids_and_sum_weights[c_chan][i][1],
                                   list_of_sub_grids_and_sum_weights[c_chan][i + 1][1])]
                else:
                    lazy = [list_of_sub_grids_and_sum_weights[c_chan][i][0],
                            list_of_sub_grids_and_sum_weights[c_chan][i][1]]
                new_list_of_sub_grids_and_sum_weights.append(lazy)
            list_of_sub_grids_and_sum_weights[c_chan] = new_list_of_sub_grids_and_sum_weights

            # Concatenate Cube
    if grid_parms['chan_mode'] == 'cube':
        while len(list_of_sub_grids_and_sum_weights) > 1:
            new_list_of_sub_grids_and_sum_weights = []
            for i in range(0, len(list_of_sub_grids_and_sum_weights), 2):
                if i < len(list_of_sub_grids_and_sum_weights) - 1:
                    lazy = [[da.concatenate(
                        [list_of_sub_grids_and_sum_weights[i][0][0], list_of_sub_grids_and_sum_weights[i + 1][0][0]],
                        axis=0), da.concatenate(
                        [list_of_sub_grids_and_sum_weights[i][0][1], list_of_sub_grids_and_sum_weights[i + 1][0][1]],
                        axis=0)]]
                else:
                    lazy = [[list_of_sub_grids_and_sum_weights[i][0][0], list_of_sub_grids_and_sum_weights[i][0][1]]]
                new_list_of_sub_grids_and_sum_weights.append(lazy)
            list_of_sub_grids_and_sum_weights = new_list_of_sub_grids_and_sum_weights

    # Put axes in image orientation. How much does this add to compute?
    list_of_sub_grids_and_sum_weights[0][0][0] = da.moveaxis(list_of_sub_grids_and_sum_weights[0][0][0], [-2, -1],
                                                             [0, 1])

    return list_of_sub_grids_and_sum_weights[0][0], correcting_cgk_image  # [0][0] for unwrapping


def _standard_grid_dask(vis_data, uvw, weight, flag_row, flag, freq_chan, cgk_1D, grid_parms):
    """
      Testing Function
      Wrapper function that is used when using dask distributed parallelism (blockwise function). 
      
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
      flag_row : boolean array
          (n_time, n_baseline)
      flag : boolean array
          (n_time, n_baseline, n_chan, n_pol)
      cgk_1D : float array
          (oversampling*(support//2 + 1))
      grid_parms : dictionary
          keys ('imsize','cell','oversampling','support')

      Returns
      -------
      grid : complex array
          (1,n_imag_chan,n_imag_pol,n_u,n_v)
      """

    n_chan = vis_data.shape[2]
    if grid_parms['chan_mode'] == 'cube':
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(np.int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(np.int)

    n_imag_pol = vis_data.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(np.int)

    n_uv = grid_parms['imsize']
    delta_lm = grid_parms['cell']
    oversampling = grid_parms['oversampling']
    support = grid_parms['support']

    grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)

    _standard_grid_jit(grid, sum_weight, vis_data, uvw, freq_chan, chan_map, pol_map, weight, flag_row, flag, cgk_1D,
                       n_uv, delta_lm, support, oversampling)

    return grid, sum_weight


@jit(nopython=True, cache=True, nogil=True)
def _standard_grid_jit(grid, sum_weight, vis_data, uvw, freq_chan, chan_map, pol_map, weight, flag_row, flag, cgk_1D,
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
      flag_row : boolean array 
          (n_time, n_baseline)
      flag : boolean array 
          (n_time, n_baseline, n_chan, n_pol)
      cgk_1D : float array 
          (oversampling*(support//2 + 1))
      grid_parms : dictionary 
          keys ('n_imag_chan','n_imag_pol','n_uv','delta_lm','oversampling','support')

      Returns
      -------
      """

    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = (freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = (freq_chan * delta_lm[1] * n_uv[1]) / c

    oversampling_center = int(oversampling // 2)
    support_center = int(support // 2)
    uv_center = n_uv // 2

    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(chan_map)
    n_pol = len(pol_map)
    n_imag_chan = chan_map.shape[0]

    n_u = n_uv[0]
    n_v = n_uv[1]

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            if flag_row[i_time, i_baseline] == 0:
                for i_chan in range(n_chan):
                    a_chan = chan_map[i_chan]

                    if a_chan >= 0 and a_chan < n_imag_chan:
                        u = -uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                        v = -uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                        u_pos = u + uv_center[0]
                        v_pos = v + uv_center[1]
                        u_center_indx = int(u_pos + 0.5)
                        v_center_indx = int(v_pos + 0.5)

                        if u_center_indx < n_u and v_center_indx < n_v and u_center_indx >= 0 and v_center_indx >= 0:
                            u_offset = u_center_indx - u_pos
                            u_center_offset_indx = math.floor(u_offset * oversampling + 0.5)
                            v_offset = v_center_indx - v_pos
                            v_center_offset_indx = math.floor(v_offset * oversampling + 0.5)

                            for i_pol in range(n_pol):
                                if weight[i_time, i_baseline, i_pol] != 0.0:
                                    a_pol = pol_map[i_pol]
                                    weigted_data = vis_data[i_time, i_baseline, i_chan, i_pol] * weight[
                                        i_time, i_baseline, i_pol]
                                    norm = 0.0

                                    if flag[i_time, i_baseline, i_chan, i_pol] == 0:
                                        for i_v in range(-support_center, support_center + 1):
                                            v_indx = v_center_indx + i_v
                                            v_offset_indx = np.abs(oversampling * i_v + v_center_offset_indx)
                                            conv_v = cgk_1D[v_offset_indx]

                                            for i_u in range(-support_center, support_center + 1):
                                                u_indx = u_center_indx + i_u
                                                u_offset_indx = np.abs(oversampling * i_u + u_center_offset_indx)
                                                conv_u = cgk_1D[u_offset_indx]
                                                conv = conv_u * conv_v

                                                grid[a_chan, a_pol, u_indx, v_indx] = grid[
                                                                                          a_chan, a_pol, u_indx, v_indx] + conv * weigted_data
                                                norm = norm + conv

                                    sum_weight[a_chan, a_pol] = sum_weight[a_chan, a_pol] + weight[
                                        i_time, i_baseline, i_pol] * norm

    return
