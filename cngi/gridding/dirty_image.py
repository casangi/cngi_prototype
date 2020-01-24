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

import numpy as np
from numba import jit
import math


###### convolution kernel
def _coordinates(npixel: int):
    return (np.arange(npixel) - npixel // 2) / npixel


def _coordinates2(npixel: int):
    return (np.mgrid[0:npixel, 0:npixel] - npixel // 2) / npixel


def create_prolate_spheroidal_kernel(oversampling, support, n_uv):
    # support//2 is the index of the zero value of the support values
    # oversampling//2 is the index of the zero value of the oversampling value
    support_center = support // 2
    oversampling_center = oversampling // 2
    
    support_values = (np.arange(support) - support_center)
    if (oversampling % 2) == 0:
        oversampling_values = ((np.arange(oversampling + 1) - oversampling_center) / oversampling)[:, None]
        kernel_points_1D = (np.broadcast_to(support_values, (oversampling + 1, support)) + oversampling_values)
    else:
        oversampling_values = ((np.arange(oversampling) - oversampling_center) / oversampling)[:, None]
        kernel_points_1D = (np.broadcast_to(support_values, (oversampling, support)) + oversampling_values)
    
    kernel_points_1D = kernel_points_1D / support_center
    
    _, kernel_1D = prolate_spheroidal_function(kernel_points_1D)
    # kernel_1D /= np.sum(np.real(kernel_1D[oversampling_center,:]))
    
    if (oversampling % 2) == 0:
        kernel = np.zeros((oversampling + 1, oversampling + 1, support, support), dtype=np.double)  # dtype=np.complex128
    else:
        kernel = np.zeros((oversampling, oversampling, support, support), dtype=np.double)
    
    for x in range(oversampling):
        for y in range(oversampling):
            kernel[x, y, :, :] = np.outer(kernel_1D[x, :], kernel_1D[y, :])
    
    # norm = np.sum(np.real(kernel))
    # kernel /= norm
    
    # Gridding correction function (applied after dirty image is created)
    kernel_image_points_1D_u = np.abs(2.0 * _coordinates(n_uv[0]))
    kernel_image_1D_u = prolate_spheroidal_function(kernel_image_points_1D_u)[0]
    
    kernel_image_points_1D_v = np.abs(2.0 * _coordinates(n_uv[1]))
    kernel_image_1D_v = prolate_spheroidal_function(kernel_image_points_1D_v)[0]
    
    kernel_image = np.outer(kernel_image_1D_u, kernel_image_1D_v)
    
    # kernel_image[kernel_image > 0.0] = kernel_image.max() / kernel_image[kernel_image > 0.0]
    
    # kernel_image =  kernel_image/kernel_image.max()
    return kernel, kernel_image


def prolate_spheroidal_function(u):
    """
    Calculate PSWF using an old SDE routine re-written in Python

    Find Spheroidal function with M = 6, alpha = 1 using the rational
    approximations discussed by Fred Schwab in 'Indirect Imaging'.

    This routine was checked against Fred's SPHFN routine, and agreed
    to about the 7th significant digit.

    The griddata function is (1-NU**2)*GRDSF(NU) where NU is the distance
    to the edge. The grid correction function is just 1/GRDSF(NU) where NU
    is now the distance to the edge of the image.
    """
    p = np.array([[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
                  [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    q = np.array([[1.0000000e0, 8.212018e-1, 2.078043e-1], [1.0000000e0, 9.599102e-1, 2.918724e-1]])
    
    _, n_p = p.shape
    _, n_q = q.shape
    
    u = np.abs(u)
    uend = np.zeros(u.shape, dtype=np.float64)
    part = np.zeros(u.shape, dtype=np.int64)
    
    part[(u >= 0.0) & (u < 0.75)] = 0
    part[(u >= 0.75) & (u <= 1.0)] = 1
    uend[(u >= 0.0) & (u < 0.75)] = 0.75
    uend[(u >= 0.75) & (u <= 1.0)] = 1.0
    
    delusq = u ** 2 - uend ** 2
    
    top = p[part, 0]
    for k in range(1, n_p):  # small constant size loop
        top += p[part, k] * np.power(delusq, k)
    
    bot = q[part, 0]
    for k in range(1, n_q):  # small constant size loop
        bot += q[part, k] * np.power(delusq, k)
    
    grdsf = np.zeros(u.shape, dtype=np.float64)
    ok = (bot > 0.0)
    grdsf[ok] = top[ok] / bot[ok]
    ok = np.abs(u > 1.0)
    grdsf[ok] = 0.0
    
    # Return the griddata function and the grid correction function
    return grdsf, (1 - u ** 2) * grdsf


def create_prolate_spheroidal_kernel_1D(oversampling, support):
    support_center = support // 2
    u = np.arange(oversampling * support_center) / (support_center * oversampling)
    
    long_half_kernel_1D = np.zeros(oversampling * (support_center + 1))
    _, long_half_kernel_1D[0:oversampling * support_center] = prolate_spheroidal_function(u)
    return long_half_kernel_1D



##############################
@jit(nopython=True, cache=True)
def _standard_grid_jit(grid, sum_weight, grid_data, uvw, freq_chan, chan_map, pol_map, weight, flag_row, flag, cgk_1D, n_uv, delta_lm,
                       support, oversampling):
    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = (freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = (freq_chan * delta_lm[1] * n_uv[1]) / c
    
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
            if flag_row[i_time, i_baseline] == 0 and np.sum(weight[i_time, i_baseline, :]) != 0.0:
                for i_chan in range(n_chan):
                    a_chan = chan_map[i_chan]
                    
                    if 0 <= a_chan < n_imag_chan:
                        u = -uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                        v = -uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                        u_pos = u + uv_center[0]
                        v_pos = v + uv_center[1]
                        u_center_indx = int(u_pos + 0.5)
                        v_center_indx = int(v_pos + 0.5)
                        
                        if (n_u > u_center_indx >= 0) and (n_v > v_center_indx >= 0):
                            u_offset = u_center_indx - u_pos
                            u_center_offset_indx = math.floor(u_offset * oversampling + 0.5)
                            v_offset = v_center_indx - v_pos
                            v_center_offset_indx = math.floor(v_offset * oversampling + 0.5)
                            
                            for i_pol in range(n_pol):
                                a_pol = pol_map[i_pol]
                                weigted_data = grid_data[i_time, i_baseline, i_chan, i_pol] * weight[i_time, i_baseline, i_pol]
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
                                            
                                            grid[a_chan, a_pol, u_indx, v_indx] = grid[a_chan, a_pol, u_indx, v_indx] + conv * weigted_data
                                            norm = norm + conv
                                
                                sum_weight[a_chan, a_pol] = sum_weight[a_chan, a_pol] + weight[i_time, i_baseline, i_pol] * norm
    
    return


def _convert_sum_weight_to_sparse(sum_weight, n_chan, n_pol, n_uv):
    import sparse
    
    sum_weight_coords = np.zeros((6, n_chan * n_pol), dtype=int)
    sum_weight_data = np.zeros((n_chan * n_pol), dtype=np.float64)
    flat_indx = 0
    for i_chan in range(n_chan):
        for i_pol in range(n_pol):
            sum_weight_coords[:, flat_indx] = np.array([0, 0, i_chan, i_pol, 0, 0])
            sum_weight_data[flat_indx] = sum_weight[i_chan, i_pol]
            flat_indx = flat_indx + 1
    
    return sparse.COO(sum_weight_coords, sum_weight_data, shape=(1, 1, n_chan, n_pol, n_uv[0], n_uv[1]))


def serial_grid_dask_sparse(grid_data, uvw, weight, flag_row, flag, freq_chan, chan_map, pol_map, cgk_1D,
                            n_imag_chan, n_imag_pol, n_uv, delta_lm, oversampling, support):
    import sparse
    grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)
    
    _standard_grid_jit(grid, sum_weight, grid_data[0][0][0], uvw[0][0], freq_chan, chan_map, pol_map, weight[0][0], flag_row[0],
                       flag[0][0][0], cgk_1D, n_uv, delta_lm, support, oversampling)
    sum_weight = _convert_sum_weight_to_sparse(sum_weight, n_imag_chan, n_imag_pol, n_uv)
    
    grid = sparse.COO(grid[None, None, :, :, :, :])  # First None for Dask packing, second None for switching between grid and sum_weight
    grid_and_sum_weight = sparse.concatenate((grid, sum_weight), axis=1)
    
    return grid_and_sum_weight



#####################################################################################################
# noinspection PyTypeChecker
def dirty_image(xds, field=None, imsize=[200,200], cell=[0.08, 0.08], nchan=1):
    """
    Grids visibilities from Visibility Dataset and returns dirty Image Dataset

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    field : str or list
        field in dataset to grid.  Default None means all
    imsize : list of ints
        number of pixels for each spatial dimension. Default [200, 200]
    nchan : int
        number of channels in the output image. Default is 1
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image Dataset
    """
    import xarray as xr
    import tests.gridding_convolutional_kernels as gck
    import dask.array as da
    import dask.array.fft as dafft
    import time
    import sparse
    from scipy.signal import resample
    
    # subselect field ID
    txds = xds
    if field is not None:
        txds = xds.where(xds.field.isin(np.atleast_1d(field)), drop=True)
    
    # Gridding Parameters
    dtr = np.pi / (3600 * 180)  # Degrees to Radians
    n_xy = np.array(imsize)  # 2048                # 2 element array containing number of pixels for x (u and l) and y (v and m) axes.
    padding = 1.2  # Padding factor
    n_uv = (padding * n_xy).astype(int)  # 2 element array containg number of pixels with padding

    # Image domain cell size converted from arcseconds to radians (negative is due to axis direction being different from indx increments)
    delta_lm = np.array([-cell[0] * dtr, cell[1] * dtr])
    
    # Creating gridding kernel (same as CASA)
    support = 7  # The support in CASA is defined as the half support, which is 3
    oversampling = 100
    cgk_1D = gck.create_prolate_spheroidal_kernel_1D(oversampling, support)
    cgk, cgk_image = gck.create_prolate_spheroidal_kernel(oversampling, support, n_uv)

    # Casa averages the weights for different polarizations. See code/msvis/MSVis/VisImagingWeight.cc VisImagingWeight::unPolChanWeight
    weight_avg = xr.DataArray.expand_dims((txds.WEIGHT[:, :, 0] + txds.WEIGHT[:, :, 1]) / 2, dim=['n_pol'], axis=2)

    #
    n_imag_chan = nchan
    n_imag_pol = 1  # Imaging only one polarization
    n_chan = txds.dims['chan']
    n_pol = 1  # txds.dims['pol'] #just using the first polarization

    #chan_map = (np.zeros(n_chan)).astype(np.int)  # np.arange(0,n_chan) for cube, n_chan array which maps to the number of channels in image (cube)
    chan_map = np.repeat(np.arange(n_imag_chan), int(np.ceil(n_chan/n_imag_chan)))[:n_chan]
    pol_map = (np.zeros(n_pol)).astype(np.int)
    chan_freqs = txds.chan.values
    chan_map_freqs = resample(chan_freqs, n_imag_chan)

    start = time.time()
    grids_and_sum_weights = da.blockwise(serial_grid_dask_sparse, ("n_time", "n_switch", "n_imag_chan", "n_imag_pol", "n_u", "n_v"),
                                         txds.DATA, ("n_time", "n_baseline", "n_chan", "n_pol"),
                                         txds.UVW, ("n_time", "n_baseline", "uvw"),
                                         weight_avg, ("n_time", "n_baseline", "n_pol"),
                                         txds.FLAG_ROW, ("n_time", "n_baseline"),
                                         txds.FLAG, ("n_time", "n_baseline", "n_vis_chan", "n_vis_pol"),
                                         new_axes={"n_switch": 2, "n_imag_chan": n_imag_chan, "n_imag_pol": n_imag_pol,
                                                   "n_u": n_uv[0], "n_v": n_uv[1]},
                                         adjust_chunks={"n_time": 1},
                                         freq_chan=chan_freqs, chan_map=chan_map, pol_map=pol_map, cgk_1D=cgk_1D, n_imag_chan=n_imag_chan,
                                         n_imag_pol=n_imag_pol, n_uv=n_uv, delta_lm=delta_lm, oversampling=oversampling, support=support,
                                         dtype=complex)
    grid_and_sum_weight = grids_and_sum_weights.sum(axis=0)
    grid_and_sum_weight = grid_and_sum_weight.map_blocks(sparse.COO.todense, dtype='complex128')  # convert back to dense array

    #vis_grid = grid_and_sum_weight[0].map_blocks(sparse.COO.todense, dtype='complex128')   # convert back to dense array
    #sum_weight = grid_and_sum_weight[1].map_blocks(sparse.COO.todense, dtype='complex128')
    vis_grid = grid_and_sum_weight[0]
    sum_weight = grid_and_sum_weight[1]
    
    # Create Dirty Image and correct for gridding convolutional kernel
    uncorrected_dirty_image = dafft.fftshift(dafft.ifft2(dafft.ifftshift(vis_grid, axes=(2,3)), axes=(2,3)), axes=(2,3))
    uncorrected_dirty_image = uncorrected_dirty_image * ((n_uv[0] * n_uv[1]) / sum_weight.sum(axis=(2,3), keepdims=True))
    corrected_dirty_image = uncorrected_dirty_image.real / cgk_image
    
    # Remove Padding
    start_xy = (n_uv // 2 - n_xy // 2)
    end_xy = start_xy + n_xy
    corrected_dirty_image = corrected_dirty_image[:, :, start_xy[0]:end_xy[0], start_xy[1]:end_xy[1]]
    
    # put axes in image orientation
    corrected_dirty_image = da.moveaxis(corrected_dirty_image.transpose(), 2, -1)
    
    coords = dict(zip(['d0','d1','frequency', 'pol'], [list(range(ss)) for ss in corrected_dirty_image.shape]))
    coords['frequency'] = chan_map_freqs
    image_xds = xr.Dataset({'image':xr.DataArray(corrected_dirty_image, dims=['d0','d1','frequency', 'pol'])}, coords=coords)
    
    return image_xds
