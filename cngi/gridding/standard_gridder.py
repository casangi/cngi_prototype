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


def standard_grid(grid_data, uvw, weight, flag_row, flag, freq_chan, chan_map, n_chan, pol_map, cgk_1D, grid_parms):
    """
    Grids visibilities.

    Parameters
    ----------
    grid : complex array (n_chan, n_pol, n_u, n_v)
    sum_weight : float array (n_chan, n_pol)
    uvw  : float array (n_time, n_baseline, 3)
    freq_chan : float array (n_chan)
    chan_map : int array (n_chan)
    pol_map : int array (n_pol)
    weight : float array (n_time, n_baseline, n_vis_chan)
    flag_row : boolean array (n_time, n_baseline)
    flag : boolean array (n_time, n_baseline, n_chan, n_pol)
    cgk_1D : float array (oversampling*(support//2 + 1))
    grid_parms : dictionary ('n_imag_chan','n_imag_pol','n_uv','delta_lm','oversampling','support')
    Returns
    -------
    grid : complex array (n_imag_chan,n_imag_pol,n_u,n_v)
    sum_weight : float array (n_imag_chan,n_imag_pol)
    """
    import numpy as np
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_imag_chan = grid_parms['n_imag_chan']
    n_imag_pol = grid_parms['n_imag_pol']
    n_uv = grid_parms['n_uv']
    oversampling = grid_parms['oversampling']
    support = grid_parms['support']
    sum_weight = np.ones((n_imag_chan, n_imag_pol), dtype=np.double)
    c = 299792458.0

    # everything needs to be ordered by the same dimensions, too confusing otherwise
    # (time x baseline x channel x pol x support^2 x uv)
    # use singleton dimensions as placeholders (np.newaxis / None) so size doesn't change
    # this allows easy broadcasting for matrix math later on
    chunks = (200,100,1,1,50,-1)
    mask = da.from_array(flag_row[..., None,None,None,None] | flag[...,None,None], chunks=chunks)
    weighted_data = da.from_array(grid_data[...,None,None] * weight[...,None,:,None,None], chunks=chunks)
    support_idxs = (np.mgrid[0:support, 0:support] - support // 2).reshape(2,-1).T[None,None,None,None,...]
    
    uv_scale = da.from_array((freq_chan[:,None,None,None] * (grid_parms['delta_lm']*n_uv/c))[None,None,...], chunks=chunks)
    uv = da.from_array(-uvw[...,None,None,None,:2], chunks=chunks) * uv_scale
    uv_pos = uv + (n_uv // 2)
    
    uv_center_indx = (uv_pos + 0.5).astype(int)
    uv_indx = (support_idxs + uv_center_indx)
    uv_center_offset_indx = da.floor((uv_center_indx - uv_pos) * oversampling + 0.5).astype(int)
    uv_offset_indx = da.fabs(oversampling * support_idxs + uv_center_offset_indx).astype(int)
    
    # TESTING
    # weighted_data_c = weighted_data.blocks[0,0,100,0,0,0].compute()
    # uv_offset_indx_c = uv_offset_indx.blocks[0,0,100,0,0,0].compute()
    # uv_indx_c = uv_indx.blocks[0,0,100,0,0,0].compute()
    # mask_c = mask.blocks[0,0,100,0,0,0].compute()
    
    # map the gridding across chunks
    # noinspection PyTypeChecker
    def set_grid(weighted_data_c, uv_offset_indx_c, uv_indx_c, mask_c, support, n_uv):
        mask_c = np.where(mask_c.repeat(support, 4).ravel() == False)  # turn in to a filter
        conv = cgk_1D[uv_offset_indx_c.clip(0,len(cgk_1D)-1)].prod(axis=5, keepdims=True)
        gvals_1d = (conv * weighted_data_c).ravel()[mask_c]
        gcoords_1d = np.ravel_multi_index(uv_indx_c.reshape(-1, 2).T, n_uv, mode='clip')[mask_c]
        #norm = np.sum(conv)
        
        grid_chunk1d = np.bincount(gcoords_1d, np.real(gvals_1d), minlength=np.prod(n_uv))
        grid_chunk1d = grid_chunk1d + 1j * np.bincount(gcoords_1d, np.imag(gvals_1d), minlength=np.prod(n_uv))
        grid_chunk = grid_chunk1d.reshape(tuple(n_uv) + tuple([1 for _ in range(6)]))
        return grid_chunk
    
    
    da_grid = da.map_blocks(set_grid, weighted_data, uv_offset_indx, uv_indx, mask, support, n_uv,
                            dtype=weighted_data.dtype, chunks=tuple(n_uv)+weighted_data.chunksize)
    da_grid = da_grid.sum(axis=[2,3,6,7], keepdims=False)
    
    with ProgressBar():
        grid = da_grid.compute()
    
    grid = grid.sum(axis=3).sum(axis=2)
    
    #sum_weight[a_chan, a_pol] = sum_weight[a_chan, a_pol] + weight[i_time, i_baseline, i_pol] * norm
    #
    #
    #grid = grid / sum_weight[:, :, None, None]
    return grid, sum_weight
