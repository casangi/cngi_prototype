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

#Removed for now.
#grid_parms['oversampling'] : int, default = 100
#    The oversampling used for the convolutional gridding kernel. This will be removed in a later release and incorporated in the function that creates gridding convolutional kernels.
#grid_parms['support'] : int, default = 7
#    The full support used for convolutional gridding kernel. This will be removed in a later release and incorporated in the function that creates gridding convolutional kernels.
#

def make_grid(vis_mxds, img_xds, grid_parms,  vis_sel_parms, img_sel_parms):
    """

    
    Parameters
    ----------
    vis_mxds : xarray.core.dataset.Dataset
        Input multi-xarray Dataset with global data.
    img_xds : xarray.core.dataset.Dataset
        Input image dataset.
    grid_parms : dictionary
    grid_parms['image_size'] : list of int, length = 2
        The image size (no padding).
    grid_parms['cell_size']  : list of number, length = 2, units = arcseconds
        The image cell size.
    grid_parms['chan_mode'] : {'continuum'/'cube'}, default = 'continuum'
        Create a continuum or cube image.
    grid_parms['fft_padding'] : number, acceptable range [1,100], default = 1.2
        The factor that determines how much the gridded visibilities are padded before the fft is done.
    vis_sel_parms : dictionary
    vis_sel_parms['xds'] : str
        The xds within the mxds to use to calculate the imaging weights for.
    vis_sel_parms['data_group_in_id'] : int, default = first id in xds.data_groups
        The data group in the xds to use.
    img_sel_parms : dictionary
    img_sel_parms['data_group_in_id'] : int, default = first id in xds.data_groups
        The data group in the image xds to use.
    img_sel_parms['image'] : str, default ='IMAGE'
        The created image name.
    img_sel_parms['sum_weight'] : str, default ='SUM_WEIGHT'
        The created sum of weights name.
    Returns
    -------
    img_xds : xarray.core.dataset.Dataset
        The image_dataset will contain the image created and the sum of weights.
    """
    print('######################### Start make_image #########################')
    import numpy as np
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
    from ._imaging_utils._check_imaging_parms import _check_grid_parms
    from ._imaging_utils._gridding_convolutional_kernels import _create_prolate_spheroidal_kernel, _create_prolate_spheroidal_kernel_1D
    from ._imaging_utils._standard_grid import _graph_standard_grid
    from ._imaging_utils._remove_padding import _remove_padding
    from ._imaging_utils._aperture_grid import _graph_aperture_grid
    from cngi.image import make_empty_sky_image
    
    #print('****',sel_parms,'****')
    _mxds = vis_mxds.copy(deep=True)
    _img_xds = img_xds.copy(deep=True)
    _vis_sel_parms = copy.deepcopy(vis_sel_parms)
    _img_sel_parms = copy.deepcopy(img_sel_parms)
    _grid_parms = copy.deepcopy(grid_parms)

    ##############Parameter Checking and Set Defaults##############
    assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"
    assert('xds' in _vis_sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_xds = _mxds.attrs[_vis_sel_parms['xds']]
    
    #Check vis data_group
    _check_sel_parms(_vis_xds,_vis_sel_parms)
    
    #Check img data_group
    _check_sel_parms(_img_xds,_img_sel_parms,new_or_modified_data_variables={'sum_weight':'SUM_WEIGHT','grid':'GRID'},append_to_in_id=True)

    ##################################################################################
    
    # Creating gridding kernel
    _grid_parms['oversampling'] = 100
    _grid_parms['support'] = 7
    
    cgk, correcting_cgk_image = _create_prolate_spheroidal_kernel(_grid_parms['oversampling'], _grid_parms['support'], _grid_parms['image_size_padded'])
    cgk_1D = _create_prolate_spheroidal_kernel_1D(_grid_parms['oversampling'], _grid_parms['support'])
    
    _grid_parms['complex_grid'] = True
    _grid_parms['do_psf'] = False
    grids_and_sum_weights = _graph_standard_grid(_vis_xds, cgk_1D, _grid_parms, _vis_sel_parms)
    
    
    if _grid_parms['chan_mode'] == 'continuum':
        freq_coords = [da.mean(_vis_xds.coords['chan'].values)]
        chan_width = da.from_array([da.mean(_vis_xds['chan_width'].data)],chunks=(1,))
        imag_chan_chunk_size = 1
    elif _grid_parms['chan_mode'] == 'cube':
        freq_coords = _vis_xds.coords['chan'].values
        chan_width = _vis_xds['chan_width'].data
        imag_chan_chunk_size = _vis_xds.DATA.chunks[2][0]
    
    phase_center = _grid_parms['phase_center']
    image_size = _grid_parms['image_size']
    cell_size = _grid_parms['cell_size']
    phase_center = _grid_parms['phase_center']
    
    pol_coords = _vis_xds.pol.data
    time_coords = [_vis_xds.time.mean().data]
    
    _img_xds = make_empty_sky_image(_img_xds,grid_parms['phase_center'],image_size,cell_size,freq_coords,chan_width,pol_coords,time_coords)
    
    _img_xds[_img_sel_parms['data_group_out']['sum_weight']] = xr.DataArray(grids_and_sum_weights[1][None,:,:], dims=['time','chan','pol'])
    _img_xds[_img_sel_parms['data_group_out']['grid']] = xr.DataArray(grids_and_sum_weights[0][:,:,None,:,:], dims=['u', 'v', 'time', 'chan', 'pol'])
    _img_xds.attrs['data_groups'][0] = {**_img_xds.attrs['data_groups'][0],**{_img_sel_parms['data_group_out']['id']:_img_sel_parms['data_group_out']}}
    
    
    print('######################### Created graph for make_image #########################')
    return _img_xds

