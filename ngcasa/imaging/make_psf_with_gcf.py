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

def make_psf_with_gcf(mxds, gcf_dataset, img_dataset, grid_parms, norm_parms, vis_sel_parms, img_sel_parms):
    """
    Creates a cube or continuum dirty image from the user specified visibility, uvw and imaging weight data. A gridding convolution function (gcf_dataset), primary beam image (img_dataset) and a primary beam weight image (img_dataset) must be supplied.
    
    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input visibility dataset.
    gcf_dataset : xarray.core.dataset.Dataset
         Input gridding convolution dataset.
    img_dataset : xarray.core.dataset.Dataset
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
    norm_parms : dictionary
    norm_parms['norm_type'] : {'none'/'flat_noise'/'flat_sky'}, default = 'flat_sky'
         Gridded (and FT'd) images represent the PB-weighted sky image.
         Qualitatively it can be approximated as two instances of the PB
         applied to the sky image (one naturally present in the data
         and one introduced during gridding via the convolution functions).
         normtype='flat_noise' : Divide the raw image by sqrt(sel_parms['weight_pb']) so that
                                             the input to the minor cycle represents the
                                             product of the sky and PB. The noise is 'flat'
                                             across the region covered by each PB.
        normtype='flat_sky' : Divide the raw image by sel_parms['weight_pb'] so that the input
                                         to the minor cycle represents only the sky.
                                         The noise is higher in the outer regions of the
                                         primary beam where the sensitivity is low.
        normtype='none' : No normalization after gridding and FFT.
    sel_parms : dictionary
    sel_parms['uvw'] : str, default ='UVW'
        The name of uvw data variable that will be used to grid the visibilities.
    sel_parms['data'] : str, default = 'DATA'
        The name of the visibility data to be gridded.
    sel_parms['imaging_weight'] : str, default ='IMAGING_WEIGHT'
        The name of the imaging weights to be used.
    sel_parms['image'] : str, default ='IMAGE'
        The created image name.
    sel_parms['sum_weight'] : str, default ='SUM_WEIGHT'
        The created sum of weights name.
    sel_parms['pb'] : str, default ='PB'
         The primary beam image to use for normalization.
    sel_parms['weight_pb'] : str, default ='WEIGHT_PB'
         The primary beam weight image to use for normalization.
    Returns
    -------
    image_dataset : xarray.core.dataset.Dataset
        The image_dataset will contain the image created and the sum of weights.
    """
    print('######################### Start make_psf_with_gcf #########################')
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
    from ._imaging_utils._check_imaging_parms import _check_grid_parms, _check_norm_parms
    #from ._imaging_utils._gridding_convolutional_kernels import _create_prolate_spheroidal_kernel, _create_prolate_spheroidal_kernel_1D
    from ._imaging_utils._standard_grid import _graph_standard_grid
    from ._imaging_utils._remove_padding import _remove_padding
    from ._imaging_utils._aperture_grid import _graph_aperture_grid
    from ._imaging_utils._normalize import _normalize
    from cngi.image import make_empty_sky_image
    from cngi.image import fit_gaussian
    
    
    #Deep copy so that inputs are not modified
    _mxds = mxds.copy(deep=True)
    _img_dataset = img_dataset.copy(deep=True)
    _vis_sel_parms = copy.deepcopy(vis_sel_parms)
    _img_sel_parms = copy.deepcopy(img_sel_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    _norm_parms = copy.deepcopy(norm_parms)
    
    ##############Parameter Checking and Set Defaults##############
    assert('xds' in _vis_sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_dataset = _mxds.attrs[_vis_sel_parms['xds']]
    
    assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"
    assert(_check_norm_parms(_norm_parms)), "######### ERROR: norm_parms checking failed"
    
    #Check vis data_group
    _check_sel_parms(_vis_dataset,_vis_sel_parms)
    
    #Check img data_group
 
    _check_sel_parms(_img_dataset,_img_sel_parms,new_or_modified_data_variables={'sum_weight':'PSF_SUM_WEIGHT','psf':'PSF','psf_fit':'PSF_FIT'},required_data_variables={'pb':'PB','weight_pb':'WEIGHT_PB'},append_to_in_id=False)
    #'pb':'PB','weight_pb':'WEIGHT_PB',
    #print('did this work',_img_sel_parms)
    
    
    _grid_parms['grid_weights'] = False
    _grid_parms['do_psf'] = True
    _grid_parms['oversampling'] = np.array(gcf_dataset.oversampling)

    grids_and_sum_weights = _graph_aperture_grid(_vis_dataset,gcf_dataset,_grid_parms,_vis_sel_parms)
    uncorrected_dirty_image = dafft.fftshift(dafft.ifft2(dafft.ifftshift(grids_and_sum_weights[0], axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        
    #Remove Padding
    #print('grid sizes',_grid_parms['image_size_padded'][0], _grid_parms['image_size_padded'][1])
    uncorrected_dirty_image = _remove_padding(uncorrected_dirty_image,_grid_parms['image_size']).real * (_grid_parms['image_size_padded'][0] * _grid_parms['image_size_padded'][1])
    
    #print(_img_sel_parms)
    normalized_image = _normalize(uncorrected_dirty_image, grids_and_sum_weights[1], img_dataset, gcf_dataset, 'forward', _norm_parms, _img_sel_parms)
    
    normalized_image = normalized_image/normalized_image[_grid_parms['image_center'][0],_grid_parms['image_center'][1],:,:]
    
    if _grid_parms['chan_mode'] == 'continuum':
        freq_coords = [da.mean(_vis_dataset.coords['chan'].values)]
        chan_width = da.from_array([da.mean(_vis_dataset['chan_width'].data)],chunks=(1,))
        imag_chan_chunk_size = 1
    elif _grid_parms['chan_mode'] == 'cube':
        freq_coords = _vis_dataset.coords['chan'].values
        chan_width = _vis_dataset['chan_width'].data
        imag_chan_chunk_size = _vis_dataset.DATA.chunks[2][0]
    
    ###Create Image Dataset
    chunks = _vis_dataset.DATA.chunks
    n_imag_pol = chunks[3][0]
    
    #coords = {'d0': np.arange(_grid_parms['image_size'][0]), 'd1': np.arange(_grid_parms['image_size'][1]),
    #          'chan': freq_coords, 'pol': np.arange(n_imag_pol), 'chan_width' : ('chan',chan_width)}
    #img_dataset = img_dataset.assign_coords(coords)
    #img_dataset[_sel_parms['sum_weight']] = xr.DataArray(grids_and_sum_weights[1], dims=['chan','pol'])
    #img_dataset[_sel_parms['image']] = xr.DataArray(normalized_image, dims=['d0', 'd1', 'chan', 'pol'])
    
    
    phase_center = _grid_parms['phase_center']
    image_size = _grid_parms['image_size']
    cell_size = _grid_parms['cell_size']
    phase_center = _grid_parms['phase_center']

    pol_coords = _vis_dataset.pol.data
    time_coords = [_vis_dataset.time.mean().data]
    
    _img_dataset = make_empty_sky_image(_img_dataset,phase_center,image_size,cell_size,freq_coords,chan_width,pol_coords,time_coords)
    
    _img_dataset[_img_sel_parms['data_group_out']['sum_weight']] = xr.DataArray(grids_and_sum_weights[1][None,:,:], dims=['time','chan','pol'])
    _img_dataset[_img_sel_parms['data_group_out']['psf']] = xr.DataArray(normalized_image[:,:,None,:,:], dims=['l', 'm', 'time', 'chan', 'pol'])
    _img_dataset.attrs['data_groups'][0] = {**_img_dataset.attrs['data_groups'][0],**{_img_sel_parms['data_group_out']['id']:_img_sel_parms['data_group_out']}}
    
    #list_xarray_data_variables = [img_dataset[_sel_parms['image']],img_dataset[_sel_parms['sum_weight']]]
    #return _store(img_dataset,list_xarray_data_variables,_storage_parms)
    _img_dataset = fit_gaussian(_img_dataset,dv=_img_sel_parms['data_group_out']['psf'],beam_set_name=_img_sel_parms['data_group_out']['psf_fit'])

    print('#########################  Created graph for make_psf_with_gcf #########################')
    return _img_dataset
    
    
    
    
    '''
    ###Create Dataset
    chunks = _vis_dataset.DATA.chunks
    n_imag_pol = chunks[3][0]
    image_dict = {}
    coords = {'d0': np.arange(_grid_parms['image_size'][0]), 'd1': np.arange(_grid_parms['image_size'][1]),
              'chan': freq_coords, 'pol': np.arange(n_imag_pol), 'chan_width' : ('chan',chan_width)}
              
    image_dict[_sel_parms['sum_weight']] = xr.DataArray(grids_and_sum_weights[1], dims=['chan','pol'])
    image_dict[_sel_parms['image']] = xr.DataArray(normalized_image, dims=['d0', 'd1', 'chan', 'pol'])
    image_dataset = xr.Dataset(image_dict, coords=coords)
    
    list_xarray_data_variables = [image_dataset[_sel_parms['image']],image_dataset[_sel_parms['sum_weight']]]
    return _store(image_dataset,list_xarray_data_variables,_storage_parms)
    '''
