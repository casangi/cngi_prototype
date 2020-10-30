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
"""
this module will be included in the api
"""

#Removed for now.
#grid_parms['oversampling'] : int, default = 100
#    The oversampling used for the convolutional gridding kernel. This will be removed in a later release and incorporated in the function that creates gridding convolutional kernels.
#grid_parms['support'] : int, default = 7
#    The full support used for convolutional gridding kernel. This will be removed in a later release and incorporated in the function that creates gridding convolutional kernels.
#

def make_psf(vis_dataset, img_dataset, grid_parms, sel_parms, storage_parms):
    """
    Creates a cube or continuum point spread function (psf) image from the user specified uvw and imaging weight data. Only the prolate spheroidal convolutional gridding function is supported (this will change in a future releases.)
    
    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input visibility dataset.
    grid_parms : dictionary
    grid_parms['image_size'] : list of int, length = 2
        The image size (no padding).
    grid_parms['cell_size']  : list of number, length = 2, units = arcseconds
        The image cell size.
    grid_parms['chan_mode'] : {'continuum'/'cube'}, default = 'continuum'
        Create a continuum or cube image.
    grid_parms['fft_padding'] : number, acceptable range [1,100], default = 1.2
        The factor that determines how much the gridded visibilities are padded before the fft is done.
    sel_parms : dictionary
    sel_parms['uvw'] : str, default ='UVW'
        The name of uvw data variable that will be used to grid the visibilities.
    sel_parms['data'] : str, default = 'DATA'
        The name of the visibility data to be gridded.
    sel_parms['imaging_weight'] : str, default ='IMAGING_WEIGHT'
        The name of the imaging weights to be used.
    sel_parms['image'] : str, default ='DIRTY_IMAGE'
        The created image name.
    sel_parms['sum_weight'] : str, default ='SUM_WEIGHT'
        The created sum of weights name.
    storage_parms : dictionary
    storage_parms['to_disk'] : bool, default = False
        If true the dask graph is executed and saved to disk in the zarr format.
    storage_parms['append'] : bool, default = False
        If storage_parms['to_disk'] is True only the dask graph associated with the function is executed and the resulting data variables are saved to an existing zarr file on disk.
        Note that graphs on unrelated data to this function will not be executed or saved.
    storage_parms['outfile'] : str
        The zarr file to create or append to.
    storage_parms['chunks_on_disk'] : dict of int, default = {}
        The chunk size to use when writing to disk. This is ignored if storage_parms['append'] is True. The default will use the chunking of the input dataset.
    storage_parms['chunks_return'] : dict of int, default = {}
        The chunk size of the dataset that is returned. The default will use the chunking of the input dataset.
    storage_parms['graph_name'] : str
        The time to compute and save the data is stored in the attribute section of the dataset and storage_parms['graph_name'] is used in the label.
    storage_parms['compressor'] : numcodecs.blosc.Blosc,default=Blosc(cname='zstd', clevel=2, shuffle=0)
        The compression algorithm to use. Available compression algorithms can be found at https://numcodecs.readthedocs.io/en/stable/blosc.html.
    
    Returns
    -------
    image_dataset : xarray.core.dataset.Dataset
        The image_dataset will contain the image created and the sum of weights.
    """
    print('######################### Start make_psf #########################')
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
    
    from ngcasa._ngcasa_utils._store import _store
    from ngcasa._ngcasa_utils._check_parms import _check_storage_parms, _check_sel_parms, _check_existence_sel_parms
    from ._imaging_utils._check_imaging_parms import _check_grid_parms
    from ._imaging_utils._gridding_convolutional_kernels import _create_prolate_spheroidal_kernel, _create_prolate_spheroidal_kernel_1D
    from ._imaging_utils._standard_grid import _graph_standard_grid
    from ._imaging_utils._remove_padding import _remove_padding
    from ._imaging_utils._aperture_grid import _graph_aperture_grid
    
    _grid_parms = copy.deepcopy(grid_parms)
    _storage_parms = copy.deepcopy(storage_parms)
    _sel_parms = copy.deepcopy(sel_parms)
    
    assert(_check_sel_parms(_sel_parms,{'uvw':'UVW','imaging_weight':'IMAGING_WEIGHT','sum_weight':'SUM_WEIGHT_PSF','image':'PSF'})), "######### ERROR: sel_parms checking failed"
    assert(_check_existence_sel_parms(vis_dataset,{'uvw':_sel_parms['uvw'],'imaging_weight':_sel_parms['imaging_weight']})), "######### ERROR: sel_parms checking failed"
    assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"
    assert(_check_storage_parms(_storage_parms,'psf.img.zarr','make_psf')), "######### ERROR: storage_parms checking failed"
    
    # Creating gridding kernel
    _grid_parms['oversampling'] = 100
    _grid_parms['support'] = 7
    
    cgk, correcting_cgk_image = _create_prolate_spheroidal_kernel(_grid_parms['oversampling'], _grid_parms['support'], _grid_parms['image_size_padded'])
    cgk_1D = _create_prolate_spheroidal_kernel_1D(_grid_parms['oversampling'], _grid_parms['support'])
    
    _grid_parms['complex_grid'] = False
    _grid_parms['do_psf'] = True
    grids_and_sum_weights = _graph_standard_grid(vis_dataset, cgk_1D, _grid_parms, _sel_parms)
    uncorrected_dirty_image = dafft.fftshift(dafft.ifft2(dafft.ifftshift(grids_and_sum_weights[0], axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    
    #Remove Padding
    correcting_cgk_image = _remove_padding(correcting_cgk_image,_grid_parms['image_size'])
    uncorrected_dirty_image = _remove_padding(uncorrected_dirty_image,_grid_parms['image_size']).real * (_grid_parms['image_size_padded'][0] * _grid_parms['image_size_padded'][1])
    
    #############Normalize#############
    def correct_image(uncorrected_dirty_image, sum_weights, correcting_cgk):
        sum_weights_copy = copy.deepcopy(sum_weights) ##Don't mutate inputs, therefore do deep copy (https://docs.dask.org/en/latest/delayed-best-practices.html).
        sum_weights_copy[sum_weights_copy == 0] = 1
        # corrected_image = (uncorrected_dirty_image/sum_weights[:,:,None,None])/correcting_cgk[None,None,:,:]
        corrected_image = (uncorrected_dirty_image / sum_weights_copy) / correcting_cgk
        return corrected_image

    corrected_dirty_image = da.map_blocks(correct_image, uncorrected_dirty_image, grids_and_sum_weights[1][None, None, :, :],correcting_cgk_image[:, :, None, None])
    ####################################################

    if _grid_parms['chan_mode'] == 'continuum':
        freq_coords = [da.mean(vis_dataset.coords['chan'].values)]
        chan_width = da.from_array([da.mean(vis_dataset['chan_width'].data)],chunks=(1,))
        imag_chan_chunk_size = 1
    elif _grid_parms['chan_mode'] == 'cube':
        freq_coords = vis_dataset.coords['chan'].values
        chan_width = vis_dataset['chan_width'].data
        imag_chan_chunk_size = vis_dataset.DATA.chunks[2][0]
    
    ###Create Image Dataset
    chunks = vis_dataset.DATA.chunks
    n_imag_pol = chunks[3][0]
    
    coords = {'d0': np.arange(_grid_parms['image_size'][0]), 'd1': np.arange(_grid_parms['image_size'][1]),
              'chan': freq_coords, 'pol': np.arange(n_imag_pol), 'chan_width' : ('chan',chan_width)}
    img_dataset = img_dataset.assign_coords(coords)
    img_dataset[_sel_parms['sum_weight']] = xr.DataArray(grids_and_sum_weights[1], dims=['chan','pol'])
    img_dataset[_sel_parms['image']] = xr.DataArray(corrected_dirty_image, dims=['d0', 'd1', 'chan', 'pol'])
    
    
    list_xarray_data_variables = [img_dataset[_sel_parms['image']],img_dataset[_sel_parms['sum_weight']]]
    return _store(img_dataset,list_xarray_data_variables,_storage_parms)
    
    '''
    image_dict = {}
    coords = {'d0': np.arange(_grid_parms['imsize'][0]), 'd1': np.arange(_grid_parms['imsize'][1]),
              'chan': freq_coords, 'pol': np.arange(n_imag_pol), 'chan_width' : ('chan',chan_width)}
              
              
    image_dict[_sel_parms['sum_weight']] = xr.DataArray(grids_and_sum_weights[1], dims=['chan','pol'])
    image_dict[_sel_parms['image']] = xr.DataArray(corrected_dirty_image, dims=['d0', 'd1', 'chan', 'pol'])
    image_dataset = xr.Dataset(image_dict, coords=coords)
    
    list_xarray_data_variables = [image_dataset[_sel_parms['image']],image_dataset[_sel_parms['sum_weight']]]
    return _store(image_dataset,list_xarray_data_variables,_storage_parms)
    '''

