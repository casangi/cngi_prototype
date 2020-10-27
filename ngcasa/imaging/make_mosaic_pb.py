#   Copyright 2020 AUI, Inc. Washington DC, USA
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

'''
    The weight terms have no ps term, therefore no division by the ps_image is required.

    To do:
    Construct a Primary Beam cube containing a weighted sum of primary beams
    Option 1 : Evaluate models directly onto the image (for common PBs)
    Option 2 : Inverse FT each gridding convolution function (for varying PBs). Heterogeneous Arrays
    (A cube with 1 channel is a continuum image (nterms=1))
'''

def make_mosaic_pb(vis_dataset,gcf_dataset,img_dataset,sel_parms,grid_parms,storage_parms):
    """
    The make_pb function currently supports rotationally symmetric airy disk primary beams. Primary beams can be generated for any number of dishes.
    The make_pb_parms['list_dish_diameters'] and make_pb_parms['list_blockage_diameters'] must be specified for each dish.
    
    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input visibility dataset.
    gcf_dataset : xarray.core.dataset.Dataset
        Input gridding convolution function dataset.
    img_dataset : xarray.core.dataset.Dataset
        Input image dataset. ()
    make_pb_parms : dictionary
    make_pb_parms['function'] : {'airy'}, default='airy'
        Only the airy disk function is currently supported.
    grid_parms['imsize'] : list of int, length = 2
        The image size (no padding).
    grid_parms['cell']  : list of number, length = 2, units = arcseconds
        The image cell size.
    make_pb_parms['list_dish_diameters'] : list of number
        The list of dish diameters.
    make_pb_parms['list_blockage_diameters'] = list of number
        The list of blockage diameters for each dish.
    make_pb_parms['pb_name'] = 'PB'
        The created PB name.
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
    img_xds : xarray.core.dataset.Dataset
    """
    print('######################### Start make_mosaic_pb #########################')
    
    from ngcasa._ngcasa_utils._store import _store
    from ngcasa._ngcasa_utils._check_parms import _check_storage_parms, _check_sel_parms, _check_existence_sel_parms
    from ._imaging_utils._check_imaging_parms import _check_grid_parms, _check_mosaic_pb_parms
    from ._imaging_utils._aperture_grid import _graph_aperture_grid
    import dask.array.fft as dafft
    import matplotlib.pylab as plt
    import numpy as np
    import dask.array as da
    import copy
    import xarray as xr
    from ._imaging_utils._remove_padding import _remove_padding
    from ._imaging_utils._normalize import _normalize
    
    _sel_parms = copy.deepcopy(sel_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    _storage_parms = copy.deepcopy(storage_parms)
    
    assert(_check_sel_parms(_sel_parms,{'pb':'PB','weight_pb':'WEIGHT_PB','weight_pb_sum_weight':'WEIGHT_PB_SUM_WEIGHT'})), "######### ERROR: sel_parms checking failed"
    assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"
    assert(_check_storage_parms(_storage_parms,'mosaic_pb.img.zarr','make_mosaic_pb')), "######### ERROR: storage_parms checking failed"
    
    
    _grid_parms['grid_weights'] = True
    _grid_parms['do_psf'] = False
    #_grid_parms['image_size_padded'] = _grid_parms['image_size']
    _grid_parms['oversampling'] = np.array(gcf_dataset.attrs['oversampling'])
    grids_and_sum_weights = _graph_aperture_grid(vis_dataset,gcf_dataset,_grid_parms,_sel_parms)
    

    #grids_and_sum_weights = _graph_aperture_grid(vis_dataset,gcf_dataset,_grid_parms)
    weight_image = _remove_padding(dafft.fftshift(dafft.ifft2(dafft.ifftshift(grids_and_sum_weights[0], axes=(0, 1)), axes=(0, 1)), axes=(0, 1)),_grid_parms['image_size']).real*(_grid_parms['image_size_padded'][0] * _grid_parms['image_size_padded'][1])
    
    #############Move this to Normalizer#############
    def correct_image(weight_image, sum_weights):
        sum_weights_copy = copy.deepcopy(sum_weights) ##Don't mutate inputs, therefore do deep copy (https://docs.dask.org/en/latest/delayed-best-practices.html).
        sum_weights_copy[sum_weights_copy == 0] = 1
        weight_image = (weight_image / sum_weights_copy[None, None, :, :])
        return weight_image

    weight_image = da.map_blocks(correct_image, weight_image, grids_and_sum_weights[1],dtype=np.double)
    mosaic_primary_beam = da.sqrt(np.abs(weight_image))
    
    if _grid_parms['chan_mode'] == 'continuum':
        freq_coords = [da.mean(vis_dataset.coords['chan'].values)]
        chan_width = da.from_array([da.mean(vis_dataset['chan_width'].data)],chunks=(1,))
        imag_chan_chunk_size = 1
    elif _grid_parms['chan_mode'] == 'cube':
        freq_coords = vis_dataset.coords['chan'].values
        chan_width = vis_dataset['chan_width'].data
        imag_chan_chunk_size = vis_dataset.DATA.chunks[2][0]
    
    chunks = vis_dataset.DATA.chunks
    n_imag_pol = chunks[3][0]
    image_dict = {}
    coords = {'d0': np.arange(_grid_parms['image_size'][0]), 'd1': np.arange(_grid_parms['image_size'][1]),
              'chan': freq_coords, 'pol': np.arange(n_imag_pol), 'chan_width' : ('chan',chan_width)}
    img_dataset = img_dataset.assign_coords(coords)
    img_dataset[_sel_parms['pb']] = xr.DataArray(mosaic_primary_beam, dims=['d0', 'd1', 'chan', 'pol'])
    img_dataset[_sel_parms['weight']] = xr.DataArray(weight_image, dims=['d0', 'd1', 'chan', 'pol'])
    img_dataset[_sel_parms['weight_pb_sum_weight']] = xr.DataArray(grids_and_sum_weights[1], dims=['chan', 'pol'])
    
    list_xarray_data_variables = [img_dataset[_sel_parms['pb']],img_dataset[_sel_parms['weight']]]
    return _store(img_dataset,list_xarray_data_variables,_storage_parms)
    
    
    
    '''
    
    ## Add PB to img_dataset

    #coords = {'d0': np.arange(pb_parms['imsize'][0]), 'd1': np.arange(_pb_parms['imsize'][1]),
    #          'chan': freq_coords.compute(), 'pol': pol,'dish_type': np.arange(len(_pb_parms['list_dish_diameters']))}
    
    img_dataset[_pb_mosaic_parms['mosaic_weight_name']] = xr.DataArray(weight_image, dims=['d0', 'd1', 'chan', 'pol','dish_type'])
    img_dataset[_pb_mosaic_parms['mosaic_pb_name']] = xr.DataArray(, dims=['d0', 'd1', 'chan', 'pol','dish_type'])
    img_dataset = img_dataset.assign_coords({'dish_type': np.arange(len(_pb_parms['list_dish_diameters']))})

    list_xarray_data_variables = [img_dataset[_pb_parms['pb_name']]]
    return _store(img_dataset,list_xarray_data_variables,_storage_parms)



    from ngcasa._ngcasa_utils._store import _store
    from ngcasa._ngcasa_utils._check_parms import _check_storage_parms
    from ._imaging_utils._check_imaging_parms import _check_pb_parms
    import numpy as np
    import dask.array as da
    import copy, os
    import xarray as xr

    import matplotlib.pylab as plt

    _pb_parms =  copy.deepcopy(pb_parms)
    _storage_parms = copy.deepcopy(storage_parms)

    assert(_check_pb_parms(img_dataset,_pb_parms)), "######### ERROR: user_imaging_weights_parms checking failed"
    assert(_check_storage_parms(_storage_parms,'dataset.img.zarr','make_pb')), "######### ERROR: user_storage_parms checking failed"

    #parameter check
    #cube continuum check


    if _pb_parms['function'] == 'airy':
        from ._imaging_utils._make_pb_1d import _airy_disk
        pb_func = _airy_disk
    else:
        print('Only the airy function has been implemented')

    _pb_parms['ipower'] = 2
    _pb_parms['center_indx'] = []


    chan_chunk_size = img_dataset.chan_width.chunks[0][0]
    freq_coords = da.from_array(img_dataset.coords['chan'].values, chunks=(chan_chunk_size))

    pol = img_dataset.pol.values #don't want chunking here

    chunksize = (_pb_parms['imsize'][0],_pb_parms['imsize'][1]) + freq_coords.chunksize + (len(pol),) + (len(_pb_parms['list_dish_diameters']),)

    pb = da.map_blocks(pb_func, freq_coords, pol, _pb_parms, chunks=chunksize ,new_axis=[0,1,3,4], dtype=np.double)

    ## Add PB to img_dataset

    coords = {'d0': np.arange(pb_parms['imsize'][0]), 'd1': np.arange(_pb_parms['imsize'][1]),
              'chan': freq_coords.compute(), 'pol': pol,'dish_type': np.arange(len(_pb_parms['list_dish_diameters']))}


    img_dataset[_pb_parms['pb_name']] = xr.DataArray(pb, dims=['d0', 'd1', 'chan', 'pol','dish_type'])
    img_dataset = img_dataset.assign_coords({'dish_type': np.arange(len(_pb_parms['list_dish_diameters']))})

    list_xarray_data_variables = [img_dataset[_pb_parms['pb_name']]]
    return _store(img_dataset,list_xarray_data_variables,_storage_parms)
    '''
