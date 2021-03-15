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

'''
    The weight terms have no ps term, therefore no division by the ps_image is required.

    To do:
    Construct a Primary Beam cube containing a weighted sum of primary beams
    Option 1 : Evaluate models directly onto the image (for common PBs)
    Option 2 : Inverse FT each gridding convolution function (for varying PBs). Heterogeneous Arrays
    (A cube with 1 channel is a continuum image (nterms=1))
'''

def make_mosaic_pb(mxds,gcf_dataset,img_dataset,vis_sel_parms,img_sel_parms,grid_parms):
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
    vis_sel_parms : dictionary
    vis_sel_parms['xds'] : str
        The xds within the mxds to use to calculate the imaging weights for.
    vis_sel_parms['data_group_in_id'] : int, default = first id in xds.data_groups
        The data group in the xds to use.
    img_sel_parms : dictionary
    img_sel_parms['data_group_in_id'] : int, default = first id in xds.data_groups
        The data group in the image xds to use.
    img_sel_parms['pb'] : str, default ='PB'
        The mosaic primary beam.
    img_sel_parms['weight_pb'] : str, default ='WEIGHT_PB'
        The weight image.
    img_sel_parms['weight_pb_sum_weight'] : str, default ='WEIGHT_PB_SUM_WEIGHT'
        The sum of weight calculated when gridding the gcfs to create the weight image.
    Returns
    -------
    img_xds : xarray.core.dataset.Dataset
    """
    print('######################### Start make_mosaic_pb #########################')
    
    #from ngcasa._ngcasa_utils._store import _store
    #from ngcasa._ngcasa_utils._check_parms import _check_storage_parms, _check_sel_parms, _check_existence_sel_parms
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms
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
    from cngi.image import make_empty_sky_image
    import dask
    
    #Deep copy so that inputs are not modified
    _mxds = mxds.copy(deep=True)
    _img_dataset = img_dataset.copy(deep=True)
    _vis_sel_parms = copy.deepcopy(vis_sel_parms)
    _img_sel_parms = copy.deepcopy(img_sel_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    
    ##############Parameter Checking and Set Defaults##############
    assert('xds' in _vis_sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_dataset = _mxds.attrs[_vis_sel_parms['xds']]
    
    assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"

    #Check vis data_group
    _check_sel_parms(_vis_dataset,_vis_sel_parms)
    #print(_vis_sel_parms)
    
    #Check img data_group
    _check_sel_parms(_img_dataset,_img_sel_parms,new_or_modified_data_variables={'pb':'PB','weight_pb':'WEIGHT_PB','weight_pb_sum_weight':'WEIGHT_PB_SUM_WEIGHT'},append_to_in_id=True)
    #print('did this work',_img_sel_parms)
    
    _grid_parms['grid_weights'] = True
    _grid_parms['do_psf'] = False
    #_grid_parms['image_size_padded'] = _grid_parms['image_size']
    _grid_parms['oversampling'] = np.array(gcf_dataset.attrs['oversampling'])
    grids_and_sum_weights = _graph_aperture_grid(_vis_dataset,gcf_dataset,_grid_parms,_vis_sel_parms)
    

    #grids_and_sum_weights = _graph_aperture_grid(_vis_dataset,gcf_dataset,_grid_parms)
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
        freq_coords = [da.mean(_vis_dataset.coords['chan'].values)]
        chan_width = da.from_array([da.mean(_vis_dataset['chan_width'].data)],chunks=(1,))
        imag_chan_chunk_size = 1
    elif _grid_parms['chan_mode'] == 'cube':
        freq_coords = _vis_dataset.coords['chan'].values
        chan_width = _vis_dataset['chan_width'].data
        imag_chan_chunk_size = _vis_dataset.DATA.chunks[2][0]
        
    phase_center = _grid_parms['phase_center']
    image_size = _grid_parms['image_size']
    cell_size = _grid_parms['cell_size']
    phase_center = _grid_parms['phase_center']

    pol_coords = _vis_dataset.pol.data
    time_coords = [_vis_dataset.time.mean().data]
    
    _img_dataset = make_empty_sky_image(_img_dataset,phase_center,image_size,cell_size,freq_coords,chan_width,pol_coords,time_coords)
    
    
    _img_dataset[_img_sel_parms['data_group_out']['pb']] = xr.DataArray(mosaic_primary_beam[:,:,None,:,:], dims=['l', 'm', 'time', 'chan', 'pol'])
    _img_dataset[_img_sel_parms['data_group_out']['weight_pb']] = xr.DataArray(weight_image[:,:,None,:,:], dims=['l', 'm', 'time', 'chan', 'pol'])
    _img_dataset[_img_sel_parms['data_group_out']['weight_pb_sum_weight']] = xr.DataArray(grids_and_sum_weights[1][None,:,:], dims=['time','chan', 'pol'])
    _img_dataset.attrs['data_groups'][0] = {**_img_dataset.attrs['data_groups'][0],**{_img_sel_parms['data_group_out']['id']:_img_sel_parms['data_group_out']}}
    
    
    #list_xarray_data_variables = [_img_dataset[_sel_parms['pb']],_img_dataset[_sel_parms['weight']]]
    #return _store(_img_dataset,list_xarray_data_variables,_storage_parms)
    
    print('#########################  Created graph for make_mosaic_pb #########################')
    return _img_dataset
    
    
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
