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
    To do:
    Construct a Primary Beam cube containing a weighted sum of primary beams
    Option 1 : Evaluate models directly onto the image (for common PBs)
    Option 2 : Inverse FT each gridding convolution function (for varying PBs). Heterogeneous Arrays
    (A cube with 1 channel is a continuum image (nterms=1))
'''

def make_pb(img_xds,pb_parms, grid_parms, sel_parms):
    """
    The make_pb function currently supports rotationally symmetric airy disk primary beams. Primary beams can be generated for any number of dishes.
    The make_pb_parms['list_dish_diameters'] and make_pb_parms['list_blockage_diameters'] must be specified for each dish.
    
    Parameters
    ----------
    img_xds : xarray.core.dataset.Dataset
        Input image dataset.
    pb_parms : dictionary
    pb_parms['list_dish_diameters'] : list of number
        The list of dish diameters.
    pb_parms['list_blockage_diameters'] = list of number
        The list of blockage diameters for each dish.
    pb_parms['function'] : {'alma_airy','airy'}, default='alma_airy'
        Only the airy disk function is currently supported.
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
    sel_parms['pb'] = 'PB'
        The created PB name.
    Returns
    -------
    img_xds : xarray.core.dataset.Dataset
    """
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms
    from ._imaging_utils._check_imaging_parms import _check_pb_parms, _check_grid_parms
    import numpy as np
    import dask.array as da
    import copy, os
    import xarray as xr
    import matplotlib.pylab as plt
    
    print('######################### Start make_pb #########################')
    
    _img_xds = img_xds.copy(deep=True)
    _grid_parms = copy.deepcopy(grid_parms)
    _pb_parms =  copy.deepcopy(pb_parms)
    _sel_parms = copy.deepcopy(sel_parms)
    
    
    #Check img data_group
    _check_sel_parms(_img_xds,_sel_parms,new_or_modified_data_variables={'pb':'PB'},append_to_in_id=True)
    
    assert(_check_pb_parms(_img_xds,_pb_parms)), "######### ERROR: user_imaging_weights_parms checking failed"
    assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"
    
    #parameter check
    #cube continuum check
    
    if _pb_parms['function'] == 'airy':
        from ._imaging_utils._make_pb_symmetric import _airy_disk
        pb_func = _airy_disk
    elif _pb_parms['function'] == 'alma_airy':
        from ._imaging_utils._make_pb_symmetric import _alma_airy_disk
        pb_func = _alma_airy_disk
    else:
        print('Only the airy function has been implemented')
    
    _pb_parms['ipower'] = 2
    _pb_parms['center_indx'] = []


    chan_chunk_size = _img_xds.chan_width.chunks[0][0]
    freq_coords = da.from_array(_img_xds.coords['chan'].values, chunks=(chan_chunk_size))
    
    pol = _img_xds.pol.values #don't want chunking here

    chunksize = (_grid_parms['image_size'][0],_grid_parms['image_size'][1]) + freq_coords.chunksize + (len(pol),) + (len(_pb_parms['list_dish_diameters']),)
    
    pb = da.map_blocks(pb_func, freq_coords, pol, _pb_parms, _grid_parms, chunks=chunksize ,new_axis=[0,1,3,4], dtype=np.double)
    
    ## Add PB to img_xds
    
#    coords = {'d0': np.arange(pb_parms['imsize'][0]), 'd1': np.arange(_pb_parms['imsize'][1]),
#              'chan': freq_coords.compute(), 'pol': pol,'dish_type': np.arange(len(_pb_parms['list_dish_diameters']))}
    
    
    _img_xds[_sel_parms['data_group_out']['pb']] = xr.DataArray(pb[:,:,None,:,:,:], dims=['l', 'm', 'time', 'chan', 'pol','dish_type'])
    _img_xds = _img_xds.assign_coords({'dish_type': np.arange(len(_pb_parms['list_dish_diameters']))})
    _img_xds.attrs['data_groups'][0] = {**_img_xds.attrs['data_groups'][0],**{_sel_parms['data_group_out']['id']:_sel_parms['data_group_out']}}
    
    print('######################### Created graph for make_pb #########################')
    return _img_xds
    
