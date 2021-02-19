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

def make_imaging_weight(vis_dataset, imaging_weights_parms, grid_parms, sel_parms):
    """
    Creates the imaging weight data variable that has dimensions time x baseline x chan x pol (matches the visibility data variable).
    The weight density can be averaged over channels or calculated independently for each channel using imaging_weights_parms['chan_mode'].
    The following imaging weighting schemes are supported 'natural', 'uniform', 'briggs', 'briggs_abs'.
    The grid_parms['image_size'] and grid_parms['cell_size'] should usually be the same values that will be used for subsequent synthesis blocks (for example making the psf).
    To achieve something similar to 'superuniform' weighting in CASA tclean grid_parms['image_size'] and imaging_weights_parms['cell_size'] can be varied relative to the values used in subsequent synthesis blocks.
    
    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input visibility dataset.
    imaging_weights_parms : dictionary
    imaging_weights_parms['weighting'] : {'natural', 'uniform', 'briggs', 'briggs_abs'}, default = natural
        Weighting scheme used for creating the imaging weights.
    imaging_weights_parms['robust'] : number, acceptable range [-2,2], default = 0.5
        Robustness parameter for Briggs weighting.
        robust = -2.0 maps to uniform weighting.
        robust = +2.0 maps to natural weighting.
    imaging_weights_parms['briggs_abs_noise'] : number, default=1.0
        Noise parameter for imaging_weights_parms['weighting']='briggs_abs' mode weighting.
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
    sel_parms['vis_description_in_indx'] : int, default = 0
        The index of the vis_dataset.vis_description dictionary to use.
    sel_parms['imaging_weight'] : str, default ='IMAGING_WEIGHT'
        The name of that will be used for the imaging weight data variable.
    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
        The vis_dataset will contain a new data variable for the imaging weights the name is defined by the input parameter sel_parms['imaging_weight'].
    """
    print('######################### Start make_imaging_weights #########################')
    import time
    import math
    import xarray as xr
    import dask.array as da
    import matplotlib.pylab as plt
    import dask.array.fft as dafft
    import dask
    import copy, os
    from numcodecs import Blosc
    from itertools import cycle
    import zarr
    
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms
    from ._imaging_utils._check_imaging_parms import _check_imaging_weights_parms, _check_grid_parms
    
    _vis_dataset = vis_dataset.copy(deep=True)
    
    _imaging_weights_parms =  copy.deepcopy(imaging_weights_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    _sel_parms = copy.deepcopy(sel_parms)
    
    assert(_check_imaging_weights_parms(_imaging_weights_parms)), "######### ERROR: imaging_weights_parms checking failed"
    if _imaging_weights_parms['weighting'] != 'natural':
        assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"
        
    if 'vis_description_in_indx' in _sel_parms:
        _sel_parms['vis_description_in'] = _vis_dataset.vis_description[sel_parms['vis_description_in_indx']]
    sel_defaults = {'vis_description_in':_vis_dataset.vis_description[0], 'imaging_weight':'IMAGING_WEIGHT'}
    assert(_check_sel_parms(_sel_parms,sel_defaults)), "######### ERROR: sel_parms checking failed"
    sel_check = {'vis_description_in':_sel_parms['vis_description_in']}
    assert(_check_existence_sel_parms(_vis_dataset,sel_check)), "######### ERROR: sel_parms checking failed"
    
    _sel_parms['vis_description_in']['imaging_weight'] =  _sel_parms['imaging_weight']
    _vis_dataset[_sel_parms['imaging_weight']] = _vis_dataset[_sel_parms['vis_description_in']['weight']]
    
    if _imaging_weights_parms['weighting'] != 'natural':
        calc_briggs_weights(_vis_dataset,_imaging_weights_parms,_grid_parms,_sel_parms['vis_description_in'])
        
    #print(_vis_dataset)
    _vis_dataset.vis_description[sel_parms['vis_description_in_indx']]['imaging_weight'] = _sel_parms['imaging_weight']
    
    
    print('######################### Created graph for make_imaging_weight #########################')
    return _vis_dataset
    
    
'''
def _match_array_shape(array_to_reshape,array_to_match):
    # Reshape in_weight to match dimnetionality of vis_data (vis_dataset[imaging_weights_parms['data_name']])
    # The order is assumed the same (there can be missing). array_to_reshape is a subset of array_to_match
    import dask.array as da
    import numpy as np
    
    match_array_chunksize = array_to_match.data.chunksize
    
    reshape_dims = np.ones(len(match_array_chunksize),dtype=int)  #Missing dimentions will be added using reshape command
    tile_dims = np.ones(len(match_array_chunksize),dtype=int) #Tiling is used so that number of elements in each dimention match
    
    array_to_match_dims = array_to_match.dims
    array_to_reshape_dims = array_to_reshape.dims
    
    for i in range(len(match_array_chunksize)):
        if array_to_match_dims[i] in array_to_reshape_dims:
            reshape_dims[i] = array_to_match.shape[i]
        else:
            tile_dims[i] = array_to_match.shape[i]
            
    return da.tile(da.reshape(array_to_reshape.data,reshape_dims),tile_dims).rechunk(match_array_chunksize)
'''


def calc_briggs_weights(vis_dataset,imaging_weights_parms,grid_parms,sel_parms):
    import dask.array as da
    import xarray as xr
    import numpy as np
    from ._imaging_utils._standard_grid import _graph_standard_grid, _graph_standard_degrid
    
    
    dtr = np.pi / (3600 * 180)
    #grid_parms = {}
    grid_parms['image_size_padded'] =  grid_parms['image_size'] #do not need to pad since no fft
    grid_parms['oversampling'] = 0
    grid_parms['support'] = 1
    grid_parms['do_psf'] = True
    grid_parms['complex_grid'] = False
    grid_parms['do_imaging_weight'] = True
    
    cgk_1D = np.ones((1))
    grid_of_imaging_weights, sum_weight = _graph_standard_grid(vis_dataset, cgk_1D, grid_parms, sel_parms)
    
    
    #############Calculate Briggs parameters#############
    def calculate_briggs_parms(grid_of_imaging_weights, sum_weight, imaging_weights_parms):
        if imaging_weights_parms['weighting'] == 'briggs':
            robust = imaging_weights_parms['robust']
            briggs_factors = np.ones((2,1,1)+sum_weight.shape)
            squared_sum_weight = (np.sum(grid_of_imaging_weights**2,axis=(0,1)))
            briggs_factors[0,0,0,:,:] =  (np.square(5.0*10.0**(-robust))/(squared_sum_weight/sum_weight))[None,None,:,:]
        elif imaging_weights_parms['weighting'] == 'briggs_abs':
            robust = imaging_weights_parms['robust']
            briggs_factors = np.ones((2,1,1)+sum_weight.shape)
            briggs_factors[0,0,0,:,:] = briggs_factor[0,0,0,:,:]*np.square(robust)
            briggs_factors[1,0,0,:,:] = briggs_factor[1,0,0,:,:]*2.0*np.square(imaging_weights_parms['briggs_abs_noise'])
        else:
            briggs_factors = np.zeros((2,1,1)+sum_weight.shape)
            briggs_factors[0,0,0,:,:] = np.ones((1,1,1)+sum_weight.shape)
            
        return briggs_factors
    
    #Map blocks can be simplified by using new_axis and swapping grid_of_imaging_weights and sum_weight
    briggs_factors = da.map_blocks(calculate_briggs_parms,grid_of_imaging_weights,sum_weight, imaging_weights_parms,chunks=(2,1,1)+sum_weight.chunksize,dtype=np.double)[:,0,0,:,:]
    
    imaging_weight = _graph_standard_degrid(vis_dataset, grid_of_imaging_weights, briggs_factors, cgk_1D, grid_parms, sel_parms)
    
    vis_dataset[sel_parms['imaging_weight']] = xr.DataArray(imaging_weight, dims=vis_dataset[sel_parms['data']].dims)
    

