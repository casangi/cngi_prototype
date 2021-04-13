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

def make_imaging_weight(vis_mxds, imaging_weights_parms, grid_parms, sel_parms):
    """
    Creates the imaging weight data variable that has dimensions time x baseline x chan x pol (matches the visibility data variable).
    The weight density can be averaged over channels or calculated independently for each channel using imaging_weights_parms['chan_mode'].
    The following imaging weighting schemes are supported 'natural', 'uniform', 'briggs', 'briggs_abs'.
    The grid_parms['image_size'] and grid_parms['cell_size'] should usually be the same values that will be used for subsequent synthesis blocks (for example making the psf).
    To achieve something similar to 'superuniform' weighting in CASA tclean grid_parms['image_size'] and imaging_weights_parms['cell_size'] can be varied relative to the values used in subsequent synthesis blocks.
    
    Parameters
    ----------
    vis_mxds : xarray.core.dataset.Dataset
        Input multi-xarray Dataset with global data.
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
    sel_parms['xds'] : str
        The xds within the mxds to use to calculate the imaging weights for.
    sel_parms['data_group_in_id'] : int, default = first id in xds.data_groups
        The data group in the xds to use.
    sel_parms['data_group_out_id'] : int, default = sel_parms['data_group_id']
        The output data group. The default will append the imaging weight to the input data group.
    sel_parms['imaging_weight'] : str, default ='IMAGING_WEIGHT'
        The name of that will be used for the imaging weight data variable.
    Returns
    -------
    vis_xds : xarray.core.dataset.Dataset
        The vis_xds will contain a new data variable for the imaging weights the name is defined by the input parameter sel_parms['imaging_weight'].
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
    
    #Deep copy so that inputs are not modified
    _mxds = vis_mxds.copy(deep=True)
    _imaging_weights_parms =  copy.deepcopy(imaging_weights_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    _sel_parms = copy.deepcopy(sel_parms)
    
    ##############Parameter Checking and Set Defaults##############
    assert('xds' in _sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_xds = _mxds.attrs[sel_parms['xds']]
    
    assert _vis_xds.dims['pol'] <= 2, "Full polarization is not supported."
    
    _check_sel_parms(_vis_xds,_sel_parms,new_or_modified_data_variables={'imaging_weight':'IMAGING_WEIGHT'},append_to_in_id=True)
    
    assert(_check_imaging_weights_parms(_imaging_weights_parms)), "######### ERROR: imaging_weights_parms checking failed"
    if _imaging_weights_parms['weighting'] != 'natural':
        assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"
    else:
        #If natural weighting reuse weight
        _sel_parms['data_group_out']['imaging_weight'] = _sel_parms['data_group_in']['weight']
        _vis_xds.attrs['data_groups'][0] = {**_vis_xds.attrs['data_groups'][0], **{_sel_parms['data_group_out']['id']:_sel_parms['data_group_out']}}
        
        print("Since weighting is natural input weight will be reused as imaging weight.")
        print('######################### Created graph for make_imaging_weight #########################')
        return _mxds
        
    #################################################################
    _vis_xds[_sel_parms['data_group_out']['imaging_weight']] = _vis_xds[_sel_parms['data_group_in']['weight']]
    _sel_parms['data_group_in']['imaging_weight'] = _sel_parms['data_group_out']['imaging_weight']
    calc_briggs_weights(_vis_xds,_imaging_weights_parms,_grid_parms,_sel_parms)
        
    #print(_vis_xds)
    _vis_xds.attrs['data_groups'][0] = {**_vis_xds.attrs['data_groups'][0], **{_sel_parms['data_group_out']['id']:_sel_parms['data_group_out']}}
    
    print('######################### Created graph for make_imaging_weight #########################')
    return _mxds
    
# void VisImagingWeight::unPolChanWeight(Matrix<Float>& chanRowWt, const Cube<Float>& corrChanRowWt)
    
'''
def _match_array_shape(array_to_reshape,array_to_match):
    # Reshape in_weight to match dimnetionality of vis_data (vis_xds[imaging_weights_parms['data_name']])
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


def calc_briggs_weights(vis_xds,imaging_weights_parms,grid_parms,sel_parms):
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
    grid_of_imaging_weights, sum_weight = _graph_standard_grid(vis_xds, cgk_1D, grid_parms, sel_parms)
    
    #print('#'*100,grid_of_imaging_weights)
    
#    import matplotlib.pyplot as plt
#    print(sum_weight)
#    plt.figure()
#    plt.imshow(grid_of_imaging_weights[:,:,84,0])
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(grid_of_imaging_weights[:,:,84,0])
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(grid_of_imaging_weights[:,:,84,0]-grid_of_imaging_weights[:,:,84,1])
#    plt.colorbar()
#    plt.show()
    
    '''
    import matplotlib.pyplot as plt
    print(sum_weight)
    plt.figure()
    plt.imshow(grid_of_imaging_weights)
    plt.plot(sum_weight[:,1])
    plt.show()
    '''
    
    
#    import matplotlib.pyplot as plt
#    print(sum_weight)
#    plt.figure()
#    plt.plot(sum_weight[:,0])
#    plt.plot(sum_weight[:,1])
#    plt.show()
    
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
    
#    import matplotlib.pyplot as plt
#    print(grid_of_imaging_weights)
#    plt.figure()
#    plt.imshow(grid_of_imaging_weights[:,:,0,0])
#
#    plt.figure()
#    plt.imshow(grid_of_imaging_weights[:,:,0,1])
#
#    print(np.sum(np.abs(grid_of_imaging_weights[:,:,0,0]-grid_of_imaging_weights[:,:,0,1])).compute())
#    plt.show()
#    print(briggs_factors)
#    a =  briggs_factors.compute()
#    print('helphelphelp',a)
#    print(a[0,:,0],a[0,:,1])
#    print(a[0,:,0]-a[0,:,1])
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.plot(briggs_factors[0,:,0])
#    plt.plot(briggs_factors[0,:,1]-briggs_factors[0,:,0])
#
#    plt.figure()
#    plt.plot(briggs_factors[1,:,0])
#    plt.plot(briggs_factors[1,:,1]-briggs_factors[1,:,0])
#    plt.show()
    
    
    imaging_weight = _graph_standard_degrid(vis_xds, grid_of_imaging_weights, briggs_factors, cgk_1D, grid_parms, sel_parms)
    
    
#    import matplotlib.pyplot as plt
#    print(imaging_weight)
#
#    plt.figure()
#    plt.imshow(imaging_weight[:,:,0,0]-imaging_weight[:,:,0,1],aspect='auto')
#    plt.colorbar()
#    plt.show()
    
    vis_xds[sel_parms['data_group_in']['imaging_weight']] = xr.DataArray(imaging_weight, dims=vis_xds[sel_parms['data_group_in']['data']].dims)
    

