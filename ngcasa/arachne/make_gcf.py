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

import numpy as np
from scipy.constants import c
import xarray as xr
import dask
import dask.array as da
from numba import jit
import numba
import time
import dask.dataframe as dd
import matplotlib.pyplot as plt


def make_gcf(gcf_dataset, list_zpc_dataset, gcf_parms, grid_parms, sel_parms):
    """
    Under construction.
    
    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input visibility dataset.
    gcf_parms : dictionary
    gcf_parms['function'] : {'alma_airy'/'airy'}, default = 'alma_airy'
        The primary beam model used (a function of the dish diameter and blockage diameter).
    gcf_parms['list_dish_diameters']  : list of number, units = meter
        A list of unique antenna dish diameters.
    gcf_parms['list_blockage_diameters']  : list of number, units = meter
        A list of unique feed blockage diameters (must be the same length as gcf_parms['list_dish_diameters']).
    gcf_parms['unique_ant_indx']  : list of int
        A list that has indeces for the gcf_parms['list_dish_diameters'] and gcf_parms['list_blockage_diameters'] lists, for each antenna.
    gcf_parms['image_phase_center']  : list of number, length = 2, units = radians
        The mosaic image phase center.
    gcf_parms['a_chan_num_chunk']  : int, default = 3
        The number of chunks in the channel dimension of the gridding convolution function data variable.
    gcf_parms['oversampling']  : list of int, length = 2, default = [10,10]
        The oversampling of the gridding convolution function.
    gcf_parms['max_support']  : list of int, length = 2, default = [15,15]
        The maximum allowable support of the gridding convolution function.
    gcf_parms['support_cut_level']  : number, default = 0.025
        The antennuation at which to truncate the gridding convolution function.
    gcf_parms['chan_tolerance_factor']  : number, default = 0.005
        It is the fractional bandwidth at which the frequency dependence of the primary beam can be ignored and determines the number of frequencies for which to calculate a gridding convolution function. Number of channels equals the fractional bandwidth devided by gcf_parms['chan_tolerance_factor'].
    grid_parms : dictionary
    grid_parms['image_size'] : list of int, length = 2
        The image size (no padding).
    grid_parms['cell_size']  : list of number, length = 2, units = arcseconds
        The image cell size.
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
    gcf_dataset : xarray.core.dataset.Dataset
            
    """
    print('#########################Arachne: Start make_gcf #########################')
    
    from ngcasa._ngcasa_utils._store import _store
    from ngcasa._ngcasa_utils._check_parms import _check_storage_parms
    #from ._imaging_utils._check_imaging_parms import _check_pb_parms
    #from ._imaging_utils._check_imaging_parms import _check_grid_parms, _check_gcf_parms
    #from ._imaging_utils._gridding_convolutional_kernels import _create_prolate_spheroidal_kernel_2D, _create_prolate_spheroidal_image_2D
    #from ._imaging_utils._remove_padding import _remove_padding
    import numpy as np
    import dask.array as da
    import copy, os
    import xarray as xr
    import itertools
    import dask
    import dask.array.fft as dafft
    from ._imaging_utils._calc_parallactic_angles import _calc_parallactic_angles_for_gcf
    from ._imaging_utils._a_term import _create_chan_map, _create_beam_map
    from ._imaging_utils._w_term import _calculate_w_list, _calc_w_sky, _create_w_map
    from ._imaging_utils._ps_term import _create_prolate_spheroidal_image_2D
    from ._imaging_utils._phase_gradients import _calc_phase_gradient_pointings, _calc_ant_pointing_ra_dec
    import matplotlib.pylab as plt
    
    _gcf_parms =  copy.deepcopy(gcf_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    _sel_parms = copy.deepcopy(sel_parms)
    
    print(gcf_dataset)
    
    #GCF_PARMS_INDX  (gcf, gcf_indx) int64 dask.array<chunksize=(280, 7), meta=np.ndarray>
    
    #GCF_A_BEAM_ID   (beam_pair, pair) int32 dask.array<chunksize=(4, 2), meta=np.ndarray>
    #GCF_A_FREQ      (cf_freq) float64 dask.array<chunksize=(10,), meta=np.ndarray>
    #GCF_A_PA        (pa) float64 dask.array<chunksize=(7,), meta=np.ndarray>
    #GCF_MAP         (time, baseline, chan) int64 dask.array<chunksize=(271, 43, 83), meta=np.ndarray>
    #GCF_W           (cf_w) float64 dask.array<chunksize=(107,), meta=np.ndarray>
    
    ##################################################### PS_TERM #####################################################################
    if _gcf_parms['ps_term']:
        print('#########  Creating ps_term')
    
    
    ##################################################### A_TERM ######################################################################
    if _gcf_parms['a_term']:
        print('#########  Creating a_term')
        
        
        
    else:
        a = 0
    
    ###################################################### W_TERM #####################################################################
    if _gcf_parms['w_term']:
        print('#########  Creating w_term ')
        cf_w_planes = _create_w_term(gcf_dataset, _gcf_parms, _grid_parms, _sel_parms)
        
      
        
    ###################################################### Phase Gradients ############################################################
    if _gcf_parms['phase_gradient_term']:
        print('#########  Creating pg_term ')
        
    
        
        
    
    
    '''
    n_chunks_gcf = gcf_dataset.GCF_PARMS_INDX.data.numblocks[0]
    gcf = []
    for c_gcf in range(n_chunks_gcf):
        #print(c_gcf)
        temp = dask.delayed(_gcf_creation)(
            gcf_dataset.GCF_PARMS_INDX.data.partitions[c_gcf,:],
            gcf_dataset. gcf_indx,
            gcf_dataset.GCF_A_FREQ,
            gcf_dataset.GCF_A_PA,
            gcf_dataset.GCF_W,
            gcf_dataset.GCF_A_BEAM_ID,
            list_zpc_dataset
            )
        gcf.append(temp)

    dask.compute(gcf)
    '''
    
def _create_w_term(gcf_dataset, gcf_parms, grid_parms, sel_parms):
    from ._imaging_utils._w_term import _calc_w_sky


    chunks = (gcf_dataset.GCF_W.chunks[0][0],gcf_parms['conv_size'][0],gcf_parms['conv_size'][1])
    cf_w_planes = da.map_blocks(_calc_w_sky,gcf_dataset.GCF_W.data,gcf_parms,grid_parms,dtype=np.complex128,chunks=chunks,new_axis=(1,2))

    '''
    cf_w_planes = cf_w_planes.compute()
    print(cf_w_planes.shape)
    plt.figure()
    plt.imshow(abs(cf_w_planes[4,:,:]))
    plt.figure()
    plt.imshow(np.real(cf_w_planes[4,:,:]))
    plt.figure()
    plt.imshow(np.imag(cf_w_planes[4,:,:]))
    plt.show()
    '''
    return cf_w_planes
    

    
    
    
    
    
def _gcf_creation(gcf_parms_indx, gcf_indx_labels, gcf_a_freq, gcf_a_pa, gcf_w, gcf_a_beam_id,list_zpc_dataset):
    #print(gcf_a_freq)
    print(gcf_parms_indx)
    #print(gcf_w)
    
    w_indx_indx = np.where(gcf_indx_labels == 'w')[0][0]
    print(w_indx_indx)
    
    for indx_set in gcf_parms_indx:
        a =1
        #print(parms)
        
        
        
        #Create w kernel
        #w_val = gcf_w[indx_set[w_indx_indx]]
        #print(w_val)

        

   
'''
def create_cf_map(mxds,gcf_dataset,beam_map,cf_beam_pair_id,pa,cf_pa_centers,chan_map, cf_pb_freq,cf_w,cf_pointing,pointing_ra_dec,sel_parms):
    import itertools
    from ._imaging_utils._general import _ndim_list
    from ._imaging_utils._dask_utils import _tree_combine_list, _find_unique_subset

    vis_dataset = mxds.attrs[sel_parms['xds']]
    n_chunks_in_each_dim = vis_dataset[sel_parms["data"]].data.numblocks
    chunk_sizes = vis_dataset[sel_parms["data"]].chunks
    
    w = vis_dataset.UVW[:,:,2]
    
    iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]),
                                         np.arange(n_chunks_in_each_dim[2]))
                                         
    ant_1 = vis_dataset.ANTENNA1
    ant_2 = vis_dataset.ANTENNA2
    ant_ids = mxds.ANTENNA.antenna_id.data
    beam_ids = mxds.beam_ids.data
    freq_chan = vis_dataset.chan.data
    n_chunks = np.prod(n_chunks_in_each_dim[:3])
    
    cf_map_list = _ndim_list((n_chunks_in_each_dim[0],n_chunks_in_each_dim[1],n_chunks_in_each_dim[2]))
    cf_parms_indx_list = _ndim_list((n_chunks,))
    
    #pg does not need chan dim, there will be redundant calculations. Maybe split later
    pg_map_list = _ndim_list((n_chunks_in_each_dim[0],n_chunks_in_each_dim[1]))
    pg_parms_indx_list = _ndim_list((n_chunks,))
    
    i_chunk = 0
    for c_time, c_baseline, c_chan in iter_chunks_indx:
        #print('c_time,c_baseline,c_chan',c_time,c_baseline,c_chan)
        chunk_cf_and_pg = dask.delayed(_cf_map_jit)(
            beam_map.data.partitions[c_baseline],
            beam_ids,
            cf_beam_pair_id.data,
            pa.data.partitions[c_time,0],
            cf_pa_centers.data,
            ant_1.data.partitions[c_time,c_baseline],
            ant_2.data.partitions[c_time,c_baseline],
            ant_ids,
            chan_map.data.partitions[c_chan],
            freq_chan,
            cf_pb_freq.data,
            w.data.partitions[c_time,c_baseline],
            cf_w.data,
            pointing_ra_dec.data.partitions[c_time,0],
            cf_pointing.data)
        
        cf_parms_indx_list[i_chunk] = chunk_cf_and_pg[0] #can't do from_delayed since number of elements are unkown
        cf_map_list[c_time][c_baseline][c_chan] = da.from_delayed(chunk_cf_and_pg[1], (chunk_sizes[0][c_time],chunk_sizes[1][c_baseline],chunk_sizes[2][c_chan]),dtype=np.int)
        
        pg_parms_indx_list[i_chunk] = chunk_cf_and_pg[2] #can't do from_delayed since number of elements are unkown
        pg_map_list[c_time][c_baseline] = da.from_delayed(chunk_cf_and_pg[3], (chunk_sizes[0][c_time],chunk_sizes[1][c_baseline]),dtype=np.int)
            
        i_chunk = i_chunk+1
        
    cf_map = da.block(cf_map_list) #Awesome function
    pg_map = da.block(pg_map_list)
     
    cf_parms_indx = da.from_delayed(_tree_combine_list(cf_parms_indx_list,_find_unique_subset),shape=(np.nan,7),dtype=int) #(nan,7) first dim length is unkown
    pg_parms_indx = da.from_delayed(_tree_combine_list(pg_parms_indx_list,_find_unique_subset),shape=(np.nan,3),dtype=int) #(nan,3) first dim length is unkown
    #cf_parms_indx = da.from_delayed(_tree_combine_list(cf_parms_indx_list,_find_unique_subset),shape=(280,7),dtype=int) #(nan,7) first dim length is unkown
    #pg_parms_indx = da.from_delayed(_tree_combine_list(pg_parms_indx_list,_find_unique_subset),shape=(23,3),dtype=int) #(nan,3) first dim length is unkown
    
    
    
    gcf_dataset = xr.Dataset()
    coords = {'gcf_indx':['pa1','b1','pa2','b2','w','c','gcf_flat'],'pg_indx':['p1','p2','pg_flat']}
    gcf_dataset = gcf_dataset.assign_coords(coords)
    
    gcf_dataset['GCF_MAP'] = xr.DataArray(cf_map, dims=('time','baseline','chan'))
    gcf_dataset['GCF_PARMS_INDX'] = xr.DataArray(cf_parms_indx, dims=('gcf','gcf_indx'))
    gcf_dataset['GCF_A_PA'] = cf_pa_centers
    gcf_dataset['GCF_A_FREQ'] = cf_pb_freq
    gcf_dataset['GCF_A_BEAM_ID'] = cf_beam_pair_id
    gcf_dataset['GCF_W'] = cf_w
    
    gcf_dataset['PG_MAP'] =  xr.DataArray(pg_map, dims=('time','baseline'))
    gcf_dataset['PG_PARMS_INDX'] =  xr.DataArray(pg_parms_indx, dims=('pg','pg_indx'))
    gcf_dataset['PG_POINTING'] = cf_pointing

    return gcf_dataset
'''
