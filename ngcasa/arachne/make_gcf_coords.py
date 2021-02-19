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

'''
    Calculate gridding convolution functions (GCF) as specified for standard, widefield and mosaic imaging.
    Construct a GCF cache (persistent or on-the-fly)

    Options : Choose a list of effects to include
    
    - PSterm : Prolate-Spheroidal gridding kernel (anti-aliasing function)
    - Aterm : Use PB model and Aperture Illumination Function per antenna to construct a GCF per baseline
        - Include support for Heterogeneous Arrays where Aterm is different per antenna
        - Include support for time-varying PB and AIF models. Rotation, etc.
    - Wterm : FT of Fresnel kernel per baseline
    
    Currently creates a gcf to correct for the primary beams of antennas and supports heterogenous arrays (antennas with different dish sizes).
    Only the airy disk and ALMA airy disk model is implemented.
    In the future support will be added for beam squint, pointing corrections, w projection, and including a prolate spheroidal term.
'''

#NB chack ant axis should have no chunking
# w in UVW, ant_1 and ant_2 are used to know what data is flagged. Channel level flagging is not checked
# PA compute problem of time x ant -> time
# https://github.com/numba/numba/issues/4584 reflected lists vs Typed list
# pg does not need chan dim, there will be redundant calculations. Maybe split later
# Futures problem since. Don't know what dims should be
# Compute has to be triggered before using in make_gcf
#dim consitancy needs to be inforced. For example ant1 over time should be the same antenna for a given index or a nan

#NB NB NB cf_parms_indx unkown dimensions
#This function must be run before make_gcf

#To do
#Notes add dish diameters to zpc so that approximations can be calculated
#Allow list of zpc (for multiple types of dishes) list_zpc_dataset
#All the zpc_dataset should have the same pol dims and (frequencies)?
#Go over what should the min support be?

def make_gcf_coords(mxds, list_zpc_dataset, gcf_parms, grid_parms, sel_parms):
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
    Returns
    -------
    gcf_dataset : xarray.core.dataset.Dataset
            
    """
    print('#########################Arachne: Start make_gcf_coords #########################')
    
    from ngcasa._ngcasa_utils._store import _store
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

    gcf_dataset = xr.Dataset()
    vis_dataset = mxds.attrs[sel_parms['xds']]

    ##################################################### PS_TERM #####################################################################
    if _gcf_parms['ps_term']:
        print('#########  Creating ps_term coordinates')
    
    
    ##################################################### A_TERM ######################################################################
    if _gcf_parms['a_term']:
        print('#########  Creating a_term coordinates')
        
        if _gcf_parms['a_function'] == 'zp':
            print('#########  Using ', _gcf_parms['a_function'], 'function')
            
            ######################################################## Beam Models ########################################################
            beam_map,cf_beam_pair_id = _create_beam_map(mxds,sel_parms)
            
            ##############################################################################################################################
            
            ####################################################### Parallactic Angle ####################################################
            ant_ra_dec = _calc_ant_pointing_ra_dec(mxds,gcf_parms['use_pointing_table_parallactic_angle'],_gcf_parms,_sel_parms)
            pa, cf_pa_centers, pa_diff = _calc_parallactic_angles_for_gcf(mxds,ant_ra_dec,_gcf_parms,_sel_parms)
            
            ################################################################################################################################
            
            ####################################################### Channel ####################################################
            chan_map, cf_pb_freq = _create_chan_map(mxds,_gcf_parms,_sel_parms)
            
            ################################################################################################################################
        else:
            print('#########  Using ', _gcf_parms['a_function'], 'function')
    else:
            a=0
#            beam_map = xr.DataArray(da.from_array(beam_map,chunks=(baseline_chunksize)), dims=('baseline'))                 baseline
#            beam_pair_id = xr.DataArray(da.from_array(beam_pair_id,chunks=beam_pair_id.shape), dims=('beam_pair','pair'))   unqiue_baseline x 2

#            ant_ra_dec = _calc_ant_pointing_ra_dec(mxds,gcf_parms['use_pointing_table_parallactic_angle'],_gcf_parms,_sel_parms)  time x ant
#            pa,
#            cf_pa_centers,
#            pa_diff
#            chan_map
#            cf_pb_freq
    
    ###################################################### W_TERM #####################################################################
    if _gcf_parms['w_term']:
        print('#########  Creating w_term coordinates')
        cf_w = _create_w_map(mxds,_gcf_parms,_grid_parms,_sel_parms)
        
    ###################################################### Phase Gradients ############################################################
    if _gcf_parms['phase_gradient_term']:
        print('#########  Creating pointing coordinates')
        
        ant_ra_dec = _calc_ant_pointing_ra_dec(mxds,gcf_parms['use_pointing_table_phase_gradient'],gcf_parms,sel_parms)
        if gcf_parms['use_pointing_table_phase_gradient']:
            cf_pointing = _calc_phase_gradient_pointings(mxds,pointing_ra_dec,_gcf_parms,_sel_parms)
        else:
            field_dataset = mxds.attrs['FIELD']
            cf_pointing = field_dataset.PHASE_DIR[:,0,:].rename({'d0':'cf_pointing','d2':'pair'}).drop_vars(('field_id','source_id'))
        
        
    print('beam',cf_beam_pair_id.shape,'pa',cf_pa_centers.shape,'c',cf_pb_freq.shape,'w',cf_w.shape,'point',cf_pointing.shape)
        
    gcf_dataset= create_cf_map(mxds,gcf_dataset,beam_map,cf_beam_pair_id,pa,cf_pa_centers,chan_map, cf_pb_freq,cf_w,cf_pointing,ant_ra_dec,sel_parms)
    return gcf_dataset
    ###################################################################################################################################

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
    a_pair_parms_indx_list = _ndim_list((n_chunks,))
    a_parms_indx_list = _ndim_list((n_chunks,))
    w_parms_indx_list = _ndim_list((n_chunks,))
    
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
            
        #w_indx_arr, a_indx_arr, a_pair_indx_arr, cf_indx_arr, cf_map, pg_indx_arr, pg_map
        w_parms_indx_list[i_chunk] = chunk_cf_and_pg[0] #can't do from_delayed since number of elements are unkown
        a_parms_indx_list[i_chunk] = chunk_cf_and_pg[1] #can't do from_delayed since number of elements are unkown
        a_pair_parms_indx_list[i_chunk] = chunk_cf_and_pg[2]
        
        cf_parms_indx_list[i_chunk] = chunk_cf_and_pg[3] #can't do from_delayed since number of elements are unkown
        cf_map_list[c_time][c_baseline][c_chan] = da.from_delayed(chunk_cf_and_pg[4], (chunk_sizes[0][c_time],chunk_sizes[1][c_baseline],chunk_sizes[2][c_chan]),dtype=np.int)
        
        pg_parms_indx_list[i_chunk] = chunk_cf_and_pg[5] #can't do from_delayed since number of elements are unkown
        pg_map_list[c_time][c_baseline] = da.from_delayed(chunk_cf_and_pg[6], (chunk_sizes[0][c_time],chunk_sizes[1][c_baseline]),dtype=np.int)
            
        i_chunk = i_chunk+1
        
    cf_map = da.block(cf_map_list) #Awesome function
    pg_map = da.block(pg_map_list)
    
    w_parms_indx = _tree_combine_list(w_parms_indx_list,_find_unique_subset)
    a_parms_indx = _tree_combine_list(a_parms_indx_list,_find_unique_subset)
    a_pair_parms_indx = _tree_combine_list(a_pair_parms_indx_list,_find_unique_subset)
    
    cf_parms_indx = _tree_combine_list(cf_parms_indx_list,_find_unique_subset)
    pg_parms_indx = _tree_combine_list(pg_parms_indx_list,_find_unique_subset)
    
    #list_of_dask_delayed = [cf_map,pg_map,cf_parms_indx,pg_parms_indx,w_parms_indx,a_parms_indx]
    
    dask.visualize([cf_map,pg_map,cf_parms_indx,pg_parms_indx,w_parms_indx,a_parms_indx,a_pair_parms_indx])
    
    list_of_arrs= dask.compute([cf_map,pg_map,cf_parms_indx,pg_parms_indx,w_parms_indx,a_parms_indx,a_pair_parms_indx])
    cf_map,pg_map,cf_parms_indx,pg_parms_indx,w_parms_indx,a_parms_indx,a_pair_parms_indx = list_of_arrs[0]
    
    
    time_chunksize = vis_dataset[sel_parms['data']].chunks[0]
    baseline_chunksize = vis_dataset[sel_parms['data']].chunks[1]
    chan_chunksize = vis_dataset[sel_parms['data']].chunks[2]
    
    cf_map = da.from_array(cf_map,chunks=(time_chunksize,baseline_chunksize,chan_chunksize))
    w_parms_indx = da.from_array(w_parms_indx,chunks=(1,1))
    a_parms_indx = da.from_array(a_parms_indx,chunks=(1,4))
    a_pair_parms_indx = da.from_array(a_pair_parms_indx,chunks=(1,3))
    cf_parms_indx = da.from_array(cf_parms_indx,chunks=(1,4))
    
    pg_parms_indx = da.from_array(pg_parms_indx,chunks=(1,3))
    pg_map = da.from_array(pg_map,chunks=(time_chunksize,baseline_chunksize))
    
    gcf_dataset = xr.Dataset()
    coords = {'gcf_indx':['a1_flat','a2_flat','w','gcf_flat'],'pg_indx':['p1','p2','pg_flat'],'a_indx':['pa','b','c','a_flat'],'w_indx':['w'],'a_pair_indx':['a1_flat','a2_flat','a12_flat']}
    #coords['gcf_a_pa'] = cf_pa_centers.data.compute()
    #coords['gcf_a_freq'] = cf_pb_freq.data.compute()
    coords['beam_id'] =  beam_ids
    #print(beam_ids)
    #coords['gcf_w'] = cf_w.data.compute()
    
    #coords['pg_pointing'] = cf_pointing.data.compute()
    
    gcf_dataset = gcf_dataset.assign_coords(coords)
    
    gcf_dataset['GCF_MAP'] = xr.DataArray(cf_map, dims=('time','baseline','chan'))
    gcf_dataset['GCF_PARMS_INDX'] = xr.DataArray(cf_parms_indx, dims=('gcf','gcf_indx'))
    gcf_dataset['W_PARMS_INDX'] = xr.DataArray(w_parms_indx, dims=('w','w_indx'))
    
    gcf_dataset['A_PARMS_INDX'] = xr.DataArray(a_parms_indx, dims=('a','a_indx'))
    
    gcf_dataset['A_PAIR_PARMS_INDX'] = xr.DataArray(a_pair_parms_indx, dims=('a_pair','a_pair_indx'))
    
    gcf_dataset['A_PA'] = cf_pa_centers
    gcf_dataset['A_FREQ'] = cf_pb_freq
    gcf_dataset['A_BEAM_ID'] = cf_beam_pair_id
    gcf_dataset['W'] = cf_w
    
    gcf_dataset['PG_MAP'] =  xr.DataArray(pg_map, dims=('time','baseline'))
    gcf_dataset['PG_PARMS_INDX'] =  xr.DataArray(pg_parms_indx, dims=('pg','pg_indx'))
    gcf_dataset['PG_POINTING'] = cf_pointing
    
        
    '''
    cf_map = da.block(cf_map_list) #Awesome function
    pg_map = da.block(pg_map_list)
    
    w_parms_indx = da.from_delayed(_tree_combine_list(w_parms_indx_list,_find_unique_subset),shape=(np.nan,1),dtype=int) #(nan,1) first dim length is unkown
    a_parms_indx = da.from_delayed(_tree_combine_list(a_parms_indx_list,_find_unique_subset),shape=(np.nan,6),dtype=int) #(nan,6) first dim length is unkown
    cf_parms_indx = da.from_delayed(_tree_combine_list(cf_parms_indx_list,_find_unique_subset),shape=(np.nan,3),dtype=int) #(nan,3) first dim length is unkown
    pg_parms_indx = da.from_delayed(_tree_combine_list(pg_parms_indx_list,_find_unique_subset),shape=(np.nan,3),dtype=int) #(nan,3) first dim length is unkown
    
    #w_parms_indx = da.from_delayed(_tree_combine_list(w_parms_indx_list,_find_unique_subset),shape=(np.nan,1),dtype=int) #(nan,1) first dim length is unkown
    #a_parms_indx = da.from_delayed(_tree_combine_list(a_parms_indx_list,_find_unique_subset),shape=(np.nan,6),dtype=int) #(nan,6) first dim length is unkown
    #cf_parms_indx = da.from_delayed(_tree_combine_list(cf_parms_indx_list,_find_unique_subset),shape=(280,7),dtype=int) #(nan,3) first dim length is unkown
    #pg_parms_indx = da.from_delayed(_tree_combine_list(pg_parms_indx_list,_find_unique_subset),shape=(23,3),dtype=int) #(nan,3) first dim length is unkown
    
    
    
    gcf_dataset = xr.Dataset()
    coords = {'gcf_indx':['a','w','gcf_flat'],'pg_indx':['p1','p2','pg_flat'],'a_indx':['pa1','b1','pa2','b2','c','a_flat'],'w_indx':['w']}
    gcf_dataset = gcf_dataset.assign_coords(coords)
    
    gcf_dataset['GCF_MAP'] = xr.DataArray(cf_map, dims=('time','baseline','chan'))
    gcf_dataset['GCF_PARMS_INDX'] = xr.DataArray(cf_parms_indx, dims=('gcf','gcf_indx'))
    
    gcf_dataset['W_PARMS_INDX'] = xr.DataArray(w_parms_indx, dims=('w','w_indx'))
    gcf_dataset['A_PARMS_INDX'] = xr.DataArray(a_parms_indx, dims=('a','a_indx'))
    
    gcf_dataset['GCF_A_PA'] = cf_pa_centers
    gcf_dataset['GCF_A_FREQ'] = cf_pb_freq
    gcf_dataset['GCF_A_BEAM_ID'] = cf_beam_pair_id
    gcf_dataset['GCF_W'] = cf_w
    
    gcf_dataset['PG_MAP'] =  xr.DataArray(pg_map, dims=('time','baseline'))
    gcf_dataset['PG_PARMS_INDX'] =  xr.DataArray(pg_parms_indx, dims=('pg','pg_indx'))
    gcf_dataset['PG_POINTING'] = cf_pointing
    '''

    #dask.visualize(gcf_dataset,'make_gcf_coords')
    return gcf_dataset
    

from ._imaging_utils._math import _combine_indx_permutation, _combine_indx_combination, _find_val_indx, _find_angle_indx, _find_ra_dec_indx

@jit(nopython=True,cache=True,nogil=True)
def _cf_map_jit(beam_map,beam_ids,cf_beam_pair_id,pa,cf_pa,ant_1,ant_2,ant_ids,chan_map,freq_chan,cf_chan,w,cf_w,pointing_ra_dec,cf_pointing):
#    print(beam_map.shape)
#    print(beam_ids.shape)
#    print(cf_beam_pair_id)
#    print(pa.shape)
#    print(cf_pa.shape)
#    print(ant_1.shape)
#    print(ant_2.shape)
#    print(ant_ids.shape)
#    print(chan_map.shape)
#    print(w.shape)
#    print(cf_w.shape)
#    print(pointing_ra_dec.shape)
#    print(cf_pointing.shape)

    n_time = ant_1.shape[0]
    n_baseline = ant_1.shape[1]
    n_chan = chan_map.shape[0]
    
    #print('n_time', n_time, 'n_baseline', n_baseline, 'n_chan', n_chan)
    c = 299792458.0
    
    n_cf_beam = len(beam_ids) #not pairs
    n_cf_pa = len(cf_pa)
    n_cf_w = len(cf_w)
    n_cf_c = len(cf_chan)
    n_cf_point = len(cf_pointing)
    
    #cf_indx_list values
    #cf_indx_list = [np.array([-42,-42,-42,-42,-42,-42,-42])] #Can't have an empty list need to tell Numba what type it is
    cf_indx_list = [np.array([-42,-42,-42,-42])]
    w_indx_list = [np.array([-42])]
    a_indx_list = [np.array([-42,-42,-42,-42])] #['pa1','b1','pa2','b2','c','a_flat']
    a_pair_indx_list = [np.array([-42,-42,-42])]
     
    cf_map = np.zeros((n_time,n_baseline,n_chan),numba.i8)
    #cf_map = np.zeros((n_time,n_baseline,n_chan),np.int) #if debug in python mode
    
    #pg_indx_list ['p1','p2','pg_flat']
    pg_indx_list = [np.array([-42,-42,-42])] #Can't have an empty list need to tell Numba what type it is
    pg_map = np.zeros((n_time,n_baseline),numba.i8)
    #pg_map = np.zeros((n_time,n_baseline),np.int) #if debug in python mode
    
    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            if not(np.isnan(w[i_time,i_baseline])) and not(np.isnan(ant_1[i_time,i_baseline])) and not(np.isnan(ant_2[i_time,i_baseline])):
                ############Beam calcs############
                beam_pair_indx = beam_map[i_baseline]
                cf_beam1_indx = np.where(beam_ids == cf_beam_pair_id[beam_pair_indx,0])[0][0]
                cf_beam2_indx = np.where(beam_ids == cf_beam_pair_id[beam_pair_indx,1])[0][0]
            
                ############PA calcs############
                a1_indx = np.where(ant_ids ==  ant_1[i_time,i_baseline])[0][0]
                a2_indx = np.where(ant_ids ==  ant_2[i_time,i_baseline])[0][0]
            
                #Add pa map if timexant is large and takes too long to compute
                pa1 = pa[i_time,a1_indx]
                pa2 = pa[i_time,a2_indx]
                
                cf_pa1_indx = _find_angle_indx(cf_pa,pa1)
                cf_pa2_indx = _find_angle_indx(cf_pa,pa1)
                
                ############Pointing calcs############
                point1 = pointing_ra_dec[i_time,a1_indx]
                point2 = pointing_ra_dec[i_time,a2_indx]
                
                cf_point1_indx = _find_ra_dec_indx(cf_pointing,point1)
                cf_point2_indx = _find_ra_dec_indx(cf_pointing,point2)
                
                ############Calculate Flat Index for Phase Gradients (pg)############
                # [P1,P2]
                i_pg, n_pg = _combine_indx_combination(cf_point1_indx,cf_point2_indx,n_cf_point,n_cf_point)
                pg_indx_info = np.array([cf_point1_indx,cf_point2_indx,i_pg]) #-42 is a dummy value. If it appears in the final result something has gone wrong
                
                pg_map[i_time,i_baseline] = i_pg
                
                #Nasty code needed due to working with list in numba. Can't put in a separate function, due to typed list inefficiencies.
                found = False
                end_of_list = False
                i_list = 0
                n_list = len(pg_indx_list)
                while not(found) and not(end_of_list):
                    #print(i_list,n_list)
                    if pg_indx_list[i_list][-1] == pg_indx_info[-1]:
                        found = True
                    i_list = i_list+1
                    if i_list >= n_list:
                        end_of_list = True
          
                if not(found):
                    pg_indx_list.append(pg_indx_info)
                
                ######################################
                w_val = np.abs(w[i_time,i_baseline])
            
                for i_chan in range(n_chan):
                    ############W calcs############
                    w_val = w_val*freq_chan[i_chan]/c
                    #print('cf_w,w_val',cf_w,w_val)
                    cf_w_indx = _find_val_indx(cf_w,w_val)
                    
                    ############Chan calcs############
                    cf_c_indx = chan_map[i_chan]
                    
                    ############Calculate Flat Index for cf (convolution function)############
                    #Calculate flat index for a term [pa,b,c,a_flat] x2 (one for each antenna)
                    i1,n1 = _combine_indx_permutation(cf_pa1_indx,cf_beam1_indx,n_cf_pa,n_cf_beam)
                    cf_a1_indx,n_cf_a1 = _combine_indx_permutation(i1,cf_c_indx,n1,n_cf_c)
                    
                    i1,n1 = _combine_indx_permutation(cf_pa2_indx,cf_beam2_indx,n_cf_pa,n_cf_beam)
                    cf_a2_indx,n_cf_a2 = _combine_indx_permutation(i1,cf_c_indx,n1,n_cf_c)
                    
                    a1_indx_info = np.array([cf_pa1_indx,cf_beam1_indx,cf_c_indx,cf_a1_indx])
                    a2_indx_info = np.array([cf_pa2_indx,cf_beam2_indx,cf_c_indx,cf_a2_indx])
                    
                    #Add w index
                    w_indx_info = np.array([cf_w_indx])
                    
                    #Combined (a and w) cf flat index [a1_flat,a2_flat,w,cf_flat]
                    i1,n1 = _combine_indx_combination(cf_a1_indx,cf_a2_indx,n_cf_a1,n_cf_a2)
                    
                    a_pair_indx_info = np.array([cf_a1_indx,cf_a2_indx,i1])
                    
                    cf_flat_index, n_cf = _combine_indx_permutation(i1,cf_w_indx,n1,n_cf_w)
                    cf_indx_info = np.array([cf_a1_indx,cf_a2_indx,cf_w_indx,cf_flat_index])
                    
                    cf_map[i_time,i_baseline,i_chan] = cf_flat_index #used by gridder
                    
                    #Nasty code needed due to working with list in numba. Can't put in a separate function, due to typed list inefficiencies.
                    found = False
                    end_of_list = False
                    i_list = 0
                    n_list = len(cf_indx_list)
                    while not(found) and not(end_of_list):
                        #print(i_list,n_list)
                        if cf_indx_list[i_list][-1] == cf_indx_info[-1]:
                            found = True
                        i_list = i_list+1
                        if i_list >= n_list:
                            end_of_list = True
              
                    if not(found):
                        cf_indx_list.append(cf_indx_info)
                        
                    #Nasty code needed due to working with list in numba. Can't put in a separate function, due to typed list inefficiencies.
                    found = False
                    end_of_list = False
                    i_list = 0
                    n_list = len(a_pair_indx_list)
                    while not(found) and not(end_of_list):
                        #print(i_list,n_list)
                        if a_pair_indx_list[i_list][-1] == a_pair_indx_info[-1]:
                            found = True
                        i_list = i_list+1
                        if i_list >= n_list:
                            end_of_list = True
              
                    if not(found):
                        a_pair_indx_list.append(a_pair_indx_info)
                        
                        
                    #Nasty code needed due to working with list in numba. Can't put in a separate function, due to typed list inefficiencies.
                    found = False
                    end_of_list = False
                    i_list = 0
                    n_list = len(w_indx_list)
                    while not(found) and not(end_of_list):
                        #print(i_list,n_list)
                        if w_indx_list[i_list][-1] == w_indx_info[-1]:
                            found = True
                        i_list = i_list+1
                        if i_list >= n_list:
                            end_of_list = True
              
                    if not(found):
                        w_indx_list.append(w_indx_info)
                        
                    #Nasty code needed due to working with list in numba. Can't put in a separate function, due to typed list inefficiencies.
                    
                    for a_indx_info in [a1_indx_info,a2_indx_info]:
                        found = False
                        end_of_list = False
                        i_list = 0
                        n_list = len(a_indx_list)
                        while not(found) and not(end_of_list):
                            #print(i_list,n_list)
                            if a_indx_list[i_list][-1] == a_indx_info[-1]:
                                found = True
                            i_list = i_list+1
                            if i_list >= n_list:
                                end_of_list = True
                  
                        if not(found):
                            a_indx_list.append(a_indx_info)
                    
    ##################
    cf_indx_list.pop(0)
    #Convert list of arrays to array (numpy functions vstack,stack,asarray don't work in numba for lists). Also avoid tight for loop.
    cf_indx_arr = np.zeros((len(cf_indx_list),len(cf_indx_list[0])),numba.i8)
    #cf_indx_arr = np.zeros((len(cf_indx_list),len(cf_indx_list[0])),int)
    
    n_cf_flat = len(cf_indx_list)
    n_i = len(cf_indx_list[0])
    
    for jj in range(n_cf_flat ):
        for ii in range(n_i):
            cf_indx_arr[jj,ii] = cf_indx_list[jj][ii]
            
    ##################
    a_pair_indx_list.pop(0)
    #Convert list of arrays to array (numpy functions vstack,stack,asarray don't work in numba for lists). Also avoid tight for loop.
    a_pair_indx_arr = np.zeros((len(a_pair_indx_list),len(a_pair_indx_list[0])),numba.i8)
    
    n_a_pair_flat = len(a_pair_indx_list)
    n_i = len(a_pair_indx_list[0])
    
    for jj in range(n_a_pair_flat):
        for ii in range(n_i):
            a_pair_indx_arr[jj,ii] = a_pair_indx_list[jj][ii]
            
                                        
            
    ##################
    w_indx_list.pop(0)
    #Convert list of arrays to array (numpy functions vstack,stack,asarray don't work in numba for lists). Also avoid tight for loop.
    w_indx_arr = np.zeros((len(w_indx_list),len(w_indx_list[0])),numba.i8)
    #w_indx_arr = np.zeros((len(w_indx_list),len(w_indx_list[0])),int)
    
    n_w_flat = len(w_indx_list)
    n_i = len(w_indx_list[0])
    
    for jj in range(n_w_flat ):
        for ii in range(n_i):
            w_indx_arr[jj,ii] = w_indx_list[jj][ii]
            
    ##################
    a_indx_list.pop(0)
    #Convert list of arrays to array (numpy functions vstack,stack,asarray don't work in numba for lists). Also avoid tight for loop.
    a_indx_arr = np.zeros((len(a_indx_list),len(a_indx_list[0])),numba.i8)
    #a_indx_arr = np.zeros((len(a_indx_list),len(a_indx_list[0])),int)
    
    n_a_flat = len(a_indx_list)
    n_i = len(a_indx_list[0])
    
    for jj in range(n_a_flat ):
        for ii in range(n_i):
            a_indx_arr[jj,ii] = a_indx_list[jj][ii]
            
    ##################
    pg_indx_list.pop(0)
    #Convert list of arrays to array (numpy functions vstack,stack,asarray don't work in numba for lists). Also avoid tight for loop.
    pg_indx_arr = np.zeros((len(pg_indx_list),len(pg_indx_list[0])),numba.i8)
    #pg_indx_arr = np.zeros((len(pg_indx_list),len(pg_indx_list[0])),int)
        
    n_pg_flat = len(pg_indx_list)
    n_i = len(pg_indx_list[0])
    
    for jj in range(n_pg_flat):
        for ii in range(n_i):
            pg_indx_arr[jj,ii] = pg_indx_list[jj][ii]
            
    
    return w_indx_arr, a_indx_arr, a_pair_indx_arr, cf_indx_arr, cf_map, pg_indx_arr, pg_map
    


@jit(nopython=True,cache=True,nogil=True)
def calc_cf_flat_indx(cf_indx_info,n_beam,n_pa,n_w,n_c):
    p1, b1, p2, b2, w, c, x = cf_indx_info
    i1,n1 = _combine_indx_permutation(p1,b1,n_pa,n_beam)
    i2,n2 = _combine_indx_permutation(p2,b2,n_pa,n_beam)
    i3,n3 = _combine_indx_combination(i1,i2,n1,n2)
    i4,n4 = _combine_indx_permutation(i3,w,n3,n_w)
    i5,n5 = _combine_indx_permutation(i4,c,n4,n_c)
    cf_indx_info[-1] = i5
    return n5
    
    
def calculate_conv_size(vis_dataset, list_zpc_dataset, grid_parms):
    
    ##########PS term Support##########
    n_ps = 7

    ##########Calculate max and min support for A term##########
    D_eta_max = 0 #Maximum dish diameter
    D_eta_min = 99999 #Min dish diameter
    for zpc_dataset in list_zpc_dataset:
        D_eta = (zpc_dataset.attrs['dish_diam']*np.max(zpc_dataset.ETA)).data.compute()
        if D_eta_max < D_eta:
            D_eta_max = D_eta
        if D_eta_min > D_eta:
            D_eta_min = D_eta
            
    lambda_min = c/np.max(vis_dataset.chan.data)
    lambda_max = c/np.min(vis_dataset.chan.data)
    delta_uv_over = 1/(grid_parms['image_size']*grid_parms['cell_size']*grid_parms['oversampling'])
    n_a_max = np.max(np.ceil(np.abs(D_eta_max/(lambda_min*delta_uv_over))))
    n_a_min = np.min(np.ceil(np.abs(D_eta_min/(lambda_max*delta_uv_over))))
    
    ##########W term Support##########
    
    print(delta_uv_over)
    print(n_a_max,n_a_min)
    
    #min_support = 3
    #assert(n_a_min > (min_support+1)*grid_parms['oversampling']), "######### ERROR: Increase image_size or cell_size"
    

