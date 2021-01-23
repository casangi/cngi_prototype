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

def make_gcf_coords(mxds, list_zpc_dataset, gcf_parms, grid_parms, sel_parms, storage_parms):
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
    print('#########################Arachne: Start make_gcf_coords #########################')
    
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
    from ._imaging_utils._phase_gradients import _calc_phase_gradient_pointings
    import matplotlib.pylab as plt
    
    #To do
    #Notes add dish diameters to zpc so that approximations can be calculated
    #Allow list of zpc (for multiple types of dishes) list_zpc_dataset
    #All the zpc_dataset should have the same pol dims and (frequencies)?
    #Go over what should the min support be?
    
    _gcf_parms =  copy.deepcopy(gcf_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    _sel_parms = copy.deepcopy(sel_parms)

    gcf_dataset = xr.Dataset()
    
    vis_dataset = mxds.attrs[sel_parms['xds']]
    
#    #if (_gcf_parms['a_term'] or _gcf_parms['do_pointing']) and (gcf_parms['use_pointing_table_phase_gradient'] or gcf_parms['use_pointing_table_parallactic_angle']):
#    if (_gcf_parms['a_term'] and gcf_parms['use_pointing_table_parallactic_angle']) or (gcf_parms['phase_gradient_term'] and gcf_parms['use_pointing_table_phase_gradient']):
#        #Add another check if available
#        ant_ra_dec = mxds.POINTING.DIRECTION.interp(time=vis_dataset.time,assume_sorted=False,method=gcf_parms['interpolation_method'])[:,:,0,:]
#        ant_ra_dec = ant_ra_dec.chunk({"time":vis_dataset[sel_parms['data']].chunks[0][0]})
#
#    if (_gcf_parms['a_term'] and not(gcf_parms['use_pointing_table_parallactic_angle'])) or (gcf_parms['phase_gradient_term'] and not(gcf_parms['use_pointing_table_phase_gradient'])):
#        field_dataset = mxds.attrs['FIELD']
#        field_id = np.max(vis_dataset.FIELD_ID,axis=1).compute() #np.max ignores int nan values (nan values are large negative numbers for int).
#        n_field = field_dataset.dims['d0']
#        ant_ra_dec = field_dataset.PHASE_DIR.isel(d0=field_id)
#        if n_field != 1:
#            ant_ra_dec = ant_ra_dec[:,0,:]
#        ant_ra_dec = ant_ra_dec.expand_dims('ant',1)
#        n_ant = len(mxds.antenna_ids)
#        ant_ra_dec = da.tile(ant_ra_dec.data,(1,n_ant,1))
#        ant_ra_dec =  xr.DataArray(ant_ra_dec,{'time':vis_dataset.time,'ant':antenna_ids}, dims=('time','ant'))

    
    ##################################################### PS_TERM #####################################################################
    if _gcf_parms['ps_term']:
        print('#########  Creating ps_term coordinates')
    
    
    ##################################################### A_TERM ######################################################################
    if _gcf_parms['a_term']:
        print('#########  Creating a_term coordinates')
        
        if _gcf_parms['a_function'] == 'zp':
            print('#########  Using ', _gcf_parms['a_function'], 'function')
            
            ######################################################## Beam Models ########################################################
            #n_unique_ant = len(_gcf_parms['list_dish_diameters'])
            #beam_map, baseline x 1
            #beam_pair_id, number of unique beam pairs x 2
            beam_map,cf_beam_pair_id = _create_beam_map(mxds,sel_parms)
            
            #print('beam_map',beam_map.data.compute())
            #print('beam_pair_id',cf_beam_pair_id.data.compute())
            ##############################################################################################################################
            
            ####################################################### Parallactic Angle ####################################################
            #This will eventaully move to convert ms.
            #try:
            #    transform_pointing_table(mxds,gcf_parms,sel_parms) #temp function, should be included in convert_ms
            #except:
            #    print('Conversion of Pointing Table Failed')
            
            #pa, time x ant
            #pa, number of unique pa x 1
            #pa_diff, time x ant
            
            if (gcf_parms['use_pointing_table_parallactic_angle']):
                ant_ra_dec = mxds.POINTING.DIRECTION.interp(time=vis_dataset.time,assume_sorted=False,method=gcf_parms['interpolation_method'])[:,:,0,:]
                ant_ra_dec = ant_ra_dec.chunk({"time":vis_dataset[sel_parms['data']].chunks[0][0]})
            else:
                antenna_ids = mxds.antenna_ids.data
                field_dataset = mxds.attrs['FIELD']
                field_id = np.max(vis_dataset.FIELD_ID,axis=1).compute() #np.max ignores int nan values (nan values are large negative numbers for int).
                n_field = field_dataset.dims['d0']
                ant_ra_dec = field_dataset.PHASE_DIR.isel(d0=field_id)
                if n_field != 1:
                    ant_ra_dec = ant_ra_dec[:,0,:]
                ant_ra_dec = ant_ra_dec.expand_dims('ant',1)
                n_ant = len(antenna_ids)
                ant_ra_dec = da.tile(ant_ra_dec.data,(1,n_ant,1))
                
                time_chunksize = mxds.attrs[sel_parms['xds']][sel_parms['data']].chunks[0][0]
                ant_ra_dec =  xr.DataArray(ant_ra_dec,{'time':vis_dataset.time,'ant':antenna_ids}, dims=('time','ant','pair')).chunk({'time':time_chunksize,'ant':n_ant,'pair':2})
    
            
            pa, cf_pa_centers, pa_diff = _calc_parallactic_angles_for_gcf(mxds,ant_ra_dec,_gcf_parms,_sel_parms)
            
            #print(pa.data.compute())
            #print(cf_pa_centers.data.compute())
            
            #pa_diff = pa_diff.data.compute()
            #print('Housten we got a problem',pa_diff[np.abs(pa_diff) > _gcf_parms['pa_step']],_gcf_parms['pa_step'])
            ################################################################################################################################
            
            ####################################################### Channel ####################################################
            chan_map, cf_pb_freq = _create_chan_map(mxds,_gcf_parms,_sel_parms)
            
            #print(chan_map)
            #print(cf_pb_freq)
            
            
            ################################################################################################################################
            #pa_pair_map, timexbaseline
            #pa_pairs, unique pa pairs x 2
            #pa_centers, pa_centers
            
            #print(cf_time_map.data.compute())
            #print(pa_centers.data.compute())
            #print(pa_dif.data.compute())
        
        else:
            print('#########  Using ', _gcf_parms['a_function'], 'function')
    else:
        pa = None
    
    ###################################################### W_TERM #####################################################################
    if _gcf_parms['w_term']:
        print('#########  Creating w_term coordinates')
        cf_w = _create_w_map(mxds,_gcf_parms,_grid_parms,_sel_parms)
        
    ###################################################### Phase Gradients ############################################################
    if _gcf_parms['phase_gradient_term']:
        print('#########  Creating pointing coordinates')
        
        if gcf_parms['use_pointing_table_phase_gradient']:
            ant_ra_dec = mxds.POINTING.DIRECTION.interp(time=vis_dataset.time,assume_sorted=False,method=gcf_parms['interpolation_method'])[:,:,0,:]
            ant_ra_dec = ant_ra_dec.chunk({"time":vis_dataset[sel_parms['data']].chunks[0][0]}).data
            
            cf_pointing = _calc_phase_gradient_pointings(mxds,pointing_ra_dec,_gcf_parms,_sel_parms)
        else:
            antenna_ids = mxds.antenna_ids.data
            field_dataset = mxds.attrs['FIELD']
            field_id = np.max(vis_dataset.FIELD_ID,axis=1).compute() #np.max ignores int nan values (nan values are large negative numbers for int).
            n_field = field_dataset.dims['d0']
            ant_ra_dec = field_dataset.PHASE_DIR.isel(d0=field_id)
            if n_field != 1:
                ant_ra_dec = ant_ra_dec[:,0,:]
            ant_ra_dec = ant_ra_dec.expand_dims('ant',1)
            n_ant = len(antenna_ids)
            ant_ra_dec = da.tile(ant_ra_dec.data,(1,n_ant,1))
            time_chunksize = mxds.attrs[sel_parms['xds']][sel_parms['data']].chunks[0][0]
            ant_ra_dec =  xr.DataArray(ant_ra_dec,{'time':vis_dataset.time,'ant':antenna_ids}, dims=('time','ant','pair')).chunk({'time':time_chunksize,'ant':n_ant,'pair':2})
            
            cf_pointing = field_dataset.PHASE_DIR[:,0,:]
        
        print(cf_pointing)
        #cf_pointing = _calc_phase_gradient_pointings(mxds,pointing_ra_dec,_gcf_parms,_sel_parms)
        #print('cf_pointing',cf_pointing)
        
        #if no pointing use fields
        
    
    #Dimension on which to do parallelization time, baseline, chan
#    print(vis_dataset)
#    print('&&&&&&&&&&&&&&&&&&&')
#    print(beam_map,'\n ***************')                # baseline*
#
#    print(pa,'\n ***************')                      # time* x ant
#    print(cf_pa_centers,'\n ***************')           # cf_pa (no chunks)
#    print(vis_dataset.ANTENNA1,'\n ***************')    # time* x baseline*
#    print(vis_dataset.ANTENNA2,'\n ***************')    # time* x baseline*
#    print(mxds.antenna_ids,'\n ***************')        # ant
#
#    print('chan_map', chan_map,'\n ***************')                # chan*
#
#
#    print('cf_w',cf_w,'\n ***************')                    # cf_w
#    print('W',vis_dataset.UVW[:,:,2],'\n ***************')  # time* x baseline*
#
#    #######
#    print('ant_ra_dec',ant_ra_dec,'\n ***************')  # time* x ant
#    print('cf_pointing', cf_pointing,'\n ***************')         # cf_pointing x 2
#
#    #########
#    print('cf_beam_pair_id',cf_beam_pair_id,'\n ***************')     # cf_beam_pair
#    print('cf_pb_freq',cf_pb_freq,'\n ***************')               # cf_freq
#
#    print('&&&&&&&&&&&&&&&&&&&')
#
    #gcf_dataset = xr.Dataset()
    
    create_cf_map(mxds,gcf_dataset,beam_map,cf_beam_pair_id,pa,cf_pa_centers,chan_map, cf_pb_freq,cf_w,cf_pointing,ant_ra_dec,sel_parms)
    
    
    '''
    rad_to_deg =  180/np.pi
    phase_center = mxds.#gcf_parms['image_phase_center']
    w = WCS(naxis=2)
    w.wcs.crpix = grid_parms['image_size_padded']//2
    w.wcs.cdelt = grid_parms['cell_size']*rad_to_deg
    w.wcs.crval = phase_center*rad_to_deg
    w.wcs.ctype = ['RA---SIN','DEC--SIN']
    
    #print('field_phase_dir ',field_phase_dir)
    pix_dist = np.array(w.all_world2pix(field_phase_dir[0]*rad_to_deg, 1)) - grid_parms['image_size_padded']//2
    pix = -(pix_dist)*2*np.pi/(grid_parms['image_size_padded']*gcf_parms['oversampling'])
    '''
    
    ###################################################################################################################################

def create_cf_map(mxds,gcf_dataset,beam_map,cf_beam_pair_id,pa,cf_pa_centers,chan_map, cf_pb_freq,cf_w,cf_pointing,pointing_ra_dec,sel_parms):
    import itertools
    from ._imaging_utils._general import _ndim_list

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
        print('c_time,c_baseline,c_chan',c_time,c_baseline,c_chan)
        chunk_cf_and_pg = dask.delayed(_cf_map_wrap)(
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
        cf_map_list[c_time][c_baseline][c_chan] = da.from_delayed(chunk_cf_and_pg[1], (chunk_sizes[0][c_time],chunk_sizes[1][c_baseline],chunk_sizes[0][c_chan]),dtype=np.int)
        
        pg_parms_indx_list[i_chunk] = chunk_cf_and_pg[2] #can't do from_delayed since number of elements are unkown
        pg_map_list[c_time][c_baseline] = da.from_delayed(chunk_cf_and_pg[3], (chunk_sizes[0][c_time],chunk_sizes[1][c_baseline]),dtype=np.int)
            
        i_chunk = i_chunk+1
        
    cf_map = da.block(cf_map_list) #Awesome function
    pg_map = da.block(pg_map_list)
     
    #print(cf_map.compute())
    #print(cf_parms_indx_list[0].compute())
    
    cf_parms_indx = _tree_unique_list(cf_parms_indx_list)
    pg_parms_indx = _tree_unique_list(pg_parms_indx_list)
    
    print(cf_parms_indx.compute().shape)
    
    print(pg_parms_indx.compute().shape)
    
    
    
    
def _tree_unique_list(list_to_sum):
    import dask.array as da
    while len(list_to_sum) > 1:
        new_list_to_sum = []
        for i in range(0, len(list_to_sum), 2):
            if i < len(list_to_sum) - 1:
                lazy = dask.delayed(_find_unique_subset)(list_to_sum[i],list_to_sum[i+1])
            else:
                lazy = list_to_sum[i]
            new_list_to_sum.append(lazy)
        list_to_sum = new_list_to_sum
    return list_to_sum[0]
    
import pandas as pd
def _find_unique_subset(a,b):
    a_pd = pd.DataFrame(a)
    b_pd = pd.DataFrame(b)
    
    a_pd = a_pd.append(b_pd)
    
    a_pd = a_pd.drop_duplicates(a_pd.columns[-1])
    #print(a_pd.columns[-1])
    return a_pd.to_numpy()
    
@jit(nopython=True,cache=True,nogil=True)
def _cf_map_wrap(beam_map,beam_ids,cf_beam_pair_id,pa,cf_pa,ant_1,ant_2,ant_ids,chan_map,freq_chan,cf_chan,w,cf_w,pointing_ra_dec,cf_pointing):
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
    
    cf_indx_list = [np.array([-42,-42,-42,-42,-42,-42,-42])] #Can't have an empty list need to tell Numba what type it is
    cf_map = np.zeros((n_time,n_baseline,n_chan),numba.i8)
    #cf_map = np.zeros((n_time,n_baseline,n_chan),np.int)
    
    pg_indx_list = [np.array([-42,-42,-42])] #Can't have an empty list need to tell Numba what type it is
    #pg_indx_list = np.zeros((n_time * n_baseline),numba.i8)
    
    pg_map = np.zeros((n_time,n_baseline),numba.i8)
    #pg_map = np.zeros((n_time,n_baseline),np.int)
    
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
                
                cf_pa1_indx = find_cf_angle(cf_pa,pa1)
                cf_pa2_indx = find_cf_angle(cf_pa,pa1)
                
                ############Pointing calcs############
                point1 = pointing_ra_dec[i_time,a1_indx]
                point2 = pointing_ra_dec[i_time,a2_indx]
                
                cf_point1_indx = find_cf_ra_dec(cf_pointing,point1)
                cf_point2_indx = find_cf_ra_dec(cf_pointing,point2)
                
                ############Calculate Flat Index for Phase Gradients (pg)############
                # [P1,P2]
                i_pg, n_pg = combine_indx_combination(cf_point1_indx,cf_point2_indx,n_cf_point,n_cf_point)
                pg_indx_info = np.array([cf_point1_indx,cf_point2_indx,i_pg]) #-42 is a dummy value. If it appears in the final result something has gone wrong
                
                pg_map[i_time,i_baseline] = i_pg

           
                pg_indx_list.append(pg_indx_info)
                
                ######################################
                w_val = np.abs(w[i_time,i_baseline])
            
                for i_chan in range(n_chan):
                    ############W calcs############
                    w_val = w_val*c/freq_chan[i_chan]
                    #print('cf_w,w_val',cf_w,w_val)
                    cf_w_indx = find_cf_val(cf_w,w_val)
                    
                    ############Chan calcs############
                    cf_c_indx = chan_map[i_chan]
                    
                    ############Calculate Flat Index for cf (convolution function)############
                    # [PA1,B1,PA2,B1,W,C,CF]
                    cf_indx_info = np.array([cf_pa1_indx,cf_beam1_indx,cf_pa2_indx,cf_beam2_indx,cf_w_indx,cf_c_indx,-42]) #-42 is a dummy value. If it appears in the final result something has gone wrong
                    n_cf = calc_cf_flat_indx(cf_indx_info,n_cf_beam,n_cf_pa,n_cf_w,n_cf_c)
                    
                    cf_map[i_time,i_baseline,i_chan] = cf_indx_info[-1]
                    
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
                    
                    
    cf_indx_list.pop(0)
    #cf_indx_list = np.vstack(cf_indx_list)
    #cf_indx_list = np.stack(cf_indx_list,axis=0)
    #cf_indx_list = np.asarray(cf_indx_list)
    
    #Convert list of arrays to array (numpy functions don't work in numba). Also avoid tight for loop.
    cf_indx_arr = np.zeros((len(cf_indx_list),len(cf_indx_list[0])),numba.i8)
    #cf_indx_arr = np.zeros((len(cf_indx_list),len(cf_indx_list[0])),int)
    
    n_cf_flat = len(cf_indx_list)
    n_i = len(cf_indx_list[0])
    
    for jj in range(n_cf_flat ):
        for ii in range(n_i):
            cf_indx_arr[jj,ii] = cf_indx_list[jj][ii]
     
    pg_indx_list.pop(0)
    #pg_indx_list = np.vstack(pg_indx_list)
    #pg_indx_list = np.stack(pg_indx_list,axis=0)
    #pg_indx_list = np.asaray(pg_indx_list)
    
    #Convert list of arrays to array (numpy functions don't work in numba). Also avoid tight for loop.
    pg_indx_arr = np.zeros((len(pg_indx_list),len(pg_indx_list[0])),numba.i8)
    #pg_indx_arr = np.zeros((len(pg_indx_list),len(pg_indx_list[0])),int)
        
    n_pg_flat = len(pg_indx_list)
    n_i = len(pg_indx_list[0])
    
    for jj in range(n_pg_flat):
        for ii in range(n_i):
            pg_indx_arr[jj,ii] = pg_indx_list[jj][ii]
    
    
    
#    print('cf_indx_list',cf_indx_arr)
#    print(len(cf_indx_list),n_cf)
#
#    print('pg_indx_list',pg_indx_arr)
#    print(len(pg_indx_list),n_pg)
#
    
    #print(cf_indx_arr.shape)
    #print(cf_map.shape)
    
    #print(pg_indx_arr.shape)
    #print(pg_map.shape)
    
    return cf_indx_arr, cf_map, pg_indx_arr, pg_map
    

#def add_to_unqiue_list(cf_indx_list,cf_indx_info):
    
    
    
'''
##Should it be i4 or i8. Try this function for large dataset instead of np.where
#@jit(nopython=True,cache=True,nogil=True)
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    idx = -42 #can't return none and integer so use -42
    return idx
'''

    
#@jit("i4(i4[:],i4,i4,i4,i4)",nopython=True,cache=True,nogil=True)
#@jit("void(i8[:],i8,i8,i8,i8)",nopython=True,cache=True,nogil=True)
@jit(nopython=True,cache=True,nogil=True)
def calc_cf_flat_indx(cf_indx_info,n_beam,n_pa,n_w,n_c):
    #print('1*******')
    p1, b1, p2, b2, w, c, x = cf_indx_info

    #print(n_pa,n_beam)
    i1,n1 = combine_indx_permutation(p1,b1,n_pa,n_beam)
    #print(i1, n1)
    i2,n2 = combine_indx_permutation(p2,b2,n_pa,n_beam)
    #print(i2, n2)
    i3,n3 = combine_indx_combination(i1,i2,n1,n2)
    #print(i3, n3)
    i4,n4 = combine_indx_permutation(i3,w,n3,n_w)
    #print(i4, n4)
    i5,n5 = combine_indx_permutation(i4,c,n4,n_c)
    #print(i5, n5)
    cf_indx_info[-1] = i5
    
    #print(n5)
    
    #print('in calc_cf_flat_indx',cf_indx_info)
    #print('2*******')
    return n5
    
    
#@jit("void(i8,i8,i8,i8,i8[:])",nopython=True,cache=True,nogil=True)
@jit(nopython=True,cache=True,nogil=True)
def combine_indx_permutation(i1,i2,n_i1,n_i2):
    n_comb = n_i1*n_i2
    if n_i1 <= n_i2:
        i_comb = i1*n_i2 + i2
    else:
        i_comb = i1 + i2*n_i1
    
    #result[0] = i_comb
    #result[1] = n_comb
    #print(i1,i2,n_i1,n_i2,i_comb,n_comb)
    return i_comb,n_comb
    
#@jit("void(i8,i8,i8,i8,i8[:])",nopython=True,cache=True,nogil=True)
@jit(nopython=True,cache=True,nogil=True)
def combine_indx_combination(i1,i2,n_i1,n_i2):
    if n_i1 <= n_i2:
        if i1 > i2:
            temp = i2
            i2 = i1
            i1 = temp
    
        n_comb = n_i1*n_i2 - (n_i1-1)*n_i1//2
        i_comb = ((2*n_i2 -1)*i1 - i1**2)//2 + i2
    else:
        if i1 < i2:
            temp = i2
            i2 = i1
            i1 = temp
    
        n_comb = n_i1*n_i2 - (n_i2-1)*n_i2//2
        i_comb = i1 + ((2*n_i1 -1)*i2 - i2**2)//2
        
    #print(i1,i2,n_i1,n_i2,i_comb,n_comb)
    return i_comb,n_comb
    

            
@jit(nopython=True,cache=True,nogil=True)
def find_cf_val(cf_pa,pa):

    min_dif = -42.0 #Dummy value
    for jj in range(len(cf_pa)):
        ang_dif = np.abs(pa-cf_pa[jj])

        if (min_dif < 0) or (min_dif > ang_dif):
            min_dif = ang_dif
            cf_pa_indx = jj
            
    return cf_pa_indx

@jit(nopython=True,cache=True,nogil=True)
def find_cf_angle(cf_pa,pa):
    min_dif = 42.0 #Dummy value
    for jj in range(len(cf_pa)):
        #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
        ang_dif = pa-cf_pa[jj]
        ang_dif = np.abs((ang_dif + np.pi)%(2*np.pi) - np.pi)
        
        if min_dif > ang_dif:
            min_dif = ang_dif
            cf_pa_indx = jj
            
    return cf_pa_indx
    
@jit(nopython=True,cache=True,nogil=True)
def find_cf_ra_dec(cf_pointing,point):
    min_dis = 42.0 #Dummy value
    for jj in range(len(cf_pointing)):
        #https://stjerneskinn.com/angular-distance-between-stars.htm
        #http://spiff.rit.edu/classes/phys373/lectures/radec/radec.html
        ra = cf_pointing[jj,0]
        dec = cf_pointing[jj,1]
        dis = np.sqrt(((ra-point[0])*np.cos(dec))**2 + (dec-point[1])**2) #approximation
        
        if min_dis > dis:
            min_dis = dis
            cf_pointing_indx = jj

    return cf_pointing_indx

    
def transform_pointing_table(mxds,gcf_parms,sel_parms):
    ### Using pointing table
    vis_dataset = mxds.attrs[sel_parms['xds']]
    
    '''
    print('****************')
    print(vis_dataset)
    print('****************')
    print(mxds.POINTING)
    print('****************')
    '''
    
    
    antenna_ids = mxds.antenna_ids.data
    point_ant_ids = mxds.POINTING.antenna_id.data.compute()
    point_times = mxds.POINTING.TIME.data.compute()
    point_unique_times = np.unique(point_times)
    n_point_time = len(point_unique_times)
    n_ant = len(antenna_ids)
    
    ra_dec = np.zeros((n_point_time,n_ant,2))
    ra_dec[:] = np.NaN
    
    for point_indx,point_ant_id in enumerate(point_ant_ids):
        ant_indx = np.where(antenna_ids==point_ant_id)[0][0]
        point_time = point_times[point_indx]
        time_indx = np.where(point_unique_times==point_times[point_indx])[0][0]
        ra_dec[time_indx,ant_indx,:] = mxds.POINTING.DIRECTION[point_indx,0,:]
        
    ra_dec = xr.DataArray(ra_dec, dims=['time', 'ant','pair'],coords=dict(time=point_unique_times,ant=antenna_ids))
    mxds.attrs['pointing_ra_dec'] = ra_dec
    

    
    '''
    # Create Framework
        # Decide convsize, support from oversampling for PS, A, W term.
        # Create Maps between visibilities and gcf.
    print('global_dataset.ANT_DISH_DIAMETER',global_dataset.ANT_DISH_DIAMETER.data.compute())
    calculate_conv_size(vis_dataset, list_zpc_dataset,_grid_parms)
    conv_size = np.array([2048,2048]) #Temporary
    
    
    w_values = _calculate_w_list(_gcf_parms,_grid_parms)
    _gcf_parms['conv_size'] = conv_size
    w_sky = _calc_w_sky(w_values,_gcf_parms,_grid_parms)
    
    #########################
    #PA should be function of Time and Antenna position (if an antenna is)
    PA, x = _calc_parallactic_angles_for_gcf(vis_dataset,global_dataset,_gcf_parms,_sel_parms)
    #print(PA)
    
    cf_chan_map, pb_freq = _create_cf_chan_map(vis_dataset.chan,_gcf_parms['chan_tolerance_factor'])
    #print(cf_chan_map)
    #print(pb_freq)
    
    
    ps_sky = _create_prolate_spheroidal_image_2D(_gcf_parms['conv_size'])
    '''
    
#    plt.figure()
#    plt.imshow(ps_sky)
#    plt.show()
#

    

    
    # Create PS, A, W term
    # FT(iFT(PS) x iFT(A) x iFT(W)) and Normalize
    # Create Phase gradients (include VLASS pointing corrections)
    
    
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
    


    
###################
###################
###################


#@jit(nopython=True,cache=True,nogil=True)
#def _cf_map_wrap(beam_map,beam_ids,cf_beam_pair_id,pa,cf_pa,ant_1,ant_2,ant_ids,chan_map,freq_chan,cf_chan,w,cf_w,pointing_ra_dec,cf_pointing):
##    print(beam_map.shape)
##    print(beam_ids.shape)
##    print(cf_beam_pair_id)
##    print(pa.shape)
##    print(cf_pa.shape)
##    print(ant_1.shape)
##    print(ant_2.shape)
##    print(ant_ids.shape)
##    print(chan_map.shape)
##    print(w.shape)
##    print(cf_w.shape)
##    print(pointing_ra_dec.shape)
##    print(cf_pointing.shape)
#
#    n_time = ant_1.shape[0]
#    n_baseline = ant_1.shape[1]
#    n_chan = chan_map.shape[0]
#
#    print('n_time', n_time, 'n_baseline', n_baseline, 'n_chan', n_chan)
#    c = 299792458.0
#
#    n_cf_beam = len(beam_ids) #not pairs
#    n_cf_pa = len(cf_pa)
#    n_cf_w = len(cf_w)
#    n_cf_c = len(cf_chan)
#    n_cf_point = len(cf_pointing)
#
#    cf_indx_list = [np.array([-42,-42,-42,-42,-42,-42,-42])] #Can't have an empty list need to tell Numba what type it is
#    cf_map = np.zeros((n_time,n_baseline,n_chan),numba.i8)
#    #cf_map = np.zeros((n_time,n_baseline,n_chan),np.int)
#
#    pg_indx_list = [np.array([-42,-42,-42])] #Can't have an empty list need to tell Numba what type it is
#    pg_map = np.zeros((n_time,n_baseline),numba.i8)
#    #pg_map = np.zeros((n_time,n_baseline),np.int)
#
#    for i_time in range(n_time):
#        for i_baseline in range(n_baseline):
#            if not(np.isnan(w[i_time,i_baseline])) and not(np.isnan(ant_1[i_time,i_baseline])) and not(np.isnan(ant_2[i_time,i_baseline])):
#                ############Beam calcs############
#                beam_pair_indx = beam_map[i_baseline]
#                cf_beam1_indx = np.where(beam_ids == cf_beam_pair_id[beam_pair_indx,0])[0][0]
#                cf_beam2_indx = np.where(beam_ids == cf_beam_pair_id[beam_pair_indx,1])[0][0]
#
#                ############PA calcs############
#                a1_indx = np.where(ant_ids ==  ant_1[i_time,i_baseline])[0][0]
#                a2_indx = np.where(ant_ids ==  ant_2[i_time,i_baseline])[0][0]
#
#                #Add pa map if timexant is large and takes too long to compute
#                pa1 = pa[i_time,a1_indx]
#                pa2 = pa[i_time,a2_indx]
#
#                cf_pa1_indx = find_cf_angle(cf_pa,pa1)
#                cf_pa2_indx = find_cf_angle(cf_pa,pa1)
#
#                ############Pointing calcs############
#                point1 = pointing_ra_dec[i_time,a1_indx]
#                point2 = pointing_ra_dec[i_time,a2_indx]
#
#                cf_point1_indx = find_cf_ra_dec(cf_pointing,point1)
#                cf_point2_indx = find_cf_ra_dec(cf_pointing,point2)
#
#                ############Calculate Flat Index for Phase Gradients (pg)############
#                # [P1,P2]
#                i_pg, n_pg = combine_indx_combination(cf_point1_indx,cf_point2_indx,n_cf_point,n_cf_point)
#                pg_indx_info = np.array([cf_point1_indx,cf_point2_indx,i_pg]) #-42 is a dummy value. If it appears in the final result something has gone wrong
#
#                pg_map[i_time,i_baseline] = i_pg
#
#                #Nasty code needed due to working with list in numba. Can't put in a separate function, due to typed list inefficiencies.
#                found = False
#                end_of_list = False
#                i_list = 0
#                n_list = len(pg_indx_list)
#                while not(found) and not(end_of_list):
#                    #print(i_list,n_list)
#                    if pg_indx_list[i_list][-1] == pg_indx_info[-1]:
#                        found = True
#                    i_list = i_list+1
#                    if i_list >= n_list:
#                        end_of_list = True
#
#                if not(found):
#                    pg_indx_list.append(pg_indx_info)
#
#                ######################################
#                w_val = np.abs(w[i_time,i_baseline])
#
#                for i_chan in range(n_chan):
#                    ############W calcs############
#                    w_val = w_val*c/freq_chan[i_chan]
#                    #print('cf_w,w_val',cf_w,w_val)
#                    cf_w_indx = find_cf_val(cf_w,w_val)
#
#                    ############Chan calcs############
#                    cf_c_indx = chan_map[i_chan]
#
#                    ############Calculate Flat Index for cf (convolution function)############
#                    # [PA1,B1,PA2,B1,W,C,CF]
#                    cf_indx_info = np.array([cf_pa1_indx,cf_beam1_indx,cf_pa2_indx,cf_beam2_indx,cf_w_indx,cf_c_indx,-42]) #-42 is a dummy value. If it appears in the final result something has gone wrong
#                    n_cf = calc_cf_flat_indx(cf_indx_info,n_cf_beam,n_cf_pa,n_cf_w,n_cf_c)
#
#                    cf_map[i_time,i_baseline,i_chan] = cf_indx_info[-1]
#
#                    #Nasty code needed due to working with list in numba. Can't put in a separate function, due to typed list inefficiencies.
#                    found = False
#                    end_of_list = False
#                    i_list = 0
#                    n_list = len(cf_indx_list)
#                    while not(found) and not(end_of_list):
#                        #print(i_list,n_list)
#                        if cf_indx_list[i_list][-1] == cf_indx_info[-1]:
#                            found = True
#                        i_list = i_list+1
#                        if i_list >= n_list:
#                            end_of_list = True
#
#                    if not(found):
#                        cf_indx_list.append(cf_indx_info)
#
#
#    cf_indx_list.pop(0)
#    #cf_indx_list = np.vstack(cf_indx_list)
#    #cf_indx_list = np.stack(cf_indx_list,axis=0)
#    #cf_indx_list = np.asarray(cf_indx_list)
#
#    #Convert list of arrays to array (numpy functions don't work in numba). Also avoid tight for loop.
#    cf_indx_arr = np.zeros((len(cf_indx_list),len(cf_indx_list[0])),numba.i8)
#    #cf_indx_arr = np.zeros((len(cf_indx_list),len(cf_indx_list[0])),int)
#
#    n_cf_flat = len(cf_indx_list)
#    n_i = len(cf_indx_list[0])
#
#    for jj in range(n_cf_flat ):
#        for ii in range(n_i):
#            cf_indx_arr[jj,ii] = cf_indx_list[jj][ii]
#
#    pg_indx_list.pop(0)
#    #pg_indx_list = np.vstack(pg_indx_list)
#    #pg_indx_list = np.stack(pg_indx_list,axis=0)
#    #pg_indx_list = np.asaray(pg_indx_list)
#
#    #Convert list of arrays to array (numpy functions don't work in numba). Also avoid tight for loop.
#    pg_indx_arr = np.zeros((len(pg_indx_list),len(pg_indx_list[0])),numba.i8)
#    #pg_indx_arr = np.zeros((len(pg_indx_list),len(pg_indx_list[0])),int)
#
#    n_pg_flat = len(pg_indx_list)
#    n_i = len(pg_indx_list[0])
#
#    for jj in range(n_pg_flat):
#        for ii in range(n_i):
#            pg_indx_arr[jj,ii] = pg_indx_list[jj][ii]
#
#
#
##    print('cf_indx_list',cf_indx_arr)
##    print(len(cf_indx_list),n_cf)
##
##    print('pg_indx_list',pg_indx_arr)
##    print(len(pg_indx_list),n_pg)
##
#
#    print(cf_indx_arr.shape)
#    print(cf_map.shape)
#
#    print(pg_indx_arr.shape)
#    print(pg_map.shape)
#
#    return cf_indx_arr, cf_map, pg_indx_arr, pg_map
#
#
