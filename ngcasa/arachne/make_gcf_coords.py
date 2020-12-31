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
    from ._imaging_utils._a_term import _create_cf_chan_map, _create_cf_baseline_map
    from ._imaging_utils._w_term import _calculate_w_list, _calc_w_sky
    from ._imaging_utils._ps_term import _create_prolate_spheroidal_image_2D
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
    
    ##################################################### PS_TERM #####################################################################
    if gcf_parms['ps_term']:
        print('#########  Creating ps_term coordinates')
    
    
    ##################################################### A_TERM ######################################################################
    if gcf_parms['a_term']:
        print('#########  Creating a_term coordinates')
        
        if gcf_parms['a_function'] == 'zp':
            print('#########  Using ', gcf_parms['a_function'], 'function')
            
            #n_unique_ant = len(_gcf_parms['list_dish_diameters'])
            cf_ant_model_baseline_map,beam_model_pairs = _create_cf_baseline_map(mxds,sel_parms)
            
            #print('cf_ant_model_baseline_map',cf_ant_model_baseline_map)
            #print('beam_model_pairs',beam_model_pairs)
            
            try:
                transform_pointing_table(mxds,gcf_parms,sel_parms) #temp function, should be included in convert_ms
            except:
                print('Conversion of Pointing Table Failed')
            
            #PA should be function of Time and Antenna position (if an antenna is)
            cf_time_map, pa_centers, pa_dif, pa_map = _calc_parallactic_angles_for_gcf(mxds,beam_model_pairs,_gcf_parms,_sel_parms)
            
            #print(cf_time_map.data.compute())
            #print(pa_centers.data.compute())
            #print(pa_dif.data.compute())
        
        else:
            print('#########  Using ', gcf_parms['a_function'], 'function')
    
    ###################################################### W_TERM #####################################################################
    if gcf_parms['w_term']:
        print('#########  Creating w_term coordinates')
        
    ###################################################### Phase Gradients ############################################################
    #if gcf_parms['do_pointing']:
    #    print('#########  Creating pointing coordinates')
    ###################################################################################################################################

    
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
    


    
