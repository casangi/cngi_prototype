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

import numpy as np
import cngi._utils._constants as const

'''
    Calculate gridding convolution functions (GCF) as specified for standard, widefield and mosaic imaging.
    Construct a GCF cache (persistent or on-the-fly)

    Options : Choose a list of effects to include
    
    - PSterm : Prolate-Spheroidal gridding kernel (anti-aliasing function)
    - Aterm : Use PB model and Aperture Illumination Function per antenna to construct a GCF per baseline
        - Include support for Heterogeneous Arrays where Aterm is different per antenna
        - Include support for time-varying PB and AIF models. Rotation, etc.
    - Wterm : FT of Fresnel kernel per baseline
'''

def make_gridding_convolution_function(mxds, gcf_parms, grid_parms, sel_parms):
    """
    Currently creates a gcf to correct for the primary beams of antennas and supports heterogenous arrays (antennas with different dish sizes).
    Only the airy disk and ALMA airy disk model is implemented.
    In the future support will be added for beam squint, pointing corrections, w projection, and including a prolate spheroidal term.
    
    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input visibility dataset.
    gcf_parms : dictionary
    gcf_parms['function'] : {'casa_airy'/'airy'}, default = 'casa_airy'
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
    print('######################### Start make_gridding_convolution_function #########################')
    
    from ._imaging_utils._check_imaging_parms import _check_pb_parms
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms
    from ._imaging_utils._check_imaging_parms import _check_grid_parms, _check_gcf_parms
    from ._imaging_utils._gridding_convolutional_kernels import _create_prolate_spheroidal_kernel_2D, _create_prolate_spheroidal_image_2D
    from ._imaging_utils._remove_padding import _remove_padding
    import numpy as np
    import dask.array as da
    import copy, os
    import xarray as xr
    import itertools
    import dask
    import dask.array.fft as dafft
    import time
    
    import matplotlib.pylab as plt
    
    #Deep copy so that inputs are not modified
    _mxds = mxds.copy(deep=True)
    _gcf_parms = copy.deepcopy(gcf_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    _sel_parms = copy.deepcopy(sel_parms)
    
    ##############Parameter Checking and Set Defaults##############
    assert('xds' in _sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_dataset = _mxds.attrs[sel_parms['xds']]
    
    assert('xds' in _sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_dataset = _mxds.attrs[sel_parms['xds']]
    
    _check_sel_parms(_vis_dataset,_sel_parms)

    #_gcf_parms['basline_ant'] = np.unique([_vis_dataset.ANTENNA1.max(axis=0), _vis_dataset.ANTENNA2.max(axis=0)], axis=0).T
    _gcf_parms['basline_ant'] = np.array([_vis_dataset.ANTENNA1.values, _vis_dataset.ANTENNA2.values]).T
    
    _gcf_parms['freq_chan'] = _vis_dataset.chan.values
    _gcf_parms['pol'] = _vis_dataset.pol.values
    _gcf_parms['vis_data_chunks'] = _vis_dataset.DATA.chunks
    
    
    _gcf_parms['field_phase_dir'] = mxds.FIELD.PHASE_DIR[:,0,:].data.compute()
    field_id = mxds.FIELD.field_id.data#.compute()
    
    
    #print(_gcf_parms['field_phase_dir'])
    #_gcf_parms['field_phase_dir'] = np.array(global_dataset.FIELD_PHASE_DIR.values[:,:,vis_dataset.attrs['ddi']])
    
    assert(_check_gcf_parms(_gcf_parms)), "######### ERROR: gcf_parms checking failed"
    assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"

    if _gcf_parms['function'] == 'airy':
        from ._imaging_utils._make_pb_symmetric import _airy_disk_rorder
        pb_func = _airy_disk_rorder
    elif _gcf_parms['function'] == 'casa_airy':
        from ._imaging_utils._make_pb_symmetric import _casa_airy_disk_rorder
        pb_func = _casa_airy_disk_rorder
    else:
        assert(False), "######### ERROR: Only airy and casa_airy function has been implemented"
        
    #For now only a_term works
    _gcf_parms['a_term'] =  True
    _gcf_parms['ps_term'] =  False
        
    _gcf_parms['resize_conv_size'] = (_gcf_parms['max_support'] + 1)*_gcf_parms['oversampling']
    #resize_conv_size = _gcf_parms['resize_conv_size']
    
    if _gcf_parms['ps_term'] == True:
        '''
        ps_term = _create_prolate_spheroidal_kernel_2D(_gcf_parms['oversampling'],np.array([7,7])) #This is only used with a_term == False. Support is hardcoded to 7 until old ps code is replaced by a general function.
        center = _grid_parms['image_center']
        center_embed = np.array(ps_term.shape)//2
        ps_term_padded = np.zeros(_grid_parms['image_size'])
        ps_term_padded[center[0]-center_embed[0]:center[0]+center_embed[0],center[1]-center_embed[1] : center[1]+center_embed[1]] = ps_term
        ps_term_padded_ifft = dafft.fftshift(dafft.ifft2(dafft.ifftshift(da.from_array(ps_term_padded))))

        ps_image = da.from_array(_remove_padding(_create_prolate_spheroidal_image_2D(_grid_parms['image_size_padded']),_grid_parms['image_size']),chunks=_grid_parms['image_size'])

        #Effecively no mapping needed if ps_term == True and a_term == False
        cf_baseline_map = np.zeros((len(_gcf_parms['basline_ant']),),dtype=int)
        cf_chan_map = np.zeros((len(_gcf_parms['freq_chan']),),dtype=int)
        cf_pol_map = np.zeros((len(_gcf_parms['pol']),),dtype=int)
        '''

    if _gcf_parms['a_term'] == True:
        n_unique_ant = len(_gcf_parms['list_dish_diameters'])
        
        cf_baseline_map,pb_ant_pairs = create_cf_baseline_map(_gcf_parms['unique_ant_indx'],_gcf_parms['basline_ant'],n_unique_ant)

        cf_chan_map, pb_freq = create_cf_chan_map(_gcf_parms['freq_chan'],_gcf_parms['chan_tolerance_factor'])
        #print('****',pb_freq)
        pb_freq = da.from_array(pb_freq,chunks=np.ceil(len(pb_freq)/_gcf_parms['a_chan_num_chunk'] ))

        cf_pol_map = np.zeros((len(_gcf_parms['pol']),),dtype=int) #create_cf_pol_map(), currently treating all pols the same
        pb_pol = da.from_array(np.array([0]),1)

        n_chunks_in_each_dim = [pb_freq.numblocks[0],pb_pol.numblocks[0]]
        iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]))
        chan_chunk_sizes = pb_freq.chunks
        pol_chunk_sizes = pb_pol.chunks

        #print(pb_freq, pb_pol,pol_chunk_sizes)
        list_baseline_pb = []
        list_weight_baseline_pb_sqrd = []
        for c_chan, c_pol in iter_chunks_indx:
                #print('chan, pol ',c_chan,c_pol)
                _gcf_parms['ipower'] = 1
                delayed_baseline_pb = dask.delayed(make_baseline_patterns)(pb_freq.partitions[c_chan],pb_pol.partitions[c_pol],dask.delayed(pb_ant_pairs),dask.delayed(pb_func),dask.delayed(_gcf_parms),dask.delayed(_grid_parms))

                list_baseline_pb.append(da.from_delayed(delayed_baseline_pb,(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],_grid_parms['image_size_padded'][0],_grid_parms['image_size_padded'][1]),dtype=np.double))

                _gcf_parms['ipower'] = 2
                delayed_weight_baseline_pb_sqrd = dask.delayed(make_baseline_patterns)(pb_freq.partitions[c_chan],pb_pol.partitions[c_pol],dask.delayed(pb_ant_pairs),dask.delayed(pb_func),dask.delayed(_gcf_parms),dask.delayed(_grid_parms))

                list_weight_baseline_pb_sqrd.append(da.from_delayed(delayed_weight_baseline_pb_sqrd,(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],_grid_parms['image_size_padded'][0],_grid_parms['image_size_padded'][1]),dtype=np.double))


        baseline_pb = da.concatenate(list_baseline_pb,axis=1)
        weight_baseline_pb_sqrd = da.concatenate(list_weight_baseline_pb_sqrd,axis=1)
        
        
#    x = baseline_pb.compute()
#    print("&*&*&*&",x.shape)
#    plt.figure()
#    plt.imshow(x[0,0,0,240:260,240:260])
#    plt.show()
    

    #Combine patterns and fft to obtain the gridding convolutional kernel
    #print(weight_baseline_pb_sqrd)

    dataset_dict = {}
    list_xarray_data_variables = []
    if (_gcf_parms['a_term'] == True) and (_gcf_parms['ps_term'] == True):
        conv_kernel = da.real(dafft.fftshift(dafft.fft2(dafft.ifftshift(ps_term_padded_ifft*baseline_pb, axes=(3, 4)), axes=(3, 4)), axes=(3, 4)))
        conv_weight_kernel = da.real(dafft.fftshift(dafft.fft2(dafft.ifftshift(weight_baseline_pb_sqrd, axes=(3, 4)), axes=(3, 4)), axes=(3, 4)))


        list_conv_kernel = []
        list_weight_conv_kernel = []
        list_conv_support = []
        iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]))
        for c_chan, c_pol in iter_chunks_indx:
                delayed_kernels_and_support = dask.delayed(resize_and_calc_support)(conv_kernel.partitions[:,c_chan,c_pol,:,:],conv_weight_kernel.partitions[:,c_chan,c_pol,:,:],dask.delayed(_gcf_parms),dask.delayed(_grid_parms))
                list_conv_kernel.append(da.from_delayed(delayed_kernels_and_support[0],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],_gcf_parms['resize_conv_size'][0],_gcf_parms['resize_conv_size'][1]),dtype=np.double))
                list_weight_conv_kernel.append(da.from_delayed(delayed_kernels_and_support[1],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],_gcf_parms['resize_conv_size'][0],_gcf_parms['resize_conv_size'][1]),dtype=np.double))
                list_conv_support.append(da.from_delayed(delayed_kernels_and_support[2],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],2),dtype=np.int))


        conv_kernel = da.concatenate(list_conv_kernel,axis=1)
        weight_conv_kernel = da.concatenate(list_weight_conv_kernel,axis=1)
        conv_support = da.concatenate(list_conv_support,axis=1)


        dataset_dict['SUPPORT'] = xr.DataArray(conv_support, dims=['conv_baseline','conv_chan','conv_pol','xy'])
        dataset_dict['PS_CORR_IMAGE'] = xr.DataArray(ps_image, dims=['l','m'])
        dataset_dict['WEIGHT_CONV_KERNEL'] = xr.DataArray(weight_conv_kernel, dims=['conv_baseline','conv_chan','conv_pol','u','v'])
    elif (_gcf_parms['a_term'] == False) and (_gcf_parms['ps_term'] == True):
        support = np.array([7,7])
        dataset_dict['SUPPORT'] = xr.DataArray(support[None,None,None,:], dims=['conv_baseline','conv_chan','conv_pol','xy'])
        conv_kernel = np.zeros((1,1,1,_gcf_parms['resize_conv_size'][0],_gcf_parms['resize_conv_size'][1]))
        center = _gcf_parms['resize_conv_size']//2
        center_embed = np.array(ps_term.shape)//2
        conv_kernel[0,0,0,center[0]-center_embed[0]:center[0]+center_embed[0],center[1]-center_embed[1] : center[1]+center_embed[1]] = ps_term
        dataset_dict['PS_CORR_IMAGE'] = xr.DataArray(ps_image, dims=['l','m'])
        ##Enabled for test
        #dataset_dict['WEIGHT_CONV_KERNEL'] = xr.DataArray(conv_kernel, dims=['conv_baseline','conv_chan','conv_pol','u','v'])
    elif (_gcf_parms['a_term'] == True) and (_gcf_parms['ps_term'] == False):
        conv_kernel =        da.real(dafft.fftshift(dafft.fft2(dafft.ifftshift(baseline_pb, axes=(3, 4)), axes=(3, 4)), axes=(3, 4)))
        conv_weight_kernel = da.real(dafft.fftshift(dafft.fft2(dafft.ifftshift(weight_baseline_pb_sqrd, axes=(3, 4)), axes=(3, 4)), axes=(3, 4)))
        
#        x = conv_weight_kernel.compute()
#        print("&*&*&*&",x.shape)
#        plt.figure()
#        #plt.imshow(x[0,0,0,240:260,240:260])
#        plt.imshow(x[0,0,0,:,:])
#        plt.show()

        list_conv_kernel = []
        list_weight_conv_kernel = []
        list_conv_support = []
        iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]))
        for c_chan, c_pol in iter_chunks_indx:
                delayed_kernels_and_support = dask.delayed(resize_and_calc_support)(conv_kernel.partitions[:,c_chan,c_pol,:,:],conv_weight_kernel.partitions[:,c_chan,c_pol,:,:],dask.delayed(_gcf_parms),dask.delayed(_grid_parms))
                list_conv_kernel.append(da.from_delayed(delayed_kernels_and_support[0],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],_gcf_parms['resize_conv_size'][0],_gcf_parms['resize_conv_size'][1]),dtype=np.double))
                list_weight_conv_kernel.append(da.from_delayed(delayed_kernels_and_support[1],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],_gcf_parms['resize_conv_size'][0],_gcf_parms['resize_conv_size'][1]),dtype=np.double))
                list_conv_support.append(da.from_delayed(delayed_kernels_and_support[2],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],2),dtype=np.int))


        conv_kernel = da.concatenate(list_conv_kernel,axis=1)
        weight_conv_kernel = da.concatenate(list_weight_conv_kernel,axis=1)
        conv_support = da.concatenate(list_conv_support,axis=1)
        
#        x = weight_conv_kernel.compute()
#        print("&*&*&*&",x.shape)
#        plt.figure()
#        #plt.imshow(x[0,0,0,240:260,240:260])
#        plt.imshow(x[0,0,0,:,:])
#        plt.show()


        dataset_dict['SUPPORT'] = xr.DataArray(conv_support, dims=['conv_baseline','conv_chan','conv_pol','xy'])
        dataset_dict['WEIGHT_CONV_KERNEL'] = xr.DataArray(weight_conv_kernel, dims=['conv_baseline','conv_chan','conv_pol','u','v'])
        dataset_dict['PS_CORR_IMAGE'] = xr.DataArray(da.from_array(np.ones(_grid_parms['image_size']),chunks=_grid_parms['image_size']), dims=['l','m'])
    else:
        assert(False), "######### ERROR: At least 'a_term' or 'ps_term' must be true."

    ###########################################################
    #Make phase gradient (one for each field)
    field_phase_dir = _gcf_parms['field_phase_dir']
    field_phase_dir = da.from_array(field_phase_dir,chunks=(np.ceil(len(field_phase_dir)/_gcf_parms['a_chan_num_chunk']),2))

    phase_gradient = da.blockwise(make_phase_gradient, ("n_field","n_x","n_y"), field_phase_dir, ("n_field","2"), gcf_parms=_gcf_parms, grid_parms=_grid_parms, dtype=complex,  new_axes={"n_x": _gcf_parms['resize_conv_size'][0], "n_y": _gcf_parms['resize_conv_size'][1]})


    ###########################################################

    #coords = {'baseline': np.arange(n_unique_ant), 'chan': pb_freq, 'pol' : pb_pol, 'u': np.arange(resize_conv_size[0]), 'v': np.arange(resize_conv_size[1]), 'xy':np.arange(2), 'field':np.arange(field_phase_dir.shape[0]),'l':np.arange(_gridding_convolution_parms['imsize'][0]),'m':np.arange(_gridding_convolution_parms['imsize'][1])}

    #coords = { 'conv_chan': pb_freq, 'conv_pol' : pb_pol, 'u': np.arange(resize_conv_size[0]), 'v': np.arange(resize_conv_size[1]), 'xy':np.arange(2), 'field':np.arange(field_phase_dir.shape[0]),'l':np.arange(_gridding_convolution_parms['imsize'][0]),'m':np.arange(_gridding_convolution_parms['imsize'][1])}

    coords = { 'u': np.arange(_gcf_parms['resize_conv_size'][0]), 'v': np.arange(_gcf_parms['resize_conv_size'][1]), 'xy':np.arange(2), 'field_id':field_id,'l':np.arange(_grid_parms['image_size'][0]),'m':np.arange(_grid_parms['image_size'][1])}

    dataset_dict['CF_BASELINE_MAP'] = xr.DataArray(cf_baseline_map, dims=('baseline')).chunk(_gcf_parms['vis_data_chunks'][1])
    dataset_dict['CF_CHAN_MAP'] = xr.DataArray(cf_chan_map, dims=('chan')).chunk(_gcf_parms['vis_data_chunks'][2])
    dataset_dict['CF_POL_MAP'] = xr.DataArray(cf_pol_map, dims=('pol')).chunk(_gcf_parms['vis_data_chunks'][3])


    dataset_dict['CONV_KERNEL'] = xr.DataArray(conv_kernel, dims=('conv_baseline','conv_chan','conv_pol','u','v'))
    dataset_dict['PHASE_GRADIENT'] = xr.DataArray(phase_gradient, dims=('field_id','u','v'))

    #print(field_id)
    gcf_dataset = xr.Dataset(dataset_dict, coords=coords)
    gcf_dataset.attrs['cell_uv'] =1/(_grid_parms['image_size_padded']*_grid_parms['cell_size']*_gcf_parms['oversampling'])
    gcf_dataset.attrs['oversampling'] = _gcf_parms['oversampling']


    #list_xarray_data_variables = [gcf_dataset['A_TERM'],gcf_dataset['WEIGHT_A_TERM'],gcf_dataset['A_SUPPORT'],gcf_dataset['WEIGHT_A_SUPPORT'],gcf_dataset['PHASE_GRADIENT']]
    #return _store(gcf_dataset,list_xarray_data_variables,_storage_parms)
    
    print('#########################  Created graph for make_gridding_convolution_function #########################')
    
    return gcf_dataset
        
#Apply Phase Gradient
#How to use WCS with Python https://astropy4cambridge.readthedocs.io/en/latest/_static/Astropy%20-%20WCS%20Transformations.html
#http://learn.astropy.org/rst-tutorials/synthetic-images.html?highlight=filtertutorials
#https://www.atnf.csiro.au/people/Mark.Calabretta/WCS/
#https://www.atnf.csiro.au/people/mcalabre/WCS/wcslib/structwcsprm.html#aadad828f07e3affd1511e533b00da19f
#https://docs.astropy.org/en/stable/api/astropy.wcs.Wcsprm.html
#The topix conversion in CASA is done at casacore/coordinates/Coordinates/Coordinate.cc, Coordinate::toPixelWCS
#    cout << "Coordinate::toPixelWCS " << std::setprecision(20) << pixel << ",*," << wcs.crpix[0] << "," << wcs.crpix[1] << ",*," << wcs.cdelt[0] << "," << wcs.cdelt[1] << ",*," << wcs.crval[0] << "," << wcs.crval[1] << ",*," << wcs.lonpole << ",*," << wcs.latpole << ",*," << wcs.ctype[0] << "," << wcs.ctype[1] << ",*," << endl;
#cout << " world phi, theta "  << std::setprecision(20) << world << ",*,"<< phi << ",*," << theta << ",*," << endl;
#Fortran numbering issue
def make_phase_gradient(field_phase_dir,gcf_parms,grid_parms):
    from astropy.wcs import WCS
    import math
    rad_to_deg =  180/np.pi
    
    #print(' make_phase_gradient ',field_phase_dir,gcf_parms,grid_parms)

    phase_center = gcf_parms['phase_center']
    w = WCS(naxis=2)
    w.wcs.crpix = grid_parms['image_size_padded']//2
    w.wcs.cdelt = grid_parms['cell_size']*rad_to_deg
    #w.wcs.cdelt = [grid_parms['cell_size'][0]*rad_to_deg, grid_parms['cell_size'][1]*rad_to_deg]
    w.wcs.crval = phase_center*rad_to_deg
    w.wcs.ctype = ['RA---SIN','DEC--SIN']
    
    #print('field_phase_dir ',field_phase_dir)
    pix_dist = np.array(w.all_world2pix(field_phase_dir[0]*rad_to_deg, 1)) - grid_parms['image_size_padded']//2
    pix = -(pix_dist)*2*np.pi/(grid_parms['image_size_padded']*gcf_parms['oversampling'])
    #print('%%%%%%%%%%%%%%%%')
    
    image_size = gcf_parms['resize_conv_size']
    center_indx = image_size//2
    x = np.arange(-center_indx[0], image_size[0]-center_indx[0])
    y = np.arange(-center_indx[1], image_size[1]-center_indx[1])
    #y_grid, x_grid = np.meshgrid(y,x)
    x_grid, y_grid = np.meshgrid(x,y,indexing='ij')

    phase_gradient = np.moveaxis(np.exp(1j*(x_grid[:,:,None]*pix[:,0] + y_grid[:,:,None]*pix[:,1])),2,0)
    return phase_gradient

def resize_and_calc_support(conv_kernel,conv_weight_kernel,gcf_parms,grid_parms):
    import itertools
    conv_shape = conv_kernel.shape[0:3]
    conv_support = np.zeros(conv_shape+(2,),dtype=int) #2 is to enable x,y support

    resized_conv_size = tuple(gcf_parms['resize_conv_size'])
    start_indx = grid_parms['image_size_padded']//2 - gcf_parms['resize_conv_size']//2
    end_indx = start_indx + gcf_parms['resize_conv_size']
    
    resized_conv_kernel = np.zeros(conv_shape + resized_conv_size,dtype=np.double)
    resized_conv_weight_kernel = np.zeros(conv_shape + resized_conv_size,dtype=np.double)
    
    for idx in itertools.product(*[range(s) for s in conv_shape]):
        conv_support[idx] = calc_conv_size(conv_weight_kernel[idx],grid_parms['image_size_padded'],gcf_parms['support_cut_level'],gcf_parms['oversampling'],gcf_parms['max_support'])
        
        
        embed_conv_size = (conv_support[idx]  + 1)*gcf_parms['oversampling']
        embed_start_indx = gcf_parms['resize_conv_size']//2 - embed_conv_size//2
        embed_end_indx = embed_start_indx + embed_conv_size
        
        resized_conv_kernel[idx] = conv_kernel[idx[0],idx[1],idx[2],start_indx[0]:end_indx[0],start_indx[1]:end_indx[1]]
        normalize_factor = np.real(np.sum(resized_conv_kernel[idx[0],idx[1],idx[2],embed_start_indx[0]:embed_end_indx[0],embed_start_indx[1]:embed_end_indx[1]])/(gcf_parms['oversampling'][0]*gcf_parms['oversampling'][1]))
        resized_conv_kernel[idx] = resized_conv_kernel[idx]/normalize_factor
        
        
        resized_conv_weight_kernel[idx] = conv_weight_kernel[idx[0],idx[1],idx[2], start_indx[0]:end_indx[0],start_indx[1]:end_indx[1]]
        normalize_factor = np.real(np.sum(resized_conv_weight_kernel[idx[0],idx[1],idx[2],embed_start_indx[0]:embed_end_indx[0],embed_start_indx[1]:embed_end_indx[1]])/(gcf_parms['oversampling'][0]*gcf_parms['oversampling'][1]))
        resized_conv_weight_kernel[idx] = resized_conv_weight_kernel[idx]/normalize_factor
        
    
    return resized_conv_kernel , resized_conv_weight_kernel , conv_support

##########################
def make_baseline_patterns(pb_freq,pb_pol,pb_ant_pairs,pb_func,gcf_parms,grid_parms):
    import copy
    
    pb_grid_parms = copy.deepcopy(grid_parms)
    pb_grid_parms['cell_size'] = grid_parms['cell_size']*gcf_parms['oversampling']
    pb_grid_parms['image_size'] =  pb_grid_parms['image_size_padded']
    pb_grid_parms['image_center'] =  pb_grid_parms['image_size']//2
    
    #print("grid_parms",grid_parms)
    #print("pb_grid_parms",pb_grid_parms)
    
    patterns = pb_func(pb_freq,pb_pol,gcf_parms,pb_grid_parms)
    baseline_pattern = np.zeros((len(pb_ant_pairs),len(pb_freq),len(pb_pol), grid_parms['image_size_padded'][0], grid_parms['image_size_padded'][1]), dtype=np.double)
    
    for ant_pair_indx, ant_pair in enumerate(pb_ant_pairs):
        for freq_indx in range(len(pb_freq)):
            baseline_pattern[ant_pair_indx,freq_indx,0,:,:] = patterns[ant_pair[0],freq_indx,0,:,:]*patterns[ant_pair[1 ],freq_indx,0,:,:]
            
    return baseline_pattern  #, conv_support_array
    
def calc_conv_size(sub_a_term,imsize,support_cut_level,oversampling,max_support):
        abs_sub_a_term = np.abs(sub_a_term)
        
        min_amplitude = np.min(abs_sub_a_term)
        max_indx = np.argmax(abs_sub_a_term)
        max_indx = np.unravel_index(max_indx, np.abs(sub_a_term).shape)
        max_amplitude = abs_sub_a_term[max_indx]
        cut_level_amplitude = support_cut_level*max_amplitude
        
        assert(min_amplitude < cut_level_amplitude), "######### ERROR: support_cut_level too small or imsize too small."
        
        #x axis support
        indx_x = imsize[0]//2
        indx_y = imsize[1]//2
        while sub_a_term[indx_x,indx_y] > cut_level_amplitude:
            indx_x = indx_x + 1
            assert(indx_x < imsize[0]), "######### ERROR: support_cut_level too small or imsize too small."
        approx_conv_size_x = (indx_x-imsize[0]//2)
        support_x = ((np.int(0.5 + approx_conv_size_x/oversampling[0]) + 1)*2 + 1)
        #support_x = int((approx_conv_size_x/oversampling[0])-1)
        #support_x = support_x if (support_x % 2) else support_x+1 #Support must be odd, to ensure symmetry
        
        #y axis support
        indx_x = imsize[0]//2
        indx_y = imsize[1]//2
        while sub_a_term[indx_x,indx_y] > cut_level_amplitude:
            indx_y = indx_y + 1
            assert(indx_y < imsize[1]), "######### ERROR: support_cut_level too small or imsize too small."
        approx_conv_size_y = (indx_y-imsize[1]//2)
        support_y = ((np.int(0.5 + approx_conv_size_y/oversampling[1]) + 1)*2 + 1)
        #approx_conv_size_y = (indx_y-imsize[1]//2)*2
        #support_y = ((approx_conv_size_y/oversampling[1])-1).astype(int)
        #support_y = support_y if (support_y % 2) else support_y+1 #Support must be odd, to ensure symmetry
        
        assert(support_x < max_support[0]), "######### ERROR: support_cut_level too small or imsize too small." + str(support_x) + ",*," + str(max_support[0])
        assert(support_y < max_support[1]), "######### ERROR: support_cut_level too small or imsize too small." + str(support_y) + ",*," + str(max_support[1])
        
        #print('approx_conv_size_x,approx_conv_size_y',approx_conv_size_x,approx_conv_size_y,support_x,support_y,max_support)
        #print('support_x, support_y',support_x, support_y)
        if support_x > support_y:
            support_y = support_x
        else:
            support_x = support_y
        return [support_x, support_y]
        
##########################
def _create_beam_map(mxds,sel_parms):
    import xarray as xr
    import dask.array as da

    #print(mxds)
    beam_ids = mxds.beam_ids.data.compute()
    n_beam = len(beam_ids)
    feed_beam_ids = mxds.FEED.beam_id.data.compute()
    
    #Assuming antenna ids remain constant over time and that feed and antenna ids line up
    #ant1_id = mxds.attrs[sel_parms['xds']].ANTENNA1[0,:]
    #ant2_id = mxds.attrs[sel_parms['xds']].ANTENNA2[0,:]
    ant1_id = mxds.attrs[sel_parms['xds']].ANTENNA1.data.compute()
    ant2_id = mxds.attrs[sel_parms['xds']].ANTENNA2.data.compute()
    
    unified_ant_indx = np.zeros((ant1_id.shape[1],2),dtype=int)
    for i in range(ant1_id.shape[1]): #for baseline
        id1 = np.unique(ant1_id[:,i])
        id1 = id1[id1 >= 0]
        id2 = np.unique(ant2_id[:,i])
        id2 = id2[id2 >= 0]
        #unified_ant_indx
        assert(len(id1) == 1 and len(id2) == 1), "Antennas not consistant over time."
        
        unified_ant_indx[i,:] = [id1[0],id2[0]]
        
    baseline_beam_id = np.full((len(unified_ant_indx),2), np.nan, dtype=np.int32)
    ant_ids = mxds.antenna_ids #same length as feed_beam_ids
    
    for indx,id in enumerate(ant_ids.data):
        baseline_beam_id[unified_ant_indx[:,0]==id,0] = feed_beam_ids[indx]
        baseline_beam_id[unified_ant_indx[:,1]==id,1] = feed_beam_ids[indx]
        

    beam_pair_id = find_unique_pairs(baseline_beam_id)
    #n_beam_pair = #max possible is int((n_beam**2 + n_beam)/2)

    beam_map = np.zeros((len(unified_ant_indx),),dtype=int)

    for k,ij in enumerate(beam_pair_id):
        #print(k,ij)
        beam_map[(baseline_beam_id[:,0] == ij[0]) & (baseline_beam_id[:,1] == ij[1])] = k
        beam_map[(baseline_beam_id[:,1] == ij[0]) & (baseline_beam_id[:,0] == ij[1])] = k
        
    vis_dataset = mxds.attrs[sel_parms['xds']]
    baseline_chunksize = vis_dataset[sel_parms['data_group_in']['data']].chunks[1]
    
    beam_map = xr.DataArray(da.from_array(beam_map,chunks=(baseline_chunksize)), dims=('baseline'))
    beam_pair_id = xr.DataArray(da.from_array(beam_pair_id,chunks=beam_pair_id.shape), dims=('beam_pair','pair'))
    return beam_map,beam_pair_id


def create_cf_baseline_map(unique_ant_indx,basline_ant,n_unique_ant):
    n_unique_ant_pairs = int((n_unique_ant**2 + n_unique_ant)/2)

    pb_ant_pairs = np.zeros((n_unique_ant_pairs,2),dtype=int)
    k = 0
    for i in range(n_unique_ant):
        for j in range(i,n_unique_ant):
           pb_ant_pairs[k,:] = [i,j]
           k = k + 1

    cf_baseline_map = np.zeros((basline_ant.shape[0],),dtype=int)
    basline_ant_unique_ant_indx = np.concatenate((unique_ant_indx[basline_ant[:,0]][:,None],unique_ant_indx[basline_ant[:,1]][:,None]),axis=1)

    for k,ij in enumerate(pb_ant_pairs):
        cf_baseline_map[(basline_ant_unique_ant_indx[:,0] == ij[0]) & (basline_ant_unique_ant_indx[:,1] == ij[1])] = k

    return cf_baseline_map,pb_ant_pairs
    
#########################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def create_cf_chan_map(freq_chan,chan_tolerance_factor):
    n_chan = len(freq_chan)
    cf_chan_map = np.zeros((n_chan,),dtype=int)
    
    orig_width = (np.max(freq_chan) - np.min(freq_chan))/len(freq_chan)
    
    tol = np.max(freq_chan)*chan_tolerance_factor
    n_pb_chan = int(np.floor( (np.max(freq_chan)-np.min(freq_chan))/tol) + 0.5) ;

    #Create PB's for each channel
    if n_pb_chan == 0:
        n_pb_chan = 1
    
    if n_pb_chan >= n_chan:
        cf_chan_map = np.arange(n_chan)
        pb_freq = freq_chan
        return cf_chan_map, pb_freq
    
    
    pb_delta_bandwdith = (np.max(freq_chan) - np.min(freq_chan))/n_pb_chan
    pb_freq = np.arange(n_pb_chan)*pb_delta_bandwdith + np.min(freq_chan) + pb_delta_bandwdith/2

    cf_chan_map = np.zeros((n_chan,),dtype=int)
    for i in range(n_chan):
        cf_chan_map[i],_ = find_nearest(pb_freq, freq_chan[i])

    

    return cf_chan_map, pb_freq



 
'''
pixel = [161.6249842951184803, 119.99951947142589859]
      161.62498429511845    119.999519470917
wcs.crpix = 120,120
wcs.cdelt = -5e-05,5e-05,
wcs.crval = 180.46846189999996568,-18.863873247222226581
wcs.lonpole = 180
wcs.latpole = -18.863873247222226581
wcs.ctype = RA---SIN,DEC--SIN
world = [-179.5337374791666889, -18.863873258333338612]
phi = -89.99933856409325017
theta = 89.997918750784648978
'''

'''
phase_center = SkyCoord(ra='05h00m00.0s', dec='-53d00m0.00s', frame='fk5')
cell_deg = np.array([-2.4,2.4])*arcsec_to_deg
img_size = np.array([140,140])
center_pixel = img_size//2
#70,*,-0.000666667,*,75,*,2
#cout << *wcs.crpix << ",*," << *wcs.cdelt << ",*," << *wcs.crval << ",*," << wcs.naxis << endl;
#####################
w2 = WCS(naxis=2)
w2.wcs.crpix = [70,70]
w2.wcs.cdelt = [-0.0006666666666666666667,0.0006666666666666666667]
w2.wcs.crval = [75,-53]
w2.wcs.lonpole = 180
w2.wcs.latpole = -53
w2.wcs.ctype = ['RA---SIN','DEC--SIN']

px, py = w2.all_world2pix(field_1[0]*rad_to_deg, field_1[1]*rad_to_deg, 1)
#px, py = w2.wcs_world2pix(74.9824, -52.9877, 0)

#pixel[85.9296, 88.3971]
print(px, py)
#84.89244220851613 87.448050465773
#84.92957845698562 87.39711173077471
#85.92957845698562 88.39711173077471
'''
#https://stackoverflow.com/questions/47531532/find-unique-pairs-of-array-with-python
def find_unique_pairs(a):
    s = np.max(a)+1
    p1 = a[:,1]*s + a[:,0]
    p2 = a[:,0]*s + a[:,1]
    p = np.maximum(p1,p2)
    sidx = p.argsort(kind='mergesort')
    ps = p[sidx]
    m = np.concatenate(([True],ps[1:] != ps[:-1]))
    sm = sidx[m]
    #print(m)
    #print(a[sm,:])
    
    return a[sm,:]
