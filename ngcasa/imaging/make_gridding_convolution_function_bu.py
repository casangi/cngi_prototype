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
import numpy as np

# Should ps_term be seperate, so that higher oversampling can be used?
# Support should always be odd

def make_gridding_convolution_function(img_cf_dataset, gridding_convolution_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    Calculate gridding convolution functions (GCF) as specified for standard, widefield and mosaic imaging.
    Construct a GCF cache (persistent or on-the-fly)

    Options : Choose a list of effects to include
    
    - PSterm : Prolate-Spheroidal gridding kernel (anti-aliasing function)
    - Aterm : Use PB model and Aperture Illumination Function per antenna to construct a GCF per baseline
        - Include support for Heterogeneous Arrays where Aterm is different per antenna
        - Include support for time-varying PB and AIF models. Rotation, etc.
    - Wterm : FT of Fresnel kernel per baseline
            
    Returns
    -------
    cf_dataset : xarray.core.dataset.Dataset
            
    """
    from ngcasa._ngcasa_utils._store import _store
    from ngcasa._ngcasa_utils._check_parms import _check_storage_parms
    from ._imaging_utils._check_imaging_parms import _check_pb_parms
    from ._imaging_utils._gridding_convolutional_kernels import _create_prolate_spheroidal_kernel_2D, _create_prolate_spheroidal_image_2D
    import numpy as np
    import dask.array as da
    import copy, os
    import xarray as xr
    import itertools
    import dask
    
    
    import matplotlib.pylab as plt
    
    _gridding_convolution_parms =  copy.deepcopy(gridding_convolution_parms)
    _storage_parms = copy.deepcopy(storage_parms)
    
    assert(_check_storage_parms(_storage_parms,'dataset.gcf.zarr','make_gcf')), "######### ERROR: user_storage_parms checking failed"
       

    if _gridding_convolution_parms['function'] == 'airy':
        from ._imaging_utils._make_pb_1d import _airy_disk
        pb_func = _airy_disk
    else:
        print('Only the airy function has been implemented')
    
    if _gridding_convolution_parms['ps_term'] == True:
        ps_kernel = _create_prolate_spheroidal_kernel_2D(_gridding_convolution_parms['ps_oversampling'],_gridding_convolution_parms['ps_support'])
        ps_image = _create_prolate_spheroidal_image_2D(_gridding_convolution_parms['imsize'])
        
    
    if _gridding_convolution_parms['a_term'] == True:
        n_unique_ant = len(_gridding_convolution_parms['list_dish_diameters'])
        a_cf_baseline_map,pb_ant_pairs = create_cf_baseline_map(_gridding_convolution_parms['unique_ant_indx'],_gridding_convolution_parms['basline_ant'],n_unique_ant)
        
        a_cf_chan_map, pb_freq = create_cf_chan_map(_gridding_convolution_parms['freq_chan'],_gridding_convolution_parms['chan_tolerance_factor'])
        pb_freq = da.from_array(pb_freq,chunks=np.ceil(len(pb_freq)/_gridding_convolution_parms['a_chan_num_chunk'] ))
        
        a_cf_pol_map = np.zeros((len(_gridding_convolution_parms['pol']),),dtype=int) #create_cf_pol_map(), currently treating all pols the same
        pb_pol = da.from_array(np.array([0]),1)
        
        #print(pb_freq,',*,',pb_ant_pairs,'*',pb_pol)
        
        #_gridding_convolution_parms['conv_size'] = (_gridding_convolution_parms['a_max_support'] +1)*_gridding_convolution_parms['a_oversampling']
        #conv_size = _gridding_convolution_parms['conv_size']
        
        _gridding_convolution_parms['conv_size'] = _gridding_convolution_parms['imsize']
        conv_size = _gridding_convolution_parms['imsize']
        
        _gridding_convolution_parms['resize_conv_size'] = (gridding_convolution_parms['resize_support'] + 1)*gridding_convolution_parms['a_oversampling']
        
        resize_conv_size = _gridding_convolution_parms['resize_conv_size']
        
        '''
        _gridding_convolution_parms['ipower'] = 1
        a_term = da.blockwise(make_a_term, ("n_ant_pairs","n_chan","n_pol","n_x","n_y"), pb_freq, ("n_chan",), pb_pol, ("n_pol",) , pb_ant_pairs=pb_ant_pairs , gridding_convolution_parms=_gridding_convolution_parms, dtype=complex,  new_axes={"n_ant_pairs":len(pb_ant_pairs),"n_x": conv_size[0], "n_y": conv_size[1]})
        
        _gridding_convolution_parms['ipower'] = 2
        weight_a_term = da.blockwise(make_a_term, ("n_ant_pairs","n_chan","n_pol","n_x","n_y"), pb_freq, ("n_chan",), pb_pol, ("n_pol",) , pb_ant_pairs=pb_ant_pairs , gridding_convolution_parms=_gridding_convolution_parms, dtype=complex,  new_axes={"n_ant_pairs":len(pb_ant_pairs),"n_x": conv_size[0], "n_y": conv_size[1]})
        
        '''
        
        n_chunks_in_each_dim = [pb_freq.numblocks[0],pb_pol.numblocks[0]]
        iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]))
        chan_chunk_sizes = pb_freq.chunks
        pol_chunk_sizes = pb_pol.chunks
        
        #print(pb_freq, pb_pol,pol_chunk_sizes)
        list_a_term = []
        list_a_support = []
        list_weight_a_term = []
        list_weight_a_support = []
        for c_chan, c_pol in iter_chunks_indx:
                print('chan, pol ',c_chan,c_pol)
                _gridding_convolution_parms['ipower'] = 1
                delayed_a_term_and_support = dask.delayed(make_a_term)(pb_freq.partitions[c_chan],pb_pol.partitions[c_pol],dask.delayed(pb_ant_pairs),dask.delayed(_gridding_convolution_parms))
                
                list_a_term.append(da.from_delayed(delayed_a_term_and_support[0],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],resize_conv_size[0],resize_conv_size[1]),dtype=np.complex))
                list_a_support.append(da.from_delayed(delayed_a_term_and_support[1],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],2),dtype=np.int))
                
                #list_a_term.append(da.from_delayed(delayed_a_term_and_support[0],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],np.nan,np.nan),dtype=np.complex))
                #list_a_support.append(da.from_delayed(delayed_a_term_and_support[1],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol]),dtype=np.int))
                
                _gridding_convolution_parms['ipower'] = 2
                delayed_weight_a_term_and_support = dask.delayed(make_a_term)(pb_freq.partitions[c_chan],pb_pol.partitions[c_pol],dask.delayed(pb_ant_pairs),dask.delayed(_gridding_convolution_parms))
                
                list_weight_a_term.append(da.from_delayed(delayed_weight_a_term_and_support[0],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],resize_conv_size[0],resize_conv_size[1]),dtype=np.complex))
                list_weight_a_support.append(da.from_delayed(delayed_weight_a_term_and_support[1],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],2),dtype=np.int))
                
                #list_weight_a_term.append(da.from_delayed(delayed_weight_a_term_and_support[0],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol],np.nan,np.nan),dtype=np.complex))
                #list_weight_a_support.append(da.from_delayed(delayed_weight_a_term_and_support[1],(len(pb_ant_pairs),chan_chunk_sizes[0][c_chan], pol_chunk_sizes[0][c_pol]),dtype=np.int))
                
        #print(list_a_term[2].compute().shape)
        #Concatenate on frequency dimension (pol is ignored for now)
        print(list_a_term)
        
        a_term = da.concatenate(list_a_term,axis=1)
        a_support = da.concatenate(list_a_support,axis=1)
        weight_a_term = da.concatenate(list_weight_a_term,axis=1)
        weight_a_support = da.concatenate(list_weight_a_support,axis=1)
        
        ###########################################################
        #Make phase gradient
        field_phase_dir = _gridding_convolution_parms['field_phase_dir']
        field_phase_dir = da.from_array(field_phase_dir,chunks=(np.ceil(len(field_phase_dir)/_gridding_convolution_parms['phase_gradient_num_chunk']),2))
        
        phase_gradient = da.blockwise(make_phase_gradient, ("n_field","n_x","n_y"), field_phase_dir, ("n_field","2"), gridding_convolution_parms=_gridding_convolution_parms, dtype=complex,  new_axes={"n_x": resize_conv_size[0], "n_y": resize_conv_size[1]})
        
        print( phase_gradient.compute().shape)
        
        #return a_term,a_support
        #print(a_term, pb_pol)
        
        ######Create Dataset
        print('ps_kernal.shape',ps_kernel.shape)
        print('a_term.shape',a_term)
        
        coords = {'baseline': np.arange(n_unique_ant), 'chan': pb_freq, 'pol' : pb_pol, 'u': np.arange(resize_conv_size[0]), 'v': np.arange(resize_conv_size[1]), 'xy':np.arange(2), 'field':np.arange(field_phase_dir.shape[0])}
        dataset_dict = {}
        dataset_dict['PS_TERM'] = xr.DataArray(ps_kernel, dims=['u_ps','v_ps'])
        dataset_dict['A_TERM'] = xr.DataArray(a_term, dims=['baseline','chan','pol','u','v'])
        dataset_dict['WEIGHT_A_TERM'] = xr.DataArray(weight_a_term, dims=['baseline','chan','pol','u','v'])
        dataset_dict['A_SUPPORT'] = xr.DataArray(a_support, dims=['baseline','chan','pol','xy'])
        dataset_dict['WEIGHT_A_SUPPORT'] = xr.DataArray(weight_a_support, dims=['baseline','chan','pol','xy'])
        dataset_dict['PHASE_GRADIENT'] = xr.DataArray(phase_gradient, dims=['field','u','v'])
        gcf_dataset = xr.Dataset(dataset_dict, coords=coords)
        gcf_dataset.attrs['cell_uv'] =1/(conv_size*_gridding_convolution_parms['cell']*_gridding_convolution_parms['a_oversampling'])
        
        list_xarray_data_variables = [gcf_dataset['A_TERM'],gcf_dataset['WEIGHT_A_TERM'],gcf_dataset['A_SUPPORT'],gcf_dataset['WEIGHT_A_SUPPORT'],gcf_dataset['PHASE_GRADIENT']]
        return _store(gcf_dataset,list_xarray_data_variables,_storage_parms)
        
        #print(a_term)
        
        #a_term = da.blockwise(resize_conv_function, ("n_ant_pairs","n_chan","n_pol","n_x","n_y"), a_term, ("n_ant_pairs","n_chan","n_pol","n_x","n_y"), , dtype=complex,  new_axes={"n_ant_pairs":len(pb_ant_pairs),"n_x": conv_size[0], "n_y": conv_size[1]})
        #a_term = resize_conv_function(a_term,a_support,_gridding_convolution_parms['a_oversampling'])
        #weight_a_term = resize_conv_function(weight_a_term,weight_a_support,_gridding_convolution_parms['a_oversampling'])
                
        #print(delayed_a_term_and_support.compute())
        
        ###Create Dataset
        
        '''
        dataset_dict = {}
        dataset_dict['test1'] = xr.DataArray(list_a_term[0], dims=['baseline','chan','pol','d0_0 ','d1_0'])
        dataset_dict['test2'] = xr.DataArray(list_a_term[1], dims=['baseline','chan','pol','d0_1','d1_1'])
        image_dataset = xr.Dataset(dataset_dict)
        print(image_dataset)
        '''
        
        '''
        chunks = vis_dataset.DATA.chunks
        n_imag_pol = chunks[3][0]
        image_dict = {}
        coords = {'d0': np.arange(_grid_parms['imsize'][0]), 'd1': np.arange(_grid_parms['imsize'][1]),
                  'chan': freq_coords, 'pol': np.arange(n_imag_pol), 'chan_width' : ('chan',chan_width)}
                  
        image_dict[_grid_parms['sum_weight_name']] = xr.DataArray(grids_and_sum_weights[1], dims=['chan','pol'])
        image_dict[_grid_parms['image_name']] = xr.DataArray(corrected_dirty_image, dims=['d0', 'd1', 'chan', 'pol'])
        image_dataset = xr.Dataset(image_dict, coords=coords)
        
        list_xarray_data_variables = [image_dataset[_grid_parms['image_name']],image_dataset[_grid_parms['sum_weight_name']]]
        return _store(image_dataset,list_xarray_data_variables,_storage_parms)
        '''

    #create_cf_channel_map()
    #create_cf_pol_map()
    #create_cf_baseline_map()
    
    

    
##########################
'''
def resize_conv_function(array,support,oversampling):
    max_support = np.max(support)
    resized_conv_size = (max_support +1)*oversampling
    
    current_conv_size = np.array(array.shape[3:5])
    
    start_indx = current_conv_size//2 - resized_conv_size//2
    end_indx = start_indx + resized_conv_size
    
    return array[:,:,:,start_indx[0]:end_indx[0],start_indx[1]:end_indx[1]]
'''


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
def make_phase_gradient(field_phase_dir,gridding_convolution_parms):
    from astropy.wcs import WCS
    rad_to_deg =  180/np.pi

    phase_center = gridding_convolution_parms['phase_center']
    w = WCS(naxis=2)
    w.wcs.crpix = gridding_convolution_parms['imsize']//2
    w.wcs.cdelt = gridding_convolution_parms['cell']*rad_to_deg
    w.wcs.crval = phase_center*rad_to_deg
    w.wcs.ctype = ['RA---SIN','DEC--SIN']
    
    #print('field_phase_dir ',field_phase_dir)
    pix_dist = np.array(w.all_world2pix(field_phase_dir[0]*rad_to_deg, 1)) - gridding_convolution_parms['imsize']//2 #Why is fortran numbering used?
    pix = -(pix_dist)*2*np.pi/(gridding_convolution_parms['imsize']*gridding_convolution_parms['a_oversampling'])
    
    print(pix)
    
    image_size = gridding_convolution_parms['resize_conv_size']
    center_indx = image_size//2
    x = np.arange(-center_indx[0], image_size[0]-center_indx[0])
    y = np.arange(-center_indx[1], image_size[1]-center_indx[1])
    y_grid, x_grid = np.meshgrid(y,x)
    
    #print((x_grid[:,:,None]*pix[:,0]).shape)
    
    phase_gradient = np.moveaxis(np.exp(1j*(x_grid[:,:,None]*pix[:,0] + y_grid[:,:,None]*pix[:,1])),2,0)
    return phase_gradient
    

##########################
def make_a_term(pb_freq,pb_pol,pb_ant_pairs,gridding_convolution_parms):
    from ._imaging_utils._make_pb_1d import _airy_disk_rorder
    
    
    pb_parms ={}
    pb_parms['cell'] = gridding_convolution_parms['cell']*gridding_convolution_parms['a_oversampling']
    pb_parms['imsize'] = gridding_convolution_parms['conv_size']
    pb_parms['center_indx'] = pb_parms['imsize']//2
    pb_parms['list_dish_diameters'] = gridding_convolution_parms['list_dish_diameters']
    pb_parms['list_blockage_diameters'] = gridding_convolution_parms['list_blockage_diameters']
    pb_parms['ipower'] = gridding_convolution_parms['ipower']
    pb_parms['pb_limit'] = 0.0
    
    #print(pb_parms)
    print('pb_pol',pb_pol)
    patterns = _airy_disk_rorder(pb_freq,pb_pol,pb_parms)
    
    a_term = np.zeros((len(pb_ant_pairs),len(pb_freq),len(pb_pol), pb_parms['imsize'][0], pb_parms['imsize'][1]), dtype=complex)
    conv_support_array = np.zeros((len(pb_ant_pairs),len(pb_freq),len(pb_pol),2),dtype=int) #2 is to enable x,y support
    
    max_support = ((gridding_convolution_parms['conv_size']/gridding_convolution_parms['a_oversampling'])-1).astype(int)
    max_support[0] = max_support[0] if (max_support[0] % 2) else max_support[0]-1 #Support must be odd, to ensure symmetry
    max_support[1] = max_support[1] if (max_support[1] % 2) else max_support[1]-1
    max_half_support = ((max_support-1)/2).astype(int)
    current_conv_size = pb_parms['imsize']
    #print(max_support,max_half_support)
    
    for ant_pair_indx, ant_pair in enumerate(pb_ant_pairs):
        for freq_indx in range(len(pb_freq)):
            sub_a_term = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(patterns[ant_pair[0],freq_indx,0,:,:]*patterns[ant_pair[1 ],freq_indx,0,:,:])))
            conv_support_array[ant_pair_indx,freq_indx,0,:] = calc_conv_size(sub_a_term,pb_parms['imsize'],gridding_convolution_parms['support_cut_level'],gridding_convolution_parms['a_oversampling'],max_support)
            
            resized_conv_size = (conv_support_array[ant_pair_indx,freq_indx,0,:]  + 1)*gridding_convolution_parms['a_oversampling']
            start_indx = current_conv_size//2 - resized_conv_size//2
            end_indx = start_indx + resized_conv_size
            normalize_factor = np.real(np.sum(sub_a_term[start_indx[0]:end_indx[0],start_indx[1]:end_indx[1]])/(gridding_convolution_parms['a_oversampling'][0]*gridding_convolution_parms['a_oversampling'][1]))
            
            a_term[ant_pair_indx,freq_indx,0,:,:] = sub_a_term/normalize_factor
            
    resized_conv_size = gridding_convolution_parms['resize_conv_size']
    #max_support = np.array([np.max(conv_support_array[:,:,:,0]),np.max(conv_support_array[:,:,:,1])])
    #resized_conv_size = (max_support +1)*gridding_convolution_parms['a_oversampling']
    start_indx = current_conv_size//2 - resized_conv_size//2
    end_indx = start_indx + resized_conv_size
    
    return a_term[:,:,:,start_indx[0]:end_indx[0],start_indx[1]:end_indx[1]], conv_support_array
    
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
        return [support_x, support_y]
        
##########################
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
