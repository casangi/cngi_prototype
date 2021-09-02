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

from numba import jit
import numba
import numpy as np
import numpy.fft as npfft
import copy

#Combine all parameters into one dict for passing
#Where should imports go
from ._imaging_utils._standard_grid import _standard_grid_psf_numpy_wrap, _standard_imaging_weight_degrid_numpy_wrap, _standard_grid_numpy_wrap
from ._imaging_utils._gridding_convolutional_kernels import _create_prolate_spheroidal_kernel, _create_prolate_spheroidal_kernel_1D
from ._imaging_utils._remove_padding import _remove_padding
from cngi.image.fit_gaussian import casa_fit

def synthesis_imaging_cube(vis_mxds, img_xds, grid_parms, imaging_weights_parms, pb_parms, vis_sel_parms, img_sel_parms):
    print('v3')

    print('######################### Start Synthesis Imaging Cube #########################')
    import numpy as np
    from numba import jit
    import time
    import math
    import dask.array.fft as dafft
    import xarray as xr
    import dask.array as da
    import matplotlib.pylab as plt
    import dask
    import copy, os
    from numcodecs import Blosc
    from itertools import cycle
    import itertools
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms
    from ._imaging_utils._check_imaging_parms import _check_imaging_weights_parms, _check_grid_parms, _check_pb_parms
    from ._imaging_utils._make_pb_symmetric import _airy_disk, _casa_airy_disk
    from cngi.image import make_empty_sky_image

    _mxds = vis_mxds.copy(deep=True)
    _vis_sel_parms = copy.deepcopy(vis_sel_parms)
    _img_sel_parms = copy.deepcopy(img_sel_parms)
    _grid_parms = copy.deepcopy(grid_parms)
    _imaging_weights_parms = copy.deepcopy(imaging_weights_parms)
    _img_xds = copy.deepcopy(img_xds)
    _pb_parms = copy.deepcopy(pb_parms)

    assert('xds' in _vis_sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_xds = _mxds.attrs[_vis_sel_parms['xds']]
    assert _vis_xds.dims['pol'] <= 2, "Full polarization is not supported."
    assert(_check_imaging_weights_parms(_imaging_weights_parms)), "######### ERROR: imaging_weights_parms checking failed"
    assert(_check_grid_parms(_grid_parms)), "######### ERROR: grid_parms checking failed"
    assert(_check_pb_parms(_img_xds,_pb_parms)), "######### ERROR: user_imaging_weights_parms checking failed"
    
    #Check vis data_group
    _check_sel_parms(_vis_xds,_vis_sel_parms)
    
    #Check img data_group
    _check_sel_parms(_img_xds,_img_sel_parms,new_or_modified_data_variables={'image_sum_weight':'IMAGE_SUM_WEIGHT','image':'IMAGE','psf_sum_weight':'PSF_SUM_WEIGHT','psf':'PSF','pb':'PB','restore_parms':'RESTORE_PARMS'},append_to_in_id=True)

    
    parms = {'grid_parms':_grid_parms,'imaging_weights_parms':_imaging_weights_parms,'pb_parms':_pb_parms,'vis_sel_parms':_vis_sel_parms,'img_sel_parms':_img_sel_parms}
    
    chunk_sizes = list(_vis_xds[_vis_sel_parms["data_group_in"]["data"]].chunks)
    chunk_sizes[0] = (np.sum(chunk_sizes[2]),)
    chunk_sizes[1] = (np.sum(chunk_sizes[1]),)
    chunk_sizes[3] = (np.sum(chunk_sizes[3]),)
    n_pol = _vis_xds.dims['pol']
 
    #assert n_chunks_in_each_dim[3] == 1, "Chunking is not allowed on pol dim."
    n_chunks_in_each_dim = list(_vis_xds[_vis_sel_parms["data_group_in"]["data"]].data.numblocks)
    n_chunks_in_each_dim[0] = 1 #time
    n_chunks_in_each_dim[1] = 1 #baseline
    n_chunks_in_each_dim[3] = 1 #pol

    #Iter over time,baseline,chan
    iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]),
                                         np.arange(n_chunks_in_each_dim[2]), np.arange(n_chunks_in_each_dim[3]))
                                         
    image_list = _ndim_list(n_chunks_in_each_dim)
    image_sum_weight_list = _ndim_list(n_chunks_in_each_dim[2:])
    psf_list = _ndim_list(n_chunks_in_each_dim)
    psf_sum_weight_list = _ndim_list(n_chunks_in_each_dim[2:])
    
    pb_list = _ndim_list(tuple(n_chunks_in_each_dim) + (1,))
    ellipse_parms_list = _ndim_list(tuple(n_chunks_in_each_dim[2:]) + (1,))
    n_dish_type = len(_pb_parms['list_dish_diameters'])
    n_elps = 3
    
    freq_chan = da.from_array(_vis_xds.coords['chan'].values, chunks=(_vis_xds[_vis_sel_parms["data_group_in"]["data"]].chunks[2]))
  
    # Build graph
    for c_time, c_baseline, c_chan, c_pol in iter_chunks_indx:
        #c_time, c_baseline, c_chan, c_pol
        #print(_vis_xds[_vis_sel_parms["data_group_in"]["data"]].data.partitions[:, :, c_chan, :].shape)
        synthesis_chunk = dask.delayed(_synthesis_imaging_cube_std_chunk)(
            _vis_xds[_vis_sel_parms["data_group_in"]["data"]].data.partitions[:, :, c_chan, :],
            _vis_xds[_vis_sel_parms["data_group_in"]["uvw"]].data.partitions[:, :, :],
            _vis_xds[_vis_sel_parms["data_group_in"]["weight"]].data.partitions[:, :, c_chan, :],
            _vis_xds[_vis_sel_parms["data_group_in"]["flag"]].data.partitions[:, :, c_chan, :],
            freq_chan.partitions[c_chan],
            dask.delayed(parms))
            
        image_list[c_time][c_baseline][c_chan][c_pol] = da.from_delayed(synthesis_chunk[0],(_grid_parms['image_size'][0], _grid_parms['image_size'][1],chunk_sizes[2][c_chan], chunk_sizes[3][c_pol]),dtype=np.double)
        image_sum_weight_list[c_chan][c_pol] = da.from_delayed(synthesis_chunk[1],(chunk_sizes[2][c_chan], chunk_sizes[3][c_pol]),dtype=np.double)
        
        psf_list[c_time][c_baseline][c_chan][c_pol] = da.from_delayed(synthesis_chunk[2],(_grid_parms['image_size'][0], _grid_parms['image_size'][1],chunk_sizes[2][c_chan], chunk_sizes[3][c_pol]),dtype=np.double)
        psf_sum_weight_list[c_chan][c_pol] = da.from_delayed(synthesis_chunk[3],(chunk_sizes[2][c_chan], chunk_sizes[3][c_pol]),dtype=np.double)
        
        pb_list[c_time][c_baseline][c_chan][c_pol][0] = da.from_delayed(synthesis_chunk[4],(_grid_parms['image_size'][0], _grid_parms['image_size'][1],chunk_sizes[2][c_chan], chunk_sizes[3][c_pol],n_dish_type),dtype=np.double)
        
        ellipse_parms_list[c_chan][c_pol][0] = da.from_delayed(synthesis_chunk[5],(chunk_sizes[2][c_chan], chunk_sizes[3][c_pol],n_elps),dtype=np.double)
        
        
        #return image, image_sum_weight, psf, psf_sum_weight, pb
        
    if _grid_parms['chan_mode'] == 'continuum':
        freq_coords = [da.mean(_vis_xds.coords['chan'].values)]
        chan_width = da.from_array([da.mean(_vis_xds['chan_width'].data)],chunks=(1,))
        imag_chan_chunk_size = 1
    elif _grid_parms['chan_mode'] == 'cube':
        freq_coords = _vis_xds.coords['chan'].values
        chan_width = _vis_xds['chan_width'].data
        imag_chan_chunk_size = _vis_xds.DATA.chunks[2][0]
    
    phase_center = _grid_parms['phase_center']
    image_size = _grid_parms['image_size']
    cell_size = _grid_parms['cell_size']
    phase_center = _grid_parms['phase_center']

    pol_coords = _vis_xds.pol.data
    time_coords = [_vis_xds.time.mean().data]
    
    _img_xds = make_empty_sky_image(_img_xds,phase_center,image_size,cell_size,freq_coords,chan_width,pol_coords,time_coords)
    
    #print(da.block(image_list))
    #print(da.block(psf_list))
    #print(pb_list)
    #print(da.block(pb_list))
    
    _img_xds[_img_sel_parms['data_group_out']['image']] = xr.DataArray(da.block(image_list)[:,:,None,:,:], dims=['l', 'm', 'time', 'chan', 'pol'])
    _img_xds[_img_sel_parms['data_group_out']['image_sum_weight']] = xr.DataArray(da.block(image_sum_weight_list)[None,:,:], dims=['time','chan','pol'])
    
    print(da.block(ellipse_parms_list))
    
    _img_xds[_img_sel_parms['data_group_out']['restore_parms']] = xr.DataArray(da.block(ellipse_parms_list)[None,:,:,:], dims=['time','chan','pol','elps_index'])
    
    _img_xds[_img_sel_parms['data_group_out']['psf']] = xr.DataArray(da.block(psf_list)[:,:,None,:,:], dims=['l', 'm', 'time', 'chan', 'pol'])
    _img_xds[_img_sel_parms['data_group_out']['psf_sum_weight']] = xr.DataArray(da.block(psf_sum_weight_list)[None,:,:], dims=['time','chan','pol'])
    
    _img_xds[_img_sel_parms['data_group_out']['pb']] = xr.DataArray(da.block(pb_list)[:,:,None,:,:,:], dims=['l', 'm', 'time', 'chan', 'pol','dish_type'])
    _img_xds = _img_xds.assign_coords({'dish_type': np.arange(len(_pb_parms['list_dish_diameters']))})
    _img_xds.attrs['data_groups'][0] = {**_img_xds.attrs['data_groups'][0],**{_img_sel_parms['data_group_out']['id']:_img_sel_parms['data_group_out']}}
        
    return _img_xds

import time

def _synthesis_imaging_cube_std_chunk(vis_data, uvw,data_weight,flag,freq_chan,parms):
    grid_parms = parms['grid_parms']
    imaging_weights_parms = parms['imaging_weights_parms']
    pb_parms = parms['pb_parms']

    #print('###########1Shapes',vis_data.shape,uvw.shape,data_weight.shape,flag.shape,flag.dtype)
    #Flag data
    #s1 = time.time()
    #vis_data[flag] = np.nan
    vis_data[flag==1] = np.nan
    #print("Flag ", time.time()-s1)
    #print('###########2Shapes',vis_data.shape,uvw.shape,data_weight.shape,flag.shape)
    #Imaging Weights
    
    #s2 = time.time()
    imaging_weights = _make_imaging_weight_chunk(uvw,data_weight,freq_chan,grid_parms,imaging_weights_parms)
    #print("Imaging Weights ", time.time()-s2)
    #Make PB
    #s3 = time.time()
    vis_data_shape = vis_data.shape
    pb = _make_pb(vis_data_shape,freq_chan,pb_parms,grid_parms)
    #print("Make PB ", time.time()-s3)
    
    # Creating gridding kernel
    #s4 = time.time()
    grid_parms['oversampling'] = 100
    grid_parms['support'] = 7
    
    #print(grid_parms)
    cgk, correcting_cgk_image = _create_prolate_spheroidal_kernel(grid_parms['oversampling'], grid_parms['support'], grid_parms['image_size_padded'])
    cgk_1D = _create_prolate_spheroidal_kernel_1D(grid_parms['oversampling'], grid_parms['support'])
    correcting_cgk_image = _remove_padding(correcting_cgk_image,grid_parms['image_size'])
    #print("Create GCF", time.time()-s4)
    
    #s5 = time.time()
    psf, psf_sum_weight = _make_psf(uvw, data_weight, freq_chan, cgk_1D, grid_parms)
    psf = correct_image(psf, psf_sum_weight[None, None, :, :], correcting_cgk_image[:, :, None, None])
    #print("Make PSF ", time.time()-s5)
    
    #s6 = time.time()
    image, image_sum_weight = _make_image(vis_data, uvw, data_weight, freq_chan, cgk_1D, grid_parms)
    image = correct_image(image, image_sum_weight[None, None, :, :], correcting_cgk_image[:, :, None, None])
    #print("Make IMAGE ", time.time()-s6)
    
    #s7 = time.time()
    cell_size = grid_parms['cell_size']
    ellipse_parms = casa_fit(psf[:,:,None,:,:],npix_window=np.array([9,9]),sampling=np.array([9,9]),cutoff=0.35,delta=np.array(cell_size))[0,:,:,:]
    #print("Fit PSF ", time.time()-s7)
        
    return image, image_sum_weight, psf, psf_sum_weight, pb, ellipse_parms
    
#############Normalize#############
def correct_image(uncorrected_dirty_image, sum_weights, correcting_cgk):
        sum_weights_copy = copy.deepcopy(sum_weights) ##Don't mutate inputs, therefore do deep copy (https://docs.dask.org/en/latest/delayed-best-practices.html).
        sum_weights_copy[sum_weights_copy == 0] = 1
        # corrected_image = (uncorrected_dirty_image/sum_weights[:,:,None,None])/correcting_cgk[None,None,:,:]
        corrected_image = (uncorrected_dirty_image / sum_weights_copy) / correcting_cgk
        return corrected_image
        
def _make_image(vis_data, uvw, data_weight, freq_chan, cgk_1D, grid_parms):
    grid_parms['complex_grid'] = True
    grid_parms['do_psf'] = False
    grid_parms['do_imaging_weight'] = False
    
    grid, sum_weight = _standard_grid_numpy_wrap(vis_data, uvw, data_weight, freq_chan, cgk_1D, grid_parms)
    
    grid = np.moveaxis(grid,(0,1),(2,3)) #Temp need to change def of image coord pos.
    uncorrected_image = npfft.fftshift(npfft.ifft2(npfft.ifftshift(grid, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    
    #Remove Padding
    uncorrected_image = _remove_padding(uncorrected_image,grid_parms['image_size']).real * (grid_parms['image_size_padded'][0] * grid_parms['image_size_padded'][1])
    
    return uncorrected_image, sum_weight
    
def _make_psf(uvw, weight, freq_chan, cgk_1D, grid_parms):
    
    grid_parms['complex_grid'] = False
    grid_parms['do_psf'] = True
    grid_parms['do_imaging_weight'] = False
    
    grid, sum_weight = _standard_grid_psf_numpy_wrap(uvw, weight, freq_chan, cgk_1D, grid_parms)
    
    grid = np.moveaxis(grid,(0,1),(2,3)) #Temp need to change def of image coord pos.
    uncorrected_psf = npfft.fftshift(npfft.ifft2(npfft.ifftshift(grid, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    
    #Remove Padding
    uncorrected_psf = _remove_padding(uncorrected_psf,grid_parms['image_size']).real * (grid_parms['image_size_padded'][0] * grid_parms['image_size_padded'][1])
    
    return uncorrected_psf, sum_weight
    
def _ndim_list(shape):
    return [_ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]
    
def _make_pb(vis_data_shape,freq_chan,pb_parms,grid_parms):
    if pb_parms['function'] == 'airy':
        from ._imaging_utils._make_pb_symmetric import _airy_disk
        pb_func = _airy_disk
    elif pb_parms['function'] == 'casa_airy':
        from ._imaging_utils._make_pb_symmetric import _casa_airy_disk
        pb_func = _casa_airy_disk
    else:
        print('Only the airy function has been implemented')
    
    pb_parms['ipower'] = 2
    pb_parms['center_indx'] = []
    
    pol = np.zeros(vis_data_shape[3])
    
    pb = pb_func(freq_chan, pol, pb_parms, grid_parms)
    
    return pb


def _make_imaging_weight_chunk(uvw,data_weight,freq_chan,grid_parms,imaging_weights_parms):
    if imaging_weights_parms['weighting'] == 'natural':
        imaging_weights = data_weight
    else:
        _grid_parms = copy.deepcopy(grid_parms)
        _grid_parms['image_size_padded'] =  grid_parms['image_size'] #do not need to pad since no fft
        _grid_parms['oversampling'] = 0
        _grid_parms['support'] = 1
        _grid_parms['do_psf'] = True
        _grid_parms['complex_grid'] = False
        _grid_parms['do_imaging_weight'] = True
        
        cgk_1D = np.ones((1))

        #Grid Weights
        weight_density_grid, sum_weight = _standard_grid_psf_numpy_wrap(uvw, data_weight, freq_chan, cgk_1D, _grid_parms)
        
        #Calculate Briggs
        briggs_factors = _calculate_briggs_parms(weight_density_grid, sum_weight, imaging_weights_parms) # 2 x chan x pol
        
        #Degrid weight density grid
        weight_density_grid = np.moveaxis(weight_density_grid,(0,1),(2,3)) #Temp need to change def of image coord pos.
        imaging_weights = _standard_imaging_weight_degrid_numpy_wrap(weight_density_grid, uvw, data_weight, briggs_factors, freq_chan, _grid_parms)
    
    return imaging_weights
    
def _calculate_briggs_parms(grid_of_imaging_weights, sum_weight, imaging_weights_parms):
    if imaging_weights_parms['weighting'] == 'briggs':
        robust = imaging_weights_parms['robust']
        briggs_factors = np.ones((2,)+sum_weight.shape)
        squared_sum_weight = (np.sum(grid_of_imaging_weights**2,axis=(2,3)))
        briggs_factors[0,:,:] =  (np.square(5.0*10.0**(-robust))/(squared_sum_weight/sum_weight))[None,None,:,:]
    elif imaging_weights_parms['weighting'] == 'briggs_abs':
        robust = imaging_weights_parms['robust']
        briggs_factors = np.ones((2,)+sum_weight.shape)
        briggs_factors[0,:,:] = briggs_factor[0,0,0,:,:]*np.square(robust)
        briggs_factors[1,:,:] = briggs_factor[1,0,0,:,:]*2.0*np.square(imaging_weights_parms['briggs_abs_noise'])
    else:
        briggs_factors = np.zeros((2,1,1)+sum_weight.shape)
        briggs_factors[0,:,:] = np.ones((1,1,1)+sum_weight.shape)
        
    return briggs_factors
