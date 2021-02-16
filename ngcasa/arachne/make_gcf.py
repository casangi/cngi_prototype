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
import time

#Important
#a_pair = a1_plane[map_mueler_to_pol[mm,0],:,:]*(a2_plane[map_mueler_to_pol[mm,1],:,:].conj().T)  ? .conj().T


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
        cf_a_planes = _create_a_term_graph(gcf_dataset, list_zpc_dataset, _gcf_parms, _grid_parms, _sel_parms)
        
        
        
    else:
        a = 0
    
    ###################################################### W_TERM #####################################################################
    if _gcf_parms['w_term']:
        print('#########  Creating w_term ')
        cf_w_planes = _create_w_term_graph(gcf_dataset,_gcf_parms, _grid_parms, _sel_parms)
        #print('cf_w_planes',cf_w_planes)
        #print('cf_w_planes.compute().shape',cf_w_planes.compute().shape)
        
        
      
        
    ###################################################### Phase Gradients ############################################################
    if _gcf_parms['phase_gradient_term']:
        print('#########  Creating pg_term ')
        
    #print(cf_a_planes)
    #print(cf_w_planes)
    
    #print(cf_a_planes.compute().shape)
    #print(cf_w_planes.compute().shape)
    
    #https://github.com/pydata/xarray/issues/3476
    del gcf_dataset.a_indx.encoding["filters"]
    del gcf_dataset.gcf_indx.encoding["filters"]
    del gcf_dataset.pg_indx.encoding["filters"]
    del gcf_dataset.w_indx.encoding["filters"]
    del gcf_dataset.a_pair_indx.encoding["filters"]
    del gcf_dataset.GCF_PARMS_INDX.encoding["filters"]
    
    '''
    coords = {'a_id':gcf_dataset.A_PARMS_INDX.data[:,3].compute(),'w_id':gcf_dataset.W_PARMS_INDX.data[:,0].compute()}
    gcf_dataset = gcf_dataset.assign_coords(coords)
    
    gcf_dataset1 = xr.Dataset()
    gcf_dataset['GCF_A_PLANES'] = xr.DataArray(cf_a_planes, dims=('a_id','pol','conv_x','conv_y'))
    gcf_dataset['GCF_W_PLANES'] = xr.DataArray(cf_w_planes, dims=('w_id','conv_x','conv_y'))
    
    gcf_dataset.to_zarr('alma_vla.gcf.zarr',mode='w')
    
    print(gcf_dataset)
    '''
    
    gcf_dataset2 = xr.open_zarr('alma_vla.gcf.zarr')
    
    _create_gcf_graph(gcf_dataset2, gcf_parms, grid_parms, sel_parms)
    
    #print(gcf_dataset)
    #gcf_dataset.to_zarr('alma_vla.gcf.zarr',mode="w")
    
    '''
    gcf_dataset1 = xr.Dataset()
    gcf_dataset1['GCF_A_PLANES'] = xr.DataArray(da.from_array(cf_a_planes.compute(),(1,1,2048,2048)), dims=('a2','pol','conv_x','conv_y'))
    gcf_dataset1['GCF_W_PLANES'] = xr.DataArray(da.from_array(cf_w_planes.compute(),(1,2048,2048)), dims=('w2','conv_x','conv_y'))
    
    print(gcf_dataset1)
    gcf_dataset1.to_zarr('alma_vla.gcf.zarr',mode="w")
    '''
    

def _create_gcf_graph(gcf_dataset, gcf_parms, grid_parms, sel_parms):

    import itertools
    n_chunks_in_each_dim = gcf_dataset.GCF_PARMS_INDX.data.numblocks
    chunk_sizes = gcf_dataset.GCF_PARMS_INDX.data.chunks
    
    #iter_chunks_indx = itertools.product(np.arange(n_chunks_in_each_dim[0]), np.arange(n_chunks_in_each_dim[1]),
     #                                    np.arange(n_chunks_in_each_dim[2]))

    gcf =[]
    
    
    print(gcf_dataset)
    for c_gcf in range(1):#range(n_chunks_in_each_dim[0]):
        print('c_gcf',c_gcf)

        a_selection = np.unique(np.ravel(gcf_dataset.GCF_PARMS_INDX.data.partitions[c_gcf,:][:,0:2])).compute()
        w_selection = np.unique(np.ravel(gcf_dataset.GCF_PARMS_INDX.data.partitions[c_gcf,:][:,2])).compute()
        
#        print(a_selection)
#        print(gcf_dataset.a_id)
#        print('*****************')
#        print(w_selection)
#        print(gcf_dataset.w_id)
        
        gcf_chunk = _create_gcf(gcf_dataset.GCF_PARMS_INDX.data.partitions[c_gcf,:].compute(),
            gcf_dataset.GCF_A_PLANES.sel(a_id=a_selection).compute(),
            gcf_dataset.GCF_W_PLANES.sel(w_id=w_selection).compute(),
            gcf_parms['mueller_selection'],
            gcf_parms['conv_size'],
            gcf_parms['relative_threshold'],
            gcf_parms['oversampling'])
        
        gcf.append(gcf_chunk)
        
        '''
        gcf_chunk = dask.delayed(_create_gcf)(
            gcf_dataset.GCF_PARMS_INDX.data.partitions[c_gcf,:],
            gcf_dataset.GCF_A_PLANES.sel(a_id=a_selection),
            gcf_dataset.GCF_W_PLANES.sel(w_id=w_selection),
            dask.delayed(gcf_parms['mueller_selection']),
            dask.delayed(gcf_parms['conv_size']))
            
        gcf.append(gcf_chunk)
        '''
        
    print('************')
    print(gcf)
    dask.compute(gcf)
    #dask.visualize(gcf,'dask_debug.png')
 
from ._imaging_utils._general import _ndim_list

def _create_gcf(gcf_parms_indx,gcf_a_planes,gcf_w_planes,mueller_selection,conv_size,relative_threshold,oversampling):
    #print(gcf_a_planes)
    
    map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]])
    #gcf_planes = np.zeros((len(gcf_parms_indx),len(mueller_selection),conv_size[0],conv_size[1]),np.complex128)
    gcf_planes = _ndim_list((len(gcf_parms_indx),len(mueller_selection)))
    
    
    #print(gcf_planes.shape)
    
    for ii,parm_indx in enumerate(gcf_parms_indx[0:1,:]):
        print(parm_indx)
        w_plane = gcf_w_planes.sel(w_id=parm_indx[2]).data
        a1_plane = gcf_a_planes.sel(a_id=parm_indx[0]).data
        a2_plane = gcf_a_planes.sel(a_id=parm_indx[1]).data
        
        for jj,mm in enumerate(mueller_selection):
            print('map_mueler_to_pol[mm]',map_mueler_to_pol[mm])
            #print(w_plane.shape,a2_plane[map_mueler_to_pol[mm,1],:,:].shape)
            a_pair = a1_plane[map_mueler_to_pol[mm,0],:,:]*(a2_plane[map_mueler_to_pol[mm,1],:,:].conj())
            temp_plane = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(w_plane*a_pair)))
            #temp_sqrd_plane = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(w_plane*a_pair*(a_pair.conj()))))
            temp_sqrd_plane = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(w_plane*a_pair*((w_plane*a_pair).conj()))))
            temp_plane = temp_plane
            #temp_plane = w_plane#*a1_plane[map_mueler_to_pol[mm,0],:,:]*a2_plane[map_mueler_to_pol[mm,1],:,:]
            
            print('temp_plane.shape',temp_plane.shape)
            if jj == 0:
                start_indx,end_indx,support = resize_and_calc_support(temp_sqrd_plane,oversampling,relative_threshold)
           
            temp_sqrd_plane = temp_sqrd_plane[start_indx[0]:end_indx[0]+1,start_indx[1]:end_indx[1]+1]
            temp_plane = temp_plane[start_indx[0]:end_indx[0]+1,start_indx[1]:end_indx[1]+1]
                
            temp_plane = reshape_image(oversampling,support,temp_plane)
            temp_sqrd_plane = reshape_image(oversampling,support,temp_sqrd_plane)
      
            print('temp_plane.shape',temp_plane.shape)
            
            #print(temp_plane.shape)
            #gcf_planes[ii][jj]
            
            plt.figure()
            plt.imshow(np.abs(temp_plane[1,1,:,:]))
            
            #plt.figure()
            #plt.imshow(np.abs(temp_sqrd_plane)/np.max(np.abs(temp_sqrd_plane)))
            
            #plt.figure()
            #plt.imshow(np.real(w_plane))
            
        
            plt.show()
            
        #https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array bounding box
        #https://github.com/pydata/xarray/issues/1482 multi index
        # VLACalcIlluminationConvFunc::fillPB only conj used for pbSquared
        # VLACalcIlluminationConvFunc::skyMuller only conj for Mueller pairs
 
    
    
    return gcf_parms_indx
    
    
def reshape_image(v,s,img):
    #v oversampling
    #s support
    print(v,s,img.shape)
    img_reshaped = np.zeros((v[0]+1,v[1]+1,s[0],s[1]),img.dtype)
    
    v_c = v//2 # oversampling center
    s_c = s//2      # support center
    c_c = np.array(img.shape)//2    #image center
    
    for v_x in range(v[0]+1):
        for v_y in range(v[1]+1):
            x_center_ind = (v_x - v_c[0]) + c_c[0]
            y_center_ind = (v_y - v_c[1]) + c_c[1]
            for x in range(s[0]):
                for y in range(s[1]):
                    x_ind = v[0]*(x-s_c[0]) + x_center_ind
                    y_ind = v[1]*(y-s_c[1]) + y_center_ind
                    #print(v_x,v_y,x,y,x_ind,y_ind)
                    img_reshaped[v_x,v_y,x,y] = img[x_ind,y_ind]
    
    print('img_reshaped',img_reshaped.shape)
    return img_reshaped
    
    
#https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def resize_and_calc_support(img,oversampling,relative_threshold):
    threshold = np.max(np.abs(img))*relative_threshold
    
    print(threshold)

    rows = np.any(img > threshold, axis=1)
    cols = np.any(img > threshold, axis=0)
    
    xmin, xmax = np.where(rows)[0][[0, -1]]
    ymin, ymax = np.where(cols)[0][[0, -1]]
    
    img_shape = np.array(img.shape)
    
    print('1.',xmin, xmax, ymin, ymax, img_shape)
    xmin, xmax = recenter(xmin,xmax,img_shape[0])
    ymin, ymax = recenter(ymin,ymax,img_shape[1])
    print('2.',xmin, xmax, ymin, ymax, img_shape)
    
    img_new_shape=np.array([xmax-xmin+1,ymax-ymin+1])
    
    support = (np.ceil((img_new_shape-1)/oversampling)).astype(int)
    print('support',support)
    
    img_new_shape = (oversampling*support + 1)
    
    print(img_shape,img_new_shape)
    start_indx = (img_shape//2 - img_new_shape//2).astype(int)
    end_indx = (img_shape//2 + (img_new_shape - img_new_shape//2 - 1)).astype(int)
    print(start_indx,end_indx)
    

    
    #print('new_img shape', new_img.shape,img_new_shape,new_img[new_img.shape[0]//2,new_img.shape[1]//2], img[img.shape[0]//2,img.shape[1]//2])

    return start_indx, end_indx, support
    
def recenter(min_indx,max_indx,axis_len):
    axis_center = axis_len//2

    if (axis_center-min_indx) > (max_indx-axis_center):
        max_indx = 2*axis_center - min_indx
        if max_indx > axis_len-1:
            max_indx = axis_len-1
    else:
        min_indx = 2*axis_center - max_indx
        if min_indx < 0:
            min_indx = 0
            
    return min_indx, max_indx
    
'''
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
'''
    
    
def _create_a_term_graph(gcf_dataset, list_zpc_dataset, gcf_parms, grid_parms, sel_parms):
    from ._imaging_utils._a_term import _calc_a_sky
    
    
    
    '''Check this
    print(a_parm_indx.shape[1])
    if a_parm_indx.shape[1] != 6:
        print("no chunking over a_indx allowed")
    '''
    
    beam_id = dask.delayed(gcf_dataset.beam_id.values)
    a_beam_id = dask.delayed(gcf_dataset.A_BEAM_ID.data.compute())
    a_freq = dask.delayed(gcf_dataset.A_FREQ.data.compute())
    a_pa = dask.delayed(gcf_dataset.A_PA.data.compute())
    
    print(gcf_dataset.A_PARMS_INDX[:,:])
    
    map_mueler_to_pol = np.array([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3],[2,0],[2,1],[3,0],[3,1],[2,2],[2,3],[3,2],[3,3]]) #place somewhere special
    #print('1')
    gcf_parms['needed_pol'] = np.unique(np.ravel(map_mueler_to_pol[gcf_parms['mueller_selection']]))
    #print('2')
    #print('needed_pol',gcf_parms['needed_pol'])
    
    chunks = (gcf_dataset.A_PARMS_INDX.chunks[0],len(gcf_parms['needed_pol']),gcf_parms['conv_size'][0],gcf_parms['conv_size'][1])
    
    cf_a_sky_planes = da.map_blocks(_calc_a_sky,gcf_dataset.A_PARMS_INDX.data,list_zpc_dataset,a_freq,a_pa,gcf_parms,grid_parms,dtype=np.complex128,chunks=chunks,drop_axis=1,new_axis=(1,2,3))
    
    '''
    start = time.time()
    cf_a_sky_planes = dask.compute(cf_a_sky_planes)[0]
    print('a time',time.time()-start)
    
    print(cf_a_sky_planes.shape)
    
    plt.figure()
    plt.imshow(np.real(cf_a_sky_planes[0,0,:,:]))
    
    plt.figure()
    plt.imshow(np.real(cf_a_sky_planes[15,1,:,:]))
    plt.show()
    '''

    return cf_a_sky_planes
    


def _create_w_term_graph(gcf_dataset, gcf_parms, grid_parms, sel_parms):
    from ._imaging_utils._w_term import _calc_w_sky

    #print('1')
    chunks = (gcf_dataset.W_PARMS_INDX.chunks[0],gcf_parms['conv_size'][0],gcf_parms['conv_size'][1])
    w_val = gcf_dataset.W.data.compute()
    
    #print(gcf_dataset.W_PARMS_INDX.data)
#    print(w_val)
#    print(gcf_dataset.W.data.compute())
#    print(gcf_dataset.W.chunks[0][0])
#
#    print('gcf_dataset.W.chunks',gcf_dataset.W.chunks)
#    print('w chunks',chunks)
    cf_w_sky_planes = da.map_blocks(_calc_w_sky,gcf_dataset.W_PARMS_INDX.data,w_val,gcf_parms,grid_parms,dtype=np.complex128,chunks=chunks,drop_axis=1,new_axis=(1,2))

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
    return cf_w_sky_planes
    
    
    
    
    


        

   
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
