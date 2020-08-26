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


import numpy as np
from numba import jit
# silence NumbaPerformanceWarning
import warnings
from numba.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning) #Suppress  NumbaPerformanceWarning: '@' is faster on contiguous arrays warning. This happens for phasor_loop and apply_rotation_matrix functions.

def phase_rotate(vis_dataset, global_dataset, rotation_parms, sel_parms, storage_parms):
    """
    ********* Experimental Function *************
    Rotate uvw with faceting style rephasing for multifield mosaic.
    The specified phasecenter and field phase centers are assumed to be in the same frame.
    This does not support east-west arrays, emphemeris objects or objects within the nearfield.
    (no refocus).
    
    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        input Visibility Dataset
    Returns
    -------
    psf_dataset : xarray.core.dataset.Dataset
    """
    #based on UVWMachine and FTMachine
    #measures/Measures/UVWMachine.cc
    
    #Important: Can not applyflags before calling rotate (uvw coordinates are also flagged). This will destroy the rotation transform.
    #Performance improvements apply_rotation_matrix (jit code)
    
    from ngcasa._ngcasa_utils._store import _store
    from scipy.spatial.transform import Rotation as R
    import scipy
    import numpy as np
    import copy
    import dask.array as da
    import xarray as xr
    from ngcasa._ngcasa_utils._check_parms import _check_storage_parms, _check_sel_parms, _check_existence_sel_parms
    from ._imaging_utils._check_imaging_parms import _check_rotation_parms
    import time
    import numba
    from numba import double
    
    _sel_parms = copy.deepcopy(sel_parms)
    _rotation_parms = copy.deepcopy(rotation_parms)
    _storage_parms = copy.deepcopy(storage_parms)
    
    assert(_check_sel_parms(_sel_parms,{'uvw_in':'UVW','uvw_out':'UVW_ROT','data_in':'DATA','data_out':'DATA_ROT'})), "######### ERROR: sel_parms checking failed"
    assert(_check_existence_sel_parms(vis_dataset,{'uvw_in':_sel_parms['uvw_in'],'data_in':_sel_parms['data_in']})), "######### ERROR: sel_parms checking failed"
    assert(_check_rotation_parms(_rotation_parms)), "######### ERROR: rotation_parms checking failed"
    assert(_check_storage_parms(_storage_parms,'dataset.vis.zarr','phase_rotate')), "######### ERROR: storage_parms checking failed"
    
    assert(_sel_parms['uvw_out'] != _sel_parms['uvw_in']), "######### ERROR: sel_parms checking failed sel_parms['uvw_out'] can not be the same as sel_parms['uvw_in']."
    assert(_sel_parms['data_out'] != _sel_parms['data_in']), "######### ERROR: sel_parms checking failed sel_parms['data_out'] can not be the same as sel_parms['data_in']."


    #I think this should be included in vis_dataset. There should also be a beter pythonic way to do the loop inside gen_field_indx.
    def gen_field_indx(vis_data_field_names, field_names):
        field_indx = np.zeros(vis_data_field_names.shape,np.int)
        for i_field, field_name in enumerate(field_names):
            field_indx[vis_data_field_names == field_name] = i_field
            
        return field_indx
    field_indx = da.map_blocks(gen_field_indx, vis_dataset['field'].data, global_dataset['field'], dtype=np.int)
    

    #If no image phase center is specified use first field
    '''
    if isinstance(_rotation_parms['image_phase_center'],int):
        ra_image = global_dataset.FIELD_PHASE_DIR.values[_rotation_parms['image_phase_center'],:,vis_dataset.attrs['ddi']][0]
        dec_image = global_dataset.FIELD_PHASE_DIR.values[_rotation_parms['image_phase_center'],:,vis_dataset.attrs['ddi']][1]
    else:
        if 'image_phase_center' in _rotation_parms:
            ra_image = _rotation_parms['image_phase_center'][0]
            dec_image = _rotation_parms['image_phase_center'][1]
        else:
            ra_image = global_dataset.FIELD_PHASE_DIR.values[0,:,vis_dataset.attrs['ddi']][0]
            dec_image = global_dataset.FIELD_PHASE_DIR.values[0,:,vis_dataset.attrs['ddi']][1]
    '''
       
    #Phase center
    ra_image = _rotation_parms['image_phase_center'][0]
    dec_image = _rotation_parms['image_phase_center'][1]
    
    rotmat_image_phase_center = R.from_euler('XZ',[[np.pi/2 - dec_image, - ra_image + np.pi/2]]).as_matrix()[0]
    image_phase_center_cosine = _directional_cosine([ra_image,dec_image])
    
    n_fields = global_dataset.dims['field']
    uvw_rotmat = np.zeros((n_fields,3,3),np.double)
    phase_rotation = np.zeros((n_fields,3),np.double)
    
    #Create a rotation matrix for each field
    for i_field in range(n_fields):
        #Not sure if last dimention in FIELD_PHASE_DIR is the ddi number
        field_phase_center = global_dataset.FIELD_PHASE_DIR.values[i_field,:,vis_dataset.attrs['ddi']]
        # Define rotation to a coordinate system with pole towards in-direction
        # and X-axis W; by rotating around z-axis over -(90-long); and around
        # x-axis (lat-90).
        rotmat_field_phase_center = R.from_euler('ZX',[[-np.pi/2 + field_phase_center[0],field_phase_center[1] - np.pi/2]]).as_matrix()[0]
        uvw_rotmat[i_field,:,:] = np.matmul(rotmat_image_phase_center,rotmat_field_phase_center).T
        
        if _rotation_parms['common_tangent_reprojection'] == True:
            uvw_rotmat[i_field,2,0:2] = 0.0 # (Common tangent rotation needed for joint mosaics, see last part of FTMachine::girarUVW in CASA)
        
        field_phase_center_cosine = _directional_cosine(field_phase_center)
        phase_rotation[i_field,:] = np.matmul(rotmat_image_phase_center,(image_phase_center_cosine - field_phase_center_cosine))

    
    #Apply rotation matrix to uvw
    @jit(nopython=True, cache=True, nogil=True)
    def apply_rotation_matrix(uvw, field_indx, uvw_rotmat):
        #print(uvw.shape,field_indx.shape,uvw_rotmat.shape)
        
        #uvw_rot = np.zeros()
        for i_time in range(uvw.shape[0]):
            #uvw[i_time,:,0:2] = -uvw[i_time,:,0:2] this gives the same result as casa (in the ftmachines uvw(negateUV(vb)) is used). In ngcasa we don't do this since the uvw definition in the gridder and vis.zarr are the same.
            #uvw[i_time,:,:] = np.matmul(uvw[i_time,:,:],uvw_rotmat[field_indx[i_time],:,:])
            uvw[i_time,:,:] = uvw[i_time,:,:] @ uvw_rotmat[field_indx[i_time,0,0],:,:]
        return uvw
    
    uvw = da.map_blocks(apply_rotation_matrix,vis_dataset[_sel_parms['uvw_in']].data, field_indx[:,None,None],uvw_rotmat,dtype=np.double)

    
    #Apply rotation to vis data
    def apply_phasor(vis_data,uvw, field_indx,freq_chan,phase_rotation,common_tangent_reprojection):
        #print(vis_data.shape,uvw.shape,field_indx.shape,freq_chan.shape,phase_rotation.shape)
        
        if common_tangent_reprojection == True:
            end_slice = 2
        else:
            end_slice = 3
        
        phase_direction = np.zeros(uvw.shape[0:2],np.double)
        phasor_loop(phase_direction,uvw,phase_rotation,field_indx,end_slice)
        #for i_time in range(uvw.shape[0]):
        #    phase_direction[i_time,:] = uvw[i_time,:,0:end_slice,0] @ phase_rotation[field_indx[i_time,0,0,0],0:end_slice]
        
        n_chan = vis_data.shape[2]
        n_pol = vis_data.shape[3]
        
        phase_direction = np.transpose(np.broadcast_to(phase_direction,((n_chan,)+uvw.shape[0:2])), axes=(1,2,0))
        phasor = np.exp(2.0*1j*np.pi*phase_direction*np.broadcast_to(freq_chan[:,0,0,0],uvw.shape[0:2]+(n_chan,))/scipy.constants.c) # phasor_ngcasa = - phasor_casa. Sign flip is due to CASA gridders convention sign flip.
        phasor = np.transpose(np.broadcast_to(phasor,((n_pol,)+vis_data.shape[0:3])), axes=(1,2,3,0))
        vis_rot = vis_data*phasor
        
        return vis_rot
        
    chan_chunk_size = vis_dataset[_sel_parms['data_in']].chunks[2][0]
    freq_chan = da.from_array(vis_dataset.coords['chan'].values, chunks=(chan_chunk_size))
    vis_rot = da.map_blocks(apply_phasor,vis_dataset[_sel_parms['data_in']].data,uvw[:,:,:,None], field_indx[:,None,None,None],freq_chan[None,None,:,None],phase_rotation,_rotation_parms['common_tangent_reprojection'],dtype=np.complex)
    
    vis_dataset[_sel_parms['uvw_out']] =  xr.DataArray(uvw, dims=vis_dataset[_sel_parms['uvw_in']].dims)
    vis_dataset[_sel_parms['data_out']] =  xr.DataArray(vis_rot, dims=vis_dataset[_sel_parms['data_in']].dims)
    
    list_xarray_data_variables = [vis_dataset[_sel_parms['uvw_out']],vis_dataset[_sel_parms['data_out']]]
    return _store(vis_dataset,list_xarray_data_variables,_storage_parms)


#@jit(nopython=True,cache=True, nogil=True)
def phasor_loop(phase_direction,uvw,phase_rotation,field_indx,end_slice):
    for i_time in range(uvw.shape[0]):
        phase_direction[i_time,:] = uvw[i_time,:,0:end_slice,0] @ phase_rotation[field_indx[i_time,0,0,0],0:end_slice]


def _directional_cosine(phase_center_in_radians):
   '''
   # In https://arxiv.org/pdf/astro-ph/0207413.pdf see equation 160
   phase_center_in_radians (RA,DEC)
   '''
   import numpy as np
   
   phase_center_cosine = np.zeros((3,))
   phase_center_cosine[0] = np.cos(phase_center_in_radians[0])*np.cos(phase_center_in_radians[1])
   phase_center_cosine[1] = np.sin(phase_center_in_radians[0])*np.cos(phase_center_in_radians[1])
   phase_center_cosine[2] = np.sin(phase_center_in_radians[1])
   return phase_center_cosine

   

