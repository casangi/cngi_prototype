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
"""
this module will be included in the api
"""

import numpy as np
from numba import jit
# silence NumbaPerformanceWarning
import warnings
from numba.errors import NumbaPerformanceWarning
import scipy

#warnings.filterwarnings("ignore", category=NumbaPerformanceWarning) #Suppress  NumbaPerformanceWarning: '@' is faster on contiguous arrays warning. This happens for phasor_loop and apply_rotation_matrix functions.

def phase_rotate_numba(vis_dataset, global_dataset, rotation_parms, sel_parms, storage_parms):
    """
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
    
    #print('1. numba',vis_dataset.DATA[:,0,0,0].values)
    
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
    import dask
    
    _sel_parms = copy.deepcopy(sel_parms)
    _rotation_parms = copy.deepcopy(rotation_parms)
    _storage_parms = copy.deepcopy(storage_parms)
    
    assert(_check_sel_parms(_sel_parms,{'uvw_in':'UVW','uvw_out':'UVW_ROT','data_in':'DATA','data_out':'DATA_ROT'})), "######### ERROR: sel_parms checking failed"
    assert(_check_existence_sel_parms(vis_dataset,{'uvw_in':_sel_parms['uvw_in'],'data_in':_sel_parms['data_in']})), "######### ERROR: sel_parms checking failed"
    assert(_check_rotation_parms(_rotation_parms)), "######### ERROR: rotation_parms checking failed"
    assert(_check_storage_parms(_storage_parms,'dataset.vis.zarr','phase_rotate')), "######### ERROR: storage_parms checking failed"
    
    assert(_sel_parms['uvw_out'] != _sel_parms['uvw_in']), "######### ERROR: sel_parms checking failed sel_parms['uvw_out'] can not be the same as sel_parms['uvw_in']."
    assert(_sel_parms['data_out'] != _sel_parms['data_in']), "######### ERROR: sel_parms checking failed sel_parms['data_out'] can not be the same as sel_parms['data_in']."

    #Phase center
    ra_image = _rotation_parms['image_phase_center'][0]
    dec_image = _rotation_parms['image_phase_center'][1]
    
    rotmat_image_phase_center = R.from_euler('XZ',[[np.pi/2 - dec_image, - ra_image + np.pi/2]]).as_matrix()[0]
    image_phase_center_cosine = _directional_cosine([ra_image,dec_image])
    
    n_fields = global_dataset.dims['field']
    field_names = global_dataset.field
    uvw_rotmat = np.zeros((n_fields,3,3),np.double)
    phase_rotation = np.zeros((n_fields,3),np.double)
    
    fields_phase_center = global_dataset.FIELD_PHASE_DIR.values[:,:,vis_dataset.attrs['ddi']]
    
    #print(fields_phase_center)
    
    #Create a rotation matrix for each field
    for i_field in range(n_fields):
        #Not sure if last dimention in FIELD_PHASE_DIR is the ddi number
        field_phase_center = fields_phase_center[i_field,:]
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
    #@jit(nopython=True, cache=True, nogil=True)
    def apply_rotation_matrix(uvw, field_id, uvw_rotmat):
        #print(uvw.shape,field_id.shape,uvw_rotmat.shape)
        for i_time in range(uvw.shape[0]):
            #print('The index is',np.where(field_names==field[i_time])[1][0],field_id[i_time,0,0])
            #uvw[i_time,:,0:2] = -uvw[i_time,:,0:2] this gives the same result as casa (in the ftmachines uvw(negateUV(vb)) is used). In ngcasa we don't do this since the uvw definition in the gridder and vis.zarr are the same.
            #uvw[i_time,:,:] = np.matmul(uvw[i_time,:,:],uvw_rotmat[field_id[i_time],:,:])
            uvw[i_time,:,:] = uvw[i_time,:,:] @ uvw_rotmat[field_id[i_time,0,0],:,:]
        return uvw
    
    uvw = da.map_blocks(apply_rotation_matrix,vis_dataset[_sel_parms['uvw_in']].data, vis_dataset.field_id.data[:,None,None],uvw_rotmat,dtype=np.double)

    #dask.visualize(uvw,filename='uvw')
    

        
    chan_chunk_size = vis_dataset[_sel_parms['data_in']].chunks[2][0]
    freq_chan = da.from_array(vis_dataset.coords['chan'].values, chunks=(chan_chunk_size))
    
    #print('2. numba',vis_dataset[_sel_parms['data_in']][:,0,0,0].values)
    vis_rot = da.map_blocks(apply_phasor,vis_dataset[_sel_parms['data_in']].data,uvw[:,:,:,None], vis_dataset.field_id.data[:,None,None,None],freq_chan[None,None,:,None],phase_rotation,_rotation_parms['common_tangent_reprojection'],dtype=np.complex)
    
    #dask.visualize(uvw,filename='uvw_rot')
    #dask.visualize(vis_rot,filename='vis_rot')
    
    vis_dataset[_sel_parms['uvw_out']] =  xr.DataArray(uvw, dims=vis_dataset[_sel_parms['uvw_in']].dims)
    vis_dataset[_sel_parms['data_out']] =  xr.DataArray(vis_rot, dims=vis_dataset[_sel_parms['data_in']].dims)
    
#    dask.visualize(vis_dataset[_sel_parms['uvw_out']],filename='uvw_rot_dataset')
#    dask.visualize(vis_dataset[_sel_parms['data_out']],filename='vis_rot_dataset')
#    dask.visualize(vis_dataset,filename='vis_dataset_before_append')
    
    list_xarray_data_variables = [vis_dataset[_sel_parms['uvw_out']],vis_dataset[_sel_parms['data_out']]]
    return _store(vis_dataset,list_xarray_data_variables,_storage_parms)


@jit(nopython=True,cache=True, nogil=True)
def phasor_loop(phase_direction,uvw,phase_rotation,field_id,end_slice):
    for i_time in range(uvw.shape[0]):
        phase_direction[i_time,:] = uvw[i_time,:,0:end_slice,0] @ phase_rotation[field_id[i_time,0,0,0],0:end_slice]


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

   

#Apply rotation to vis data
@jit(nopython=True,cache=True, nogil=True)
def apply_phasor(vis_data,uvw, field_id,freq_chan,phase_rotation,common_tangent_reprojection):
    #print(vis_data.shape,uvw.shape,field_id.shape,freq_chan.shape,phase_rotation.shape)
    
    #print(vis_data[:,0,0,0])
    
    if common_tangent_reprojection:
        end_slice = 2
    else:
        end_slice = 3
        
        
    #start = time.time()
    
    N_time = vis_data.shape[0]
    N_baseline = vis_data.shape[1]
    N_chan = vis_data.shape[2]
    N_pol = vis_data.shape[3]
    
    #########################
    for i_time in range(N_time):
        for i_baseline in range(N_baseline):
            if common_tangent_reprojection:
                phase_direction = uvw[i_time,i_baseline,0,0] * phase_rotation[field_id[i_time,0,0,0],0] + uvw[i_time,i_baseline,1,0] * phase_rotation[field_id[i_time,0,0,0],1]
            else:
                phase_direction = uvw[i_time,i_baseline,0,0] * phase_rotation[field_id[i_time,0,0,0],0] + uvw[i_time,i_baseline,1,0] * phase_rotation[field_id[i_time,0,0,0],1] +uvw[i_time,i_baseline,2,0] * phase_rotation[field_id[i_time,0,0,0],2]
                
            
            for i_chan in range(N_chan):
                for i_pol in range(N_pol):
                    vis_data[i_time,i_baseline,i_chan,i_pol] = vis_data[i_time,i_baseline,i_chan,i_pol]*np.exp(2.0*1j*np.pi*phase_direction*freq_chan[0,0,i_chan,0]/scipy.constants.c)
    #print('Time to apply phasor',time.time()-start)
    
    #print(vis_data[:,0,0,0])
    
    #vis_rot[np.isnan(vis_rot)] = np.nan
    vis_data = (vis_data.astype(np.complex64)).astype(np.complex128)
    
    return vis_data
