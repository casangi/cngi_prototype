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

def mosaic_rotate_uvw(vis_dataset, global_dataset, user_rotation_parms,user_storage_parms):
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
    
    from ngcasa._ngcasa_utils._store import _store
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    import copy
    import dask.array as da
    
    rotation_parms = copy.deepcopy(user_rotation_parms)
    # if append true than rotation_parms['uvw_out_name'] != rotation_parms['uvw_in_name']
    

    #I think this should be included in vis_dataset. There should also be a beter pythonic way to do the loop inside apply_rotation_matrix.
    def apply_rotation_matrix(vis_data_field_names, field_names):
        field_indx = np.zeros(vis_data_field_names.shape,np.int)
        for i_field, field_name in enumerate(field_names):
            field_indx[vis_data_field_names == field_name] = i_field
            
        return field_indx
    field_indx = da.map_blocks(apply_rotation_matrix, vis_dataset['field'].data, global_dataset['field'].data, dtype=np.int)
    print(field_indx)
    
    #Create Rotation Matrices
    ra_image = rotation_parms['image_phase_center'].ra.radian
    dec_image = rotation_parms['image_phase_center'].dec.radian
    rotmat_image_phase_center = R.from_euler('XZ',[[np.pi/2 - dec_image, - ra_image + np.pi/2]]).as_matrix()[0]
    image_phase_center_cosine = _directional_cosine([ra_image,dec_image])
    
    n_fields = global_dataset.dims['field']
    uvw_rotmat = np.zeros((n_fields,3,3),np.double)
    phase_rotation = np.zeros((n_fields,3),np.double)
    
    for i_field in range(n_fields):
        #Not sure if last dimention in FIELD_PHASE_DIR is the ddi number
        field_phase_center = global_dataset.FIELD_PHASE_DIR.values[i_field,:,vis_dataset.attrs['ddi']]
        # Define rotation to a coordinate system with pole towards in-direction
        # and X-axis W; by rotating around z-axis over -(90-long); and around
        # x-axis (lat-90).
        rotmat_field_phase_center = R.from_euler('ZX',[[-np.pi/2 + field_phase_center[0],field_phase_center[1] - np.pi/2]]).as_matrix()[0]
        uvw_rotmat[i_field,:,:] = np.matmul(rotmat_image_phase_center,rotmat_field_phase_center).T
        uvw_rotmat[i_field,2,0:2] = 0.0 #Not sure why this should be done (see last part of FTMachine::girarUVW in CASA)
        
        field_phase_center_cosine = _directional_cosine(field_phase_center)
        phase_rotation[i_field,:] = np.matmul(rotmat_image_phase_center,(image_phase_center_cosine - field_phase_center_cosine))
    
    
    def apply_rotation_matrix(uvw, field_indx, uvw_rotmat):
        print(uvw.shape,field_indx.shape,uvw_rotmat.shape)
        for i_time in range(uvw.shape[0]):
            #uvw[i_time,:,0:2] = -uvw[i_time,:,0:2] this gives the same result as casa (in the ftmachines uvw(negateUV(vb)) is used). In ngcasa we don't do this since the uvw definition in the gridder and vis.zarr are the same.
            uvw[i_time,:,:] = np.matmul(uvw[i_time,:,:],uvw_rotmat[field_indx[i_time],:,:])
        return uvw
    uvw = da.map_blocks(apply_rotation_matrix,vis_dataset['UVW'].data, field_indx,uvw_rotmat,dtype=np.double)
    
    def calc_uv_phase_direction(uvw, field_indx,phase_rotation):
        phase_direction = np.zeros(uvw.shape[0:2] + (1,),np.double)
        for i_time in range(uvw.shape[0]):
            phase_direction[i_time,:,0] = np.matmul(uvw[i_time,:,0:2],phase_rotation[field_indx[i_time],0:2])
        
        return phase_direction
        
    phase_direction = da.map_blocks(calc_uv_phase_direction,uvw, field_indx,phase_rotation,dtype=np.double,chunks=vis_dataset['UVW'].data.chunksize[0:2] + (1,))[:,:,0]
    
    vis_dataset[rotation_parms['uvw_out_name']] =  xr.DataArray(uvw, dims=vis_dataset[rotation_parms['uvw_in_name']].dims)
    
    list_xarray_data_variables = [vis_dataset[imaging_weights_parms['uvw_name']]]
    return _store(vis_dataset,list_xarray_data_variables,storage_parms)

def _directional_cosine(phase_center_in_radians):
   '''
   # In https://arxiv.org/pdf/astro-ph/0207413.pdf see equation 160
   phase_center_in_radians (RA,DEC)
   '''
   phase_center_cosine = np.zeros((3,))
   phase_center_cosine[0] = np.cos(phase_center_in_radians[0])*np.cos(phase_center_in_radians[1])
   phase_center_cosine[1] = np.sin(phase_center_in_radians[0])*np.cos(phase_center_in_radians[1])
   phase_center_cosine[2] = np.sin(phase_center_in_radians[1])
   return phase_center_cosine
