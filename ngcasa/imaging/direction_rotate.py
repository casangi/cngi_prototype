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
import scipy
#from numba import jit
# silence NumbaPerformanceWarning
#import warnings
#from numba.errors import NumbaPerformanceWarning
#warnings.filterwarnings("ignore", category=NumbaPerformanceWarning) #Suppress  NumbaPerformanceWarning: '@' is faster on contiguous arrays warning. This happens for phasor_loop and apply_rotation_matrix functions.

def direction_rotate(mxds, rotation_parms, sel_parms):
    """
    Rotate uvw coordinates and phase rotate visibilities. For a joint mosaics rotation_parms['common_tangent_reprojection'] must be true.
    The specified phasecenter and field phase centers are assumed to be in the same frame.
    East-west arrays, emphemeris objects or objects within the nearfield are not supported.
    
    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        Input vis.zarr multi dataset.
    rotation_parms : dictionary
    rotation_parms['image_phase_center'] : list of number, length = 2, units = radians
       The phase center to rotate to (right ascension and declination).
    rotation_parms['common_tangent_reprojection']  : bool, default = True
       If true common tangent reprojection is used (should be true if a joint mosaic image is being created).
    rotation_parms['single_precision'] : bool, default = True
       If rotation_parms['single_precision'] is true then the output visibilities are cast from 128 bit complex to 64 bit complex. Mathematical operations are always done in double precision.
    sel_parms : dict
    sel_parms['vis_description_in'] : dict, default = _vis_dataset.vis_description[0]
        The names of the input data and uvw data variables.
        For example sel_parms['vis_description_in'] = {'data':'DATA','uvw':'UVW'}.
    sel_parms['vis_description_out'] : dict, default = {**_vis_dataset.vis_description[0],**{'uvw':'UVW_ROT','data':'DATA_ROT','field_id':rotation_parms['image_phase_center']}}
        The names of the new data and uvw data variables that have been direction rotated.
        For example sel_parms['vis_description_in'] = {'data':'DATA_ROT','uvw':'UVW_ROT'}.
    Returns
    -------
    psf_dataset : xarray.core.dataset.Dataset
    """
    #Based on CASACORE measures/Measures/UVWMachine.cc and CASA code/synthesis/TransformMachines2/FTMachine.cc::girarUVW
    
    print('######################### Start direction_rotate #########################')
    
    import copy
    import dask.array as da
    import xarray as xr
    from cngi._utils._check_parms import _check_sel_parms, _check_existence_sel_parms
    from ._imaging_utils._check_imaging_parms import _check_rotation_parms
    import time
    import dask
    
    #Deep copy so that inputs are not modified
    _mxds = mxds.copy(deep=True)
    _sel_parms = copy.deepcopy(sel_parms)
    _rotation_parms = copy.deepcopy(rotation_parms)
    
    ##############Parameter Checking and Set Defaults##############
    assert(_check_rotation_parms(_rotation_parms)), "######### ERROR: rotation_parms checking failed"
    
    assert('xds' in _sel_parms), "######### ERROR: xds must be specified in sel_parms" #Can't have a default since xds names are not fixed.
    _vis_dataset = _mxds.attrs[sel_parms['xds']]

    #if 'vis_description_in_indx' in _sel_parms:
    #   _sel_parms['vis_description_in'] = _vis_dataset.vis_description[sel_parms['vis_description_in_indx']] # vis_description_in_indx is used so that is con be
    #   print("Setting vis_description_in  to ",_sel_parms['vis_description_in'])
        
    vis_description_out_defaults = {**_vis_dataset.vis_description[0],**{'uvw':'UVW_ROT','data':'DATA_ROT','field_id':rotation_parms['image_phase_center']}} #{**x, **y} merges dicts with y taking precedence for repeated entries.
    sel_defaults = {'vis_description_in':_vis_dataset.vis_description[0],'vis_description_out':vis_description_out_defaults}
    assert(_check_sel_parms(_sel_parms,sel_defaults)), "######### ERROR: sel_parms checking failed"
    
    sel_check = {'vis_description_in':_sel_parms['vis_description_in']}
    assert(_check_existence_sel_parms(_vis_dataset,sel_check)), "######### ERROR: sel_parms checking failed"
    
    #Possibly not needed anymore
    assert(_sel_parms['vis_description_in']['uvw'] != _sel_parms['vis_description_out']['uvw']), "######### ERROR: sel_parms checking failed sel_parms['uvw_out'] can not be the same as sel_parms['uvw_in']."
    assert(_sel_parms['vis_description_in']['data'] != _sel_parms['vis_description_out']['data']), "######### ERROR: sel_parms checking failed sel_parms['data_out'] can not be the same as sel_parms['data_in']."
    #assert(_sel_parms['vis_description_in']['field_id'] != _sel_parms['vis_description_out']['field_id']), "######### ERROR: sel_parms checking failed sel_parms['data_out'] can not be the same as sel_parms['data_in']."
    #################################################################
    
    #Calculate rotation matrices for each field (not parallelized)
    #n_field number of fields in _vis_dataset
    #uvw_rotmat n_field x 3 x 3
    #phase_rotation n_field x 3
    #rot_field_id n_field
    uvw_rotmat, phase_rotation, rot_field_id = calc_rotation_mats(_vis_dataset,_mxds.FIELD,_rotation_parms)
    
    #Apply rotation matrix to uvw
    #uvw time x baseline x 3
    uvw = da.map_blocks(apply_rotation_matrix,_vis_dataset[_sel_parms['vis_description_in']['uvw']].data, _vis_dataset.FIELD_ID.data[:,:,None],uvw_rotmat,rot_field_id,dtype=np.double)

    #Phase shift vis data
    #vis_rot time x baseline x chan x pol
    chan_chunk_size = _vis_dataset[_sel_parms['vis_description_in']['data']].chunks[2]
    freq_chan = da.from_array(_vis_dataset.coords['chan'].values, chunks=(chan_chunk_size))
    vis_rot = da.map_blocks(apply_phasor,_vis_dataset[_sel_parms['vis_description_in']['data']].data,uvw[:,:,:,None], _vis_dataset.FIELD_ID.data[:,:,None,None],freq_chan[None,None,:,None],phase_rotation,rot_field_id,_rotation_parms['common_tangent_reprojection'],_rotation_parms['single_precision'],dtype=np.complex)
    
    #Add new datavariables
    _vis_dataset[_sel_parms['vis_description_out']['uvw']] =  xr.DataArray(uvw, dims=_vis_dataset[_sel_parms['vis_description_in']['uvw']].dims)
    _vis_dataset[_sel_parms['vis_description_out']['data']] =  xr.DataArray(vis_rot, dims=_vis_dataset[_sel_parms['vis_description_in']['data']].dims)
    
    #Update vis_description
    _vis_dataset.attrs['vis_description'] = _vis_dataset.attrs['vis_description'] + [_sel_parms['vis_description_out']]

    
    ''' ? Should we do this
    #Update field
    field_id_rot = _mxds.FIELD.PHASE_DIR.where((_mxds.FIELD.PHASE_DIR[:,0,0] == rotation_parms['image_phase_center'][0]) & (_mxds.FIELD.PHASE_DIR[:,0,1] == rotation_parms['image_phase_center'][1]),drop=True).field_id.values
    
    if not field_id_rot:
        new_field_id = np.max(_mxds.FIELD.field_id) + 1
        #Add to FIELD table
    else:
        new_field_id = field_id_rot
        
    #Add new FIELD_ID data variable
    '''
    
    print('######################### Created graph for direction_rotate #########################')
    return _mxds
    
    
    
    
def calc_rotation_mats(vis_dataset,field_dataset,rotation_parms):
    from scipy.spatial.transform import Rotation as R
    #Phase center
    ra_image = rotation_parms['image_phase_center'][0]
    dec_image = rotation_parms['image_phase_center'][1]
    
    rotmat_image_phase_center = R.from_euler('XZ',[[np.pi/2 - dec_image, - ra_image + np.pi/2]]).as_matrix()[0]
    image_phase_center_cosine = _directional_cosine([ra_image,dec_image])
    
    field_id = np.unique(vis_dataset.FIELD_ID) #??????? Or should the roation matrix be calculated for all fields #field_id = field_dataset.field_id
    
    n_fields = len(field_id)
    uvw_rotmat = np.zeros((n_fields,3,3),np.double)
    phase_rotation = np.zeros((n_fields,3),np.double)

    # Can't use fields_phase_center = field_dataset.PHASE_DIR.sel(field_id=field_id) because PHASE_DIR dims are d0,d1,d2. field_id(d0) a non-dimension coordinate
    # https://github.com/pydata/xarray/issues/934, https://github.com/pydata/xarray/issues/2028
    # https://github.com/pydata/xarray/issues/1603 is a planned refactor
    # Consequently have to use where. All subtables have this issue.
    fields_phase_center = field_dataset.PHASE_DIR.where(field_dataset.field_id==field_id, drop=True)[:,0,:] #n_fields x 2
    
    #Create a rotation matrix for each field
    for i_field in range(n_fields):
        field_phase_center = fields_phase_center[i_field,:]
        # Define rotation to a coordinate system with pole towards in-direction
        # and X-axis W; by rotating around z-axis over -(90-long); and around
        # x-axis (lat-90).
        rotmat_field_phase_center = R.from_euler('ZX',[[-np.pi/2 + field_phase_center[0],field_phase_center[1] - np.pi/2]]).as_matrix()[0]
        uvw_rotmat[i_field,:,:] = np.matmul(rotmat_image_phase_center,rotmat_field_phase_center).T
        
        if rotation_parms['common_tangent_reprojection'] == True:
            uvw_rotmat[i_field,2,0:2] = 0.0 # (Common tangent rotation needed for joint mosaics, see last part of FTMachine::girarUVW in CASA)
        
        field_phase_center_cosine = _directional_cosine(field_phase_center)
        phase_rotation[i_field,:] = np.matmul(rotmat_image_phase_center,(image_phase_center_cosine - field_phase_center_cosine))
    
    return uvw_rotmat, phase_rotation, field_id
    
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

#Apply rotation to uvw data
def apply_rotation_matrix(uvw, field_id, uvw_rotmat, rot_field_id):
    for i_time in range(uvw.shape[0]):
        #uvw[i_time,:,0:2] = -uvw[i_time,:,0:2] this gives the same result as casa (in the ftmachines uvw(negateUV(vb)) is used). In ngcasa we don't do this since the uvw definition in the gridder and vis.zarr are the same.
        field_id_t = field_id[i_time,:,:]
        rot_field_indx = np.where(rot_field_id == np.unique(field_id_t[~np.isnan(field_id_t)])[0])[0][0] #should be len 1
        uvw[i_time,:,:] = uvw[i_time,:,:] @ uvw_rotmat[rot_field_indx,:,:] #uvw time x baseline x uvw_indx, uvw_rotmat n_field x 3 x 3.  1 x 3 @  3 x 3
    return uvw
   

#Phase shift vis data
def apply_phasor(vis_data,uvw, field_id,freq_chan,phase_rotation,rot_field_id,common_tangent_reprojection,single_precision):

    if common_tangent_reprojection:
        end_slice = 2
    else:
        end_slice = 3
    
    phase_direction = np.zeros(uvw.shape[0:2],np.double) #time x baseline
    
    for i_time in range(uvw.shape[0]):
        field_id_t = field_id[i_time,:,:,:]
        rot_field_indx = np.where(rot_field_id == np.unique(field_id_t[~np.isnan(field_id_t)])[0])[0][0]
        phase_direction[i_time,:] = uvw[i_time,:,0:end_slice,0] @ phase_rotation[rot_field_indx,0:end_slice]
        
    
    n_chan = vis_data.shape[2]
    n_pol = vis_data.shape[3]
    
    #Broadcast top include values for channels
    phase_direction = np.transpose(np.broadcast_to(phase_direction,((n_chan,)+uvw.shape[0:2])), axes=(1,2,0)) #N_time x N_baseline x N_chan
    
    phasor = np.exp(2.0*1j*np.pi*phase_direction*np.broadcast_to(freq_chan[0,0,:,0],uvw.shape[0:2]+(n_chan,))/scipy.constants.c) # phasor_ngcasa = - phasor_casa. Sign flip is due to CASA gridders convention sign flip.
    phasor = np.transpose(np.broadcast_to(phasor,((n_pol,)+vis_data.shape[0:3])), axes=(1,2,3,0))
    vis_data = vis_data*phasor
    
    if single_precision:
        vis_data = (vis_data.astype(np.complex64)).astype(np.complex128)
    
    return vis_data
