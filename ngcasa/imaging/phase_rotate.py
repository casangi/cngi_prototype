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

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning) #Suppress  NumbaPerformanceWarning: '@' is faster on contiguous arrays warning. This happens for phasor_loop and apply_rotation_matrix functions.

def phase_rotate(vis_dataset, global_dataset, rotation_parms, sel_parms, storage_parms):
    """
    Rotate uvw coordinates and phase rotate visibilities. For a joint mosaics rotation_parms['common_tangent_reprojection'] must be true.
    The specified phasecenter and field phase centers are assumed to be in the same frame.
    East-west arrays, emphemeris objects or objects within the nearfield are not supported.
    
    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input visibility dataset.
    global_dataset : xarray.core.dataset.Dataset
        Input global dataset.
    rotation_parms : dictionary
    rotation_parms['image_phase_center'] : list of number, length = 2, units = radians
       The phase center to rotate to (right ascension and declination).
    rotation_parms['common_tangent_reprojection']  : bool, default = True
       If true common tangent reprojection is used (should be true if a joint mosaic image is being created).
    rotation_parms['single_precision'] : bool, default = True
       If rotation_parms['single_precision'] is true then the output visibilities are cast from 128 bit complex to 64 bit complex. Mathematical operations are always done in double precision.
    sel_parms : dictionary
    sel_parms['uvw_in'] : str, default = 'UVW'
        The uvw data variable to rotate.
    sel_parms['uvw_out'] : str, default = 'UVW_ROT'
        The output uvw data variable (must not be the same as sel_parms['uvw_in']).
    sel_parms['data_in'] : str, default = 'DATA'
        The visbility data variable to phase rotate.
    sel_parms['data_out'] : str, default = 'DATA_ROT'
        The output visibility data variable (must not be the same as sel_parms['data_in']).
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
    psf_dataset : xarray.core.dataset.Dataset
    """
    #based on UVWMachine and FTMachine
    #measures/Measures/UVWMachine.cc
    
    print('######################### Start phase_rotate #########################')
    
    from ngcasa._ngcasa_utils._store import _store
    from scipy.spatial.transform import Rotation as R
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
    def apply_rotation_matrix(uvw, field_id, uvw_rotmat):
        #print(uvw.shape,field_id.shape,uvw_rotmat.shape)
        for i_time in range(uvw.shape[0]):
            #print('The index is',np.where(field_names==field[i_time])[1][0],field_id[i_time,0,0])
            #uvw[i_time,:,0:2] = -uvw[i_time,:,0:2] this gives the same result as casa (in the ftmachines uvw(negateUV(vb)) is used). In ngcasa we don't do this since the uvw definition in the gridder and vis.zarr are the same.
            #uvw[i_time,:,:] = np.matmul(uvw[i_time,:,:],uvw_rotmat[field_id[i_time],:,:])
            uvw[i_time,:,:] = uvw[i_time,:,:] @ uvw_rotmat[field_id[i_time,0,0],:,:]
        return uvw
    
    uvw = da.map_blocks(apply_rotation_matrix,vis_dataset[_sel_parms['uvw_in']].data, vis_dataset.field_id.data[:,None,None],uvw_rotmat,dtype=np.double)

        
    chan_chunk_size = vis_dataset[_sel_parms['data_in']].chunks[2][0]
    freq_chan = da.from_array(vis_dataset.coords['chan'].values, chunks=(chan_chunk_size))
    
    vis_rot = da.map_blocks(apply_phasor,vis_dataset[_sel_parms['data_in']].data,uvw[:,:,:,None], vis_dataset.field_id.data[:,None,None,None],freq_chan[None,None,:,None],phase_rotation,_rotation_parms['common_tangent_reprojection'],_rotation_parms['single_precision'],dtype=np.complex)
    
    vis_dataset[_sel_parms['uvw_out']] =  xr.DataArray(uvw, dims=vis_dataset[_sel_parms['uvw_in']].dims)
    vis_dataset[_sel_parms['data_out']] =  xr.DataArray(vis_rot, dims=vis_dataset[_sel_parms['data_in']].dims)
    
    list_xarray_data_variables = [vis_dataset[_sel_parms['uvw_out']],vis_dataset[_sel_parms['data_out']]]
    return _store(vis_dataset,list_xarray_data_variables,_storage_parms)

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
def apply_phasor(vis_data,uvw, field_id,freq_chan,phase_rotation,common_tangent_reprojection,single_precision):

    if common_tangent_reprojection:
        end_slice = 2
    else:
        end_slice = 3
    
    phase_direction = np.zeros(uvw.shape[0:2],np.double) #N_time x N_baseline
    
    for i_time in range(uvw.shape[0]):
        phase_direction[i_time,:] = uvw[i_time,:,0:end_slice,0] @ phase_rotation[field_id[i_time,0,0,0],0:end_slice]
    
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
