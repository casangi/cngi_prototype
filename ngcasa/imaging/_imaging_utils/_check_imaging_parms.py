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

#ducting - code is complex and might fail after some time if parameters is wrong. Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has 14 parameters.

import numpy as np
from ngcasa._ngcasa_utils._check_parms import _check_parms, _check_dataset, _check_storage_parms

def _check_grid_params(vis_dataset, grid_parms, default_image_name='DIRTY_IMAGE', default_sum_weight_name='SUM_WEIGHT'):
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    n_chunks_in_each_dim = vis_dataset.DATA.data.numblocks
    
    if n_chunks_in_each_dim[3] != 1:
        print('######### ERROR chunking along polarization is not supported')
        return False
    
    if not(_check_parms(grid_parms, 'data_name', [str], default='DATA')): parms_passed = False
    if not(_check_dataset(vis_dataset,grid_parms['data_name'])): parms_passed = False
    
    if not(_check_parms(grid_parms, 'uvw_name', [str], default='UVW')): parms_passed = False
    if not(_check_dataset(vis_dataset,grid_parms['uvw_name'])): parms_passed = False
    
    if not(_check_parms(grid_parms, 'imaging_weight_name', [str], default='IMAGING_WEIGHT')): parms_passed = False
    if not(_check_dataset(vis_dataset,grid_parms['imaging_weight_name'])): parms_passed = False
    
    if not(_check_parms(grid_parms, 'image_name', [str], default=default_image_name)): parms_passed = False
    
    if not(_check_parms(grid_parms, 'sum_weight_name', [str], default=default_sum_weight_name)): parms_passed = False
    
    if not(_check_parms(grid_parms, 'chan_mode', [str], acceptable_data=['cube','continuum'], default='cube')): parms_passed = False
    
    if not(_check_parms(grid_parms, 'imsize', [list], list_acceptable_data_types=[np.int], list_len=2)): parms_passed = False
    
    if not(_check_parms(grid_parms, 'cell', [list], list_acceptable_data_types=[numbers.Number], list_len=2)): parms_passed = False
    
    if not(_check_parms(grid_parms, 'oversampling', [np.int], default=100)): parms_passed = False
    
    if not(_check_parms(grid_parms, 'support', [np.int], default=7)): parms_passed = False
    
    if not(_check_parms(grid_parms, 'fft_padding', [numbers.Number], default=1.2,acceptable_range=[1,100])): parms_passed = False
    
    if parms_passed == True:
        grid_parms['imsize'] = np.array(grid_parms['imsize']).astype(int)
        grid_parms['imsize_padded'] = (grid_parms['fft_padding']* grid_parms['imsize']).astype(int)

        grid_parms['cell'] = arc_sec_to_rad * np.array(grid_parms['cell'])
        grid_parms['cell'][0] = -grid_parms['cell'][0]
        
        grid_parms['complex_grid'] = True
    
    return parms_passed

#########################################################################################################################################################################################
def _check_imaging_weights_parms(vis_dataset, imaging_weights_parms):
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)
    
    if not(_check_parms(imaging_weights_parms, 'data_name', [str], default='DATA')): parms_passed = False
    if not(_check_dataset(vis_dataset,imaging_weights_parms['data_name'])): parms_passed = False
    
    if not(_check_parms(imaging_weights_parms, 'uvw_name', [str], default='UVW')): parms_passed = False
    if not(_check_dataset(vis_dataset,imaging_weights_parms['uvw_name'])): parms_passed = False
    
    if not(_check_parms(imaging_weights_parms, 'imaging_weight_name', [str], default='IMAGING_WEIGHT')): parms_passed = False

    if not(_check_parms(imaging_weights_parms, 'weighting', [str], acceptable_data=['natural','uniform','briggs','briggs_abs'], default='natural')): parms_passed = False
    
    if imaging_weights_parms['weighting'] == 'briggs_abs':
        if not(_check_parms(imaging_weights_parms, 'briggs_abs_noise', [numbers.Number], default=1.0)): parms_passed = False

    if not(_check_parms(imaging_weights_parms, 'robust', [numbers.Number], default=0.5, acceptable_range=[-2,2])): parms_passed = False
    
    if imaging_weights_parms['weighting'] != 'natural':
        if not(_check_parms(imaging_weights_parms, 'chan_mode', [str], acceptable_data=['cube','continuum'], default='cube')): parms_passed = False
        if not(_check_parms(imaging_weights_parms, 'imsize', [list], list_acceptable_data_types=[int,np.int64], list_len=2)): parms_passed = False
        if not(_check_parms(imaging_weights_parms, 'cell', [list], list_acceptable_data_types=[numbers.Number], list_len=2)): parms_passed = False

    if parms_passed == True:
        imaging_weights_parms['imsize'] = np.array(imaging_weights_parms['imsize']).astype(int)

        imaging_weights_parms['cell'] = arc_sec_to_rad * np.array(imaging_weights_parms['cell'])
        imaging_weights_parms['cell'][0] = -imaging_weights_parms['cell'][0]
        
    return parms_passed

#########################################################################################################################################################################################
def _check_pb_parms(img_dataset, pb_parms):
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)
    
    if not(_check_parms(pb_parms, 'pb_name', [str], default='PB')): parms_passed = False
    
    if not(_check_parms(pb_parms, 'function', [str], default='airy')): parms_passed = False
    
    if pb_parms['function'] == 'airy':
        if not(_check_parms(pb_parms, 'list_dish_diameters', [list],list_acceptable_data_types=[numbers.Number],list_len=-1)): parms_passed = False
        if not(_check_parms(pb_parms, 'list_blockage_diameters', [list],list_acceptable_data_types=[numbers.Number],list_len=-1)): parms_passed = False
    
        if len(pb_parms['list_dish_diameters']) != len(pb_parms['list_blockage_diameters']):
            print('######### ERROR:Parameter ', 'list_dish_diameters and list_blockage_diameters must be the same length.')
            parms_passed = False
    
    if not(_check_parms(pb_parms, 'imsize', [list], list_acceptable_data_types=[int,np.int64], list_len=2)): parms_passed = False
    if not(_check_parms(pb_parms, 'cell', [list], list_acceptable_data_types=[numbers.Number], list_len=2)): parms_passed = False
    if not(_check_parms(pb_parms, 'pb_limit', [numbers.Number], acceptable_range=[0,1], default=0.0)): parms_passed = False

    if parms_passed == True:
        pb_parms['imsize'] = np.array(pb_parms['imsize']).astype(int)
        pb_parms['cell'] = arc_sec_to_rad * np.array(pb_parms['cell'])
        pb_parms['cell'][0] = -pb_parms['cell'][0]
        
    return parms_passed
