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


def _check_grid_parms(grid_parms):
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)
    
    if not(_check_parms(grid_parms, 'image_size', [list], list_acceptable_data_types=[np.int], list_len=2)): parms_passed = False
    if not(_check_parms(grid_parms, 'image_center', [list], list_acceptable_data_types=[np.int], list_len=2, default = np.array(grid_parms['image_size'])//2)): parms_passed = False
    if not(_check_parms(grid_parms, 'cell_size', [list], list_acceptable_data_types=[numbers.Number], list_len=2)): parms_passed = False
    if not(_check_parms(grid_parms, 'fft_padding', [numbers.Number], default=1.2,acceptable_range=[1,2])): parms_passed = False
    if not(_check_parms(grid_parms, 'chan_mode', [str], acceptable_data=['cube','continuum'], default='cube')): parms_passed = False
    
    if parms_passed == True:
        grid_parms['image_size'] = np.array(grid_parms['image_size']).astype(int)
        grid_parms['image_size_padded'] = (grid_parms['fft_padding']* grid_parms['image_size']).astype(int)
        grid_parms['image_center'] = np.array(grid_parms['image_center'])
        grid_parms['cell_size'] = arc_sec_to_rad * np.array(grid_parms['cell_size'])
        grid_parms['cell_size'][0] = -grid_parms['cell_size'][0]
    
    return parms_passed
    
    
##################### Function Specific Parms #####################
def _check_gcf_parms(gcf_parms):
    import numbers
    parms_passed = True
    
    if not(_check_parms(gcf_parms, 'function', [str], acceptable_data=['alma_airy','airy'], default='alma_airy')): parms_passed = False
    if not(_check_parms(gcf_parms, 'chan_tolerance_factor', [numbers.Number], default=0.005)): parms_passed = False
    if not(_check_parms(gcf_parms, 'oversampling', [list], list_acceptable_data_types=[np.int], list_len=2)): parms_passed = False
    if not(_check_parms(gcf_parms, 'max_support', [list], list_acceptable_data_types=[np.int], list_len=2)): parms_passed = False
    if not(_check_parms(gcf_parms, 'image_phase_center', [list], list_acceptable_data_types=[numbers.Number], list_len=2)): parms_passed = False
    
    if gcf_parms['function'] == 'airy':
        if not(_check_parms(gcf_parms, 'list_dish_diameters', [list],list_acceptable_data_types=[numbers.Number],list_len=-1)): parms_passed = False
        if not(_check_parms(gcf_parms, 'list_blockage_diameters', [list],list_acceptable_data_types=[numbers.Number],list_len=-1)): parms_passed = False
    
        if len(gcf_parms['list_dish_diameters']) != len(gcf_parms['list_blockage_diameters']):
            print('######### ERROR:Parameter ', 'list_dish_diameters and list_blockage_diameters must be the same length.')
            parms_passed = False
    
    
    
    if parms_passed == True:
        gcf_parms['oversampling'] = np.array(gcf_parms['oversampling']).astype(int)
        gcf_parms['max_support'] = np.array(gcf_parms['max_support']).astype(int)
        gcf_parms['image_phase_center'] =  np.array(gcf_parms['image_phase_center'])
        
    return parms_passed
    
def _check_mosaic_pb_parms(pb_mosaic_parms):
    parms_passed = True
    
    if not(_check_parms(pb_mosaic_parms, 'pb_name', [str], default='PB')): parms_passed = False
    
    if not(_check_parms(pb_mosaic_parms, 'weight_name', [str], default='WEIGHT_PB')): parms_passed = False
    
    return parms_passed
    
    
def _check_rotation_parms(rotation_parms):
    import numbers
    parms_passed = True
    
    if not(_check_parms(rotation_parms, 'image_phase_center', [list], list_acceptable_data_types=[numbers.Number], list_len=2)): parms_passed = False
    
    if not(_check_parms(rotation_parms, 'common_tangent_reprojection', [bool], default=True)): parms_passed = False
    
    return parms_passed
    
    
    
def _check_norm_parms(norm_parms):
    import numbers
    parms_passed = True
    
    if not(_check_parms(norm_parms, 'norm_type', [str], default='flat_noise', acceptable_data=['flat_noise','flat_sky','pb_square'])): parms_passed = False
    
    return parms_passed
    
    
def _check_imaging_weights_parms(imaging_weights_parms):
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)
    
    '''
    if not(_check_parms(imaging_weights_parms, 'data_name', [str], default='DATA')): parms_passed = False
    if not(_check_dataset(vis_dataset,imaging_weights_parms['data_name'])): parms_passed = False
    
    if not(_check_parms(imaging_weights_parms, 'uvw_name', [str], default='UVW')): parms_passed = False
    if not(_check_dataset(vis_dataset,imaging_weights_parms['uvw_name'])): parms_passed = False
    
    if not(_check_parms(imaging_weights_parms, 'imaging_weight_name', [str], default='IMAGING_WEIGHT')): parms_passed = False
    '''

    if not(_check_parms(imaging_weights_parms, 'weighting', [str], acceptable_data=['natural','uniform','briggs','briggs_abs'], default='natural')): parms_passed = False
    
    if imaging_weights_parms['weighting'] == 'briggs_abs':
        if not(_check_parms(imaging_weights_parms, 'briggs_abs_noise', [numbers.Number], default=1.0)): parms_passed = False

    if (imaging_weights_parms['weighting'] == 'briggs') or (imaging_weights_parms['weighting'] == 'briggs_abs'):
        if not(_check_parms(imaging_weights_parms, 'robust', [numbers.Number], default=0.5, acceptable_range=[-2,2])): parms_passed = False
        
    return parms_passed

def _check_pb_parms(img_dataset, pb_parms):
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)
    
    if not(_check_parms(pb_parms, 'pb_name', [str], default='PB')): parms_passed = False
    
    if not(_check_parms(pb_parms, 'function', [str], default='alma_airy')): parms_passed = False
    
    if not(_check_parms(pb_parms, 'list_dish_diameters', [list],list_acceptable_data_types=[numbers.Number],list_len=-1)): parms_passed = False
    if not(_check_parms(pb_parms, 'list_blockage_diameters', [list],list_acceptable_data_types=[numbers.Number],list_len=-1)): parms_passed = False
    
    if len(pb_parms['list_dish_diameters']) != len(pb_parms['list_blockage_diameters']):
            print('######### ERROR:Parameter ', 'list_dish_diameters and list_blockage_diameters must be the same length.')
            parms_passed = False
        
    return parms_passed
    
'''
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
    
    #if not(_check_parms(grid_parms, 'oversampling', [np.int], default=100)): parms_passed = False
    if not(_check_parms(grid_parms, 'oversampling', [list], list_acceptable_data_types=[np.int], list_len=2)): parms_passed = False
    
    if not(_check_parms(grid_parms, 'support', [np.int], default=7)): parms_passed = False
    
    if not(_check_parms(grid_parms, 'fft_padding', [numbers.Number], default=1.2,acceptable_range=[1,100])): parms_passed = False
    
    if parms_passed == True:
        grid_parms['oversampling'] = np.array(grid_parms['oversampling']).astype(int)
    
        grid_parms['imsize'] = np.array(grid_parms['imsize']).astype(int)
        grid_parms['imsize_padded'] = (grid_parms['fft_padding']* grid_parms['imsize']).astype(int)

        grid_parms['cell'] = arc_sec_to_rad * np.array(grid_parms['cell'])
        grid_parms['cell'][0] = -grid_parms['cell'][0]
        
        grid_parms['complex_grid'] = True
    
    return parms_passed
'''


