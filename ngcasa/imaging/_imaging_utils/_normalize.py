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

def _normalize(image, sum_weight, img_dataset, gcf_dataset, direction, normtype, sel_parms):
    """
    PB normalization on the cubes

    direction : 'forward''reverse'
    normtype : 'flatnoise','flatsky','common','pbsquare'

    Multiply and/or divide by PB models, accounting for masks/regions.

    #See https://casa.nrao.edu/casadocs/casa-5.6.0/imaging/synthesis-imaging/data-weighting Normalizationm Steps
    #Standard gridder (only ps term) devide by sum of weight and ps correcting image.
    
    #

    """
    import dask.array as da
    import numpy as np
    import copy
    
    def normalize_image(image, sum_weights,  normalizing_image):
        sum_weights_copy = copy.deepcopy(sum_weights) ##Don't mutate inputs, therefore do deep copy (https://docs.dask.org/en/latest/delayed-best-practices.html).
        sum_weights_copy[sum_weights_copy == 0] = 1
        normalized_image = (image / sum_weights_copy) / normalizing_image
        return normalized_image
    
    if direction == 'forward':
        
        if normtype == 'flat_noise':
            # Divide the raw image by sqrt(.weight) so that the input to the minor cycle represents the product of the sky and PB. The noise is 'flat' across the region covered by each PB.
            normalizing_image = gcf_dataset.PS_CORR_IMAGE.data[:,:,None,None]*img_dataset[sel_parms['pb']].data
            normalized_image = da.map_blocks(normalize_image, image, sum_weight[None,None,:,:], normalizing_image, dtype=np.double)
        elif normtype == 'flat_sky':
            #  Divide the raw image by .weight so that the input to the minor cycle represents only the sky. The noise is higher in the outer regions of the primary beam where the sensitivity is low.
            normalizing_image = gcf_dataset.PS_CORR_IMAGE.data[:,:,None,None]*img_dataset[sel_parms['weight_pb']].data
            normalized_image = da.map_blocks(normalize_image, image, sum_weight[None,None,:,:], normalizing_image, dtype=np.double)
        elif normtype == 'pb_square':
            #No normalization after gridding and FFT. The minor cycle sees the sky times pb square
            normalizing_image = gcf_dataset.PS_CORR_IMAGE.data[:,:,None,None]
            normalized_image = da.map_blocks(normalize_image, image, sum_weight[None,None,:,:], normalizing_image, dtype=np.double)
        
        
        normalized_image[img_dataset[sel_parms['pb']].data < 0.2] = 0.0
        
        return normalized_image
    elif direction == 'reverse':
            print('reverse operation not yet implemented not yet implemented')
        
    
    print(direction,normtype)
    
    
