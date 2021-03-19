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

def _normalize(image, sum_weight, img_dataset, gcf_dataset, direction, norm_parms, sel_parms):
    """
    PB normalization on the cubes

    direction : 'forward''reverse'
    norm_type : 'flatnoise','flatsky','common','pbsquare'

    Multiply and/or divide by PB models, accounting for masks/regions.

    #See https://casa.nrao.edu/casadocs/casa-5.6.0/imaging/synthesis-imaging/data-weighting Normalizationm Steps
    #Standard gridder (only ps term) devide by sum of weight and ps correcting image.
    #https://library.nrao.edu/public/memos/evla/EVLAM_198.pdf
    
    #

    """
    import dask.array as da
    import numpy as np
    import copy
    
    norm_type = norm_parms['norm_type']
    
    def normalize_image(image, sum_weights,  normalizing_image, oversampling, correct_oversampling):
        sum_weights_copy = copy.deepcopy(sum_weights) ##Don't mutate inputs, therefore do deep copy (https://docs.dask.org/en/latest/delayed-best-practices.html).
        sum_weights_copy[sum_weights_copy == 0] = 1
        
        if correct_oversampling:
            image_size = np.array(image.shape)
            image_center = image_size//2
            sincx = np.sinc(np.arange(-image_center[0], image_size[0]-image_center[0])/(image_size[0]*oversampling[0]))
            sincy = np.sinc(np.arange(-image_center[1], image_size[1]-image_center[1])/(image_size[1]*oversampling[1]))
            
            oversampling_correcting_func = np.dot(sincx[:,None],sincy[None,:]) #Last section for sinc correcting function https://library.nrao.edu/public/memos/evla/EVLAM_198.pdf
            
            normalized_image = (image / sum_weights_copy) / (oversampling_correcting_func[:,:,None,None]*normalizing_image)
            
            #print(sum_weights_copy,oversampling_correcting_func[500,360,None,None])
            #normalized_image = (image / sum_weights_copy ) / (oversampling_correcting_func[:,:,None,None])
        else:
            normalized_image = (image / sum_weights_copy) / normalizing_image
        return normalized_image
    
    if direction == 'forward':
        oversampling = gcf_dataset.oversampling
        correct_oversampling = True
        if norm_type == 'flat_noise':
            # Divide the raw image by sqrt(.weight) so that the input to the minor cycle represents the product of the sky and PB. The noise is 'flat' across the region covered by each PB.
            normalizing_image = gcf_dataset.PS_CORR_IMAGE.data[:,:,None,None]*img_dataset[sel_parms['data_group_in']['pb']].data[:,:,0,:,:]
            normalized_image = da.map_blocks(normalize_image, image, sum_weight[None,None,:,:], normalizing_image, oversampling, correct_oversampling, dtype=np.double)
        elif norm_type == 'flat_sky':
            #  Divide the raw image by .weight so that the input to the minor cycle represents only the sky. The noise is higher in the outer regions of the primary beam where the sensitivity is low.
            normalizing_image = gcf_dataset.PS_CORR_IMAGE.data[:,:,None,None]*img_dataset[sel_parms['data_group_in']['weight_pb']].data[:,:,0,:,:]
            
            #print(sel_parms['data_group_in']['weight_pb'])
            #print('$%$%',img_dataset[sel_parms['data_group_in']['weight_pb']].data.compute())
            
            normalized_image = da.map_blocks(normalize_image, image, sum_weight[None,None,:,:], normalizing_image, oversampling, correct_oversampling, dtype=np.double)
            
            #print(normalized_image.compute())
        elif norm_type == 'none':
            print('in normalize none ')
            #No normalization after gridding and FFT. The minor cycle sees the sky times pb square
            normalizing_image = gcf_dataset.PS_CORR_IMAGE.data[:,:,None,None]
            normalized_image = da.map_blocks(normalize_image, image, sum_weight[None,None,:,:], normalizing_image, oversampling, correct_oversampling, dtype=np.double)
            #normalized_image = image
        
        if norm_parms['pb_limit'] > 0:
            normalized_image[img_dataset[sel_parms['data_group_in']['pb']].data[:,:,0,:,:] < norm_parms['pb_limit']] = 0.0
        
        if norm_parms['single_precision']:
            normalized_image = (normalized_image.astype(np.float32)).astype(np.float64)
        
        return normalized_image
    elif direction == 'reverse':
            print('reverse operation not yet implemented not yet implemented')
        
    
    
    
