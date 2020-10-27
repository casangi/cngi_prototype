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

def linear_mosaic(img_dataset, img_mosaic):
    """
    .. todo::
        This function is not yet implemented
    
    Construct a linear mosaic as a primary-beam weighted sum of a set of input images.
    Individual images are re-sampled onto a larger image grid and summed.
       
    Assume flat-noise normalization for the inputs.  ( TBD : Or flatsky? )
    
    Output image :  sum( input_images ) / sum ( input_pbs )
    
    TBD :
    
    This requires some sort of merging of img_datasets.  CNGI demo on how to append/add images to an image_set and ensure
    that meta-data are consistent?
    
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """
