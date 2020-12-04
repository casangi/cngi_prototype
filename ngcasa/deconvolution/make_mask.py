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

def make_mask(img_dataset,mask_parms,storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    Make a region to identify a mask for use in deconvolution.
    
    One or more of the following options are allowed
    
    - Supply a mask in the form of a cngi.image.region
    - Run an auto-masking algorithm to detect structure and define a cngi.image.region
    - Apply a pblimit based mask

    An existing deconvolution mask from img_dataset may either be included in the above, or ignored.

    The output is a region (array?) in the img_dataset containing the intersection of all above regions

    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """
