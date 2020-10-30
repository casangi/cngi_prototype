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

def feather(img_dataset_lowres, img_dataset_highres):
    """
    .. todo::
        This function is not yet implemented
    
    Feather two images together, based on restoring beam information stored in both.
    
    Output image = iFT( FT(lowres_image) + [1-FT(lowres_beam)] x FT(highres_image) )
    
    TBD : Do this for the entire image_set (psf, image) and updated restoring-beam information as well ?
    
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """
