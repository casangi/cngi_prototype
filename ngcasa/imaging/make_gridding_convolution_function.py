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


def make_gridding_convolution_function(img_apeture_dataset, gridding_convolution_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    Calculate gridding convolution functions (GCF) as specified for standard, widefield and mosaic imaging.
    Construct a GCF cache (persistent or on-the-fly)

    Options : Choose a list of effects to include
    
    - PSterm : Prolate-Spheroidal gridding kernel (anti-aliasing function)
    - Aterm : Use PB model and Aperture Illumination Function per antenna to construct a GCF per baseline
        - Include support for Heterogeneous Arrays where Aterm is different per antenna
        - Include support for time-varying PB and AIF models. Rotation, etc.
    - Wterm : FT of Fresnel kernel per baseline
            
    Returns
    -------
    cf_dataset : xarray.core.dataset.Dataset
            
    """

