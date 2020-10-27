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

def predict_modelvis_image(img_dataset, vis_dataset, grid_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
        
    Predict model visibilities from an input model image cube (units Jy/pixel) using a pre-specified gridding convolution function cache.
    
    Save the model visibilities in arr_name (default = 'MODEL')
    
    Optionally overwrite the model or add to existing model (incremental=T)
    
    (A input cube with 1 channel is a continuum image (nterms=1))

    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """
    #ngcasa.imaging._normalize(direction='forward') # Apply PB models to go to flat-sky
    #cngi.image.stokes_to_corr()
    #cngi.image.fourier_transform()
    #ngcasa.imaging._degrid()
