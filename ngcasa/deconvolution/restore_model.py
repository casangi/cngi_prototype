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

def restore_model(img_dataset, restore_parms):
    """
    .. todo::
        This function is not yet implemented
        
    Restore a deconvolved model.
    
    Inputs - target resolution could be native or 'common' or explicitly specified.
    
    Cube and single-term imaging :
    
    - Smooth the model image (Jy/pixel) to the target resolution
    - Smooth the residual image (Jy/beam) to the target resolution
    - Add the two smoothed images
    
    Multi-term imaging :
    
    - Smooth the model taylor coefficient images to the target resolution
    - Apply the inverse Hessian to the residual image vector (data-space to model-space)
      (At non-native target resolution, also compute a new Hessian matched to the scale of the restoring beam.)
    - Smooth the model-space residuals to the target resolution
    
    Re-restoration may be done simply by calling this same method again with a different target resolution.
    Calculations will start with the native model and residual images.
    Note that re-restoration with cngi.image.imsmooth() will not be accurate for multi-term imaging.
    
    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """
