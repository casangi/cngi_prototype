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
