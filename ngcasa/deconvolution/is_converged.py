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

def is_converged(img_dataset, iterpars, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    An iteration controller for image reconstruction
    
    The current image set (residual, psf, model, etc) is evaluated against stopping criteria derived
    from input parameters (iterpars) and the image set itself.
    
    Step 1 : Derive stopping criteria
    
    - Merge explicit user-parameters in iterpars (niter,threshold,etc..) with criteria that are calculated from
      the imageset (psfsidelobelevel, cyclethreshold, N-sigma-based thresholds, mask-sensitive thresholds)
    - Calculate 'cycleniter' and 'cyclethreshold' to be used in Step 2.
    
    Step 2 : Apply stopping criteria (as an ordered list)
       
    - Peak residual within the mask region for imagename.residual <= threshold
    - Total iters done >= niter
        
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
        An convergence history list of dict is added to the attributes of img_dataset.
    
    """
