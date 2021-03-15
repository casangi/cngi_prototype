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
