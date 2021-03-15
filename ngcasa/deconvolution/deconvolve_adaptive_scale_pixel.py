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

def deconvolve_adaptive_scale_pixel(img_dataset, deconvolve_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    An iterative solver to construct a 2D mixed model from an observed image(set) and psf(set).
    
    Sky Model - A linear combination of 2D Gaussians
    Algorithm - Chi-square / TV minimization on atom parameters, with subspace selections.
       
    Options - Narrow-band, Wide-band
    
    Input - Requires an input cube (mfs is a cube with nchan=1)
    Output - Cube model image  and/or a list of flux components.

    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """

