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

def deconvolve_rotation_measure_clean(img_dataset, deconvolve_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    An iterative solver to construct a full-polarization model from an observed image(set) and psf(set).
   
    Sky Model : Per flux component, delta-functions in lambda-square space
    
    Algorithm :
    
    - Step (1) : Transform the cube to lambda-square space
    - Step (2) : Construct a 3D RM-synthesis PSF
    - Step (3) : Run CLEAN based-deconvolution
    - Step (4) : Transform back to frequency space.
   
    Input : Requires an input cube (mfs is a cube with nchan=1)
    
    Output : Cube model image, Error map (Spectral index map)
    
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """

