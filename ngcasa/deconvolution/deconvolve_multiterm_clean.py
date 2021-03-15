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

def deconvolve_multiterm_clean(img_dataset, deconvolve_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    An iterative solver to construct a model from an observed image(set) and psf(set).
    
    Sky model - A (multi-term) linear combination of basis functions.
    
    Multi-scale : Basis functions are inverted tapered paraboloids
    
    Multi-scale MFS : Basis functions are Taylor polynomials in frequency
    
    Options :
    
    - MS-Clean - Multi-scale CLEAN ( MS-MFS Clean with nterms=1 )
        - Input - Requires an input cube (mfs is a cube with nchan=1)
        - Output - Cube model image
                  
    - MS-MFS Clean - Wideband Imaging that solves for a set of Taylor coefficient maps.
        - Input - Multi-channel cube.
        - Output : Taylor coefficient maps, Spectral Index + Evaluation of the model to a Cube model image
                  
    Step (1) cngi.image.cube_to_mfs()
    
    Step (2) Implement the multi-term deconvolution algorithm
    
    Step (3) cngi.image.mfs_to_cube()
     
    The special case of nscales=1 and nterms=1 is the same use-case as deconvolve_point_clean.
    
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """

