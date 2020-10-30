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

