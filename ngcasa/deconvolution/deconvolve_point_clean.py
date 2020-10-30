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

def deconvolve_point_clean(img_dataset, deconvolve_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    An iterative solver to construct a model from an observed image(set) and psf(set).
    
    Sky model : Point source
    
    Algorithm : CLEAN (a greedy algorithm for chi-square minimization)
       
    Options : Hogbom, Clark
    
    Input : Requires an input cube (mfs is a cube with nchan=1)
    
    Output : Cube model image
    
    Returns
    -------
    img_dataset : xarray.core.dataset.Dataset
    """
