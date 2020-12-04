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

def shadow(vis_dataset, shadow_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    Flag all baselines for antennas that are shadowed beyond the specified tolerance.
    
    All antennas in the zarr-file metadata (and their corresponding diameters)
    will be considered for shadow-flag calculations.
    For a given timestep, an antenna is flagged if any of its baselines
    (projected onto the uv-plane) is shorter than  radius_1 + radius_2 - tolerance.
    The value of 'w' is used to determine which antenna is behind the other.
    The phase-reference center is used for antenna-pointing direction.
    
    Antennas that are not part of the observation, but still physically
    present and shadowing other antennas that are being used, must be added
    to the meta-data list in the zarr prior to calling this method.
    
    Inputs :
    
    (1) shadowlimit or tolerance (in m)
    (2) array name for output flags. Default = FLAG
        
    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """

