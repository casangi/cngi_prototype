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


def apply_calibration(vis_dataset, cal_dataset, apply_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    Apply antenna gain solutions according to the parameters in solpars.
    
    Calculate  V_ij(corrected) = V_ij(observed) / g_i g_j*
    
    Inputs :
    
    List of calibration solution datasets (to apply in the specified order)
    Interpolation type ?
    Data array on which to operate. Default='DATA'
    Data array in which to fill the output.  Default='CORRECTED_DATA'
    - this option exists in order to support simulator's corrupt operation where we'd pick 'DATA'
    
    TBD : Should this translation of the caltable back to the original un-averaged data be done here,
    or in a centralized CNGI method that handles such interpolations for all methods that need
    to convert averaged values into un-averaged values.  Note the difference between just copying
    and expanding values, versus interpolation.
          
    TBD : Single apply() method, or several ?
    
    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """

