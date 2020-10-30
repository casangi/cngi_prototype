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


def solve_calibration(vis_dataset, cal_dataset, solve_parms, storage_parm):
    """
    .. todo::
        This function is not yet implemented
    
    Calculate antenna gain solutions according to the parameters in solpars.
    The input dataset has been pre-averaged/processed and the model visibilities exist
    
    Iteratively solve the system of equations g_i g_j* = V_data_ij/V_model_ij  for all ij.
    Construct a separate solution for each timestep and channel in the input dataset.
    
    Options :
    
    amp, phase or both
    solution type (?) G-term, D-term, etc...
    Data array for which to calculate solutions. Default='DATA'
    
    TBD :
    
    Single method with options for solutions of different types ?
    Or, separate methods for G/B, D, P etc.. : solve_B, solve_D, solve_B, etc...
          
    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """

