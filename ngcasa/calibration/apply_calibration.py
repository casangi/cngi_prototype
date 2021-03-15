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

