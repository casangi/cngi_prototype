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

def predict_modelvis_component(img_dataset, vis_dataset, component_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
        
    Predict model visibilities from a component list by analytical evaluation
    
    Apply PB models to the components prior to evaluation.
    
    Save the model visibilities in arr_name (default = 'MODEL')
    
    Optionally overwrite the model or add to existing model (incremental=T)

    """
