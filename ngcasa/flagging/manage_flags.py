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

def manage_flags(vis_dataset, flag_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    A flag manager that applies Boolean logic operators to
    
    - merge multiple flag versions into one.
    
    - delete flag versions
    
    - list existing version names
    
    Inputs :
    
    (1) List of input flag array names
    (2) array name for output flags. Default = FLAG
    (3) Operation (AND, OR)  TBD : supply the expression, instead of canned options.
    
    If an existing flag array is named as the output, the contents are over-written.
    
    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """

