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

def manual_flag(vis_dataset, flag_parms, storage_parms):
    """
    .. todo::
        This function is not yet implemented
    
    Define a set of data selection queries to mark as flags.
    
    For each query in the input list, set flag=1 for the intersection of
    selections along multiple dimensions.
    
    Inputs :
    
    (1) list of selection queries
    (2) array name for output flags. Default = FLAG
        
    TBD :
    
    How to implement this ? Just a series of vis_dataset.isel(...) calls in sequence ?
    What magic does the framework provide to make this efficient for the several 1000s
    of selections that are typical for online flags.
       
    list_sels : [{'time':[76,77,78], 'chan':[6,7,8,12]}, {'time':[112,113], 'chan':[6,7,56]}]
       
    for sel in list_sels :
    
       new_xds = vis_dataset.isel(**sel)
       new_xds.FLAG = 1   ( or equivalent )
       <save them to original vis_dataset ? >
       
    Other ideas :
    
    -- xarray.ones_like(xds.DATA).where(<some conditions>, other=0)
    
    Returns
    -------
    vis_dataset : xarray.core.dataset.Dataset
    """
