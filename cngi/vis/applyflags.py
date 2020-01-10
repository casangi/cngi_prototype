#   Copyright 2019 AUI, Inc. Washington DC, USA
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


########################
def applyflags(xds, flags=None):
    """
    Apply flag variables to other data in Visibility Dataset

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    flags : list of str
        list of data var names to use as flags. Default None uses all bool types
    Returns
    -------
    xarray.core.dataset.Dataset
        output Visibility Dataset
    """
    import xarray

    new_xds = xarray.Dataset(attrs=xds.attrs)
    
    # find all variables with bool data type to use as flags
    if flags is None:
        flags = [dv for dv in xds.data_vars if xds.data_vars[dv].dtype == 'bool']
    
    # apply flags to each variable of compatible shape
    for dv in xds.data_vars:
        if dv in flags: continue
        
        xda = xds.data_vars[dv]
        for flag in flags:
            if xds.data_vars[flag].ndim <= xda.ndim:
                xda = xda.where(~xds.data_vars[flag])
        
        new_xds = new_xds.assign(dict([(dv, xda)]))
    
    return new_xds
