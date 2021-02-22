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
"""
this module will be included in the api
"""

########################
def apply_flags(mxds, vis, flags='FLAG'):
    """
    Apply flag variables to other data in Visibility Dataset

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    vis : str
        visibility partition in the mxds to use
    flags : list or str
        data var name or list of names to use as flags. Default 'FLAG' uses the FLAG field
    Returns
    -------
    xarray.core.dataset.Dataset
        output multi-xarray Dataset with global data
    """
    import numpy as np
    from cngi._utils._io import mxds_copier

    xds = mxds.attrs[vis]
    
    flags = np.atleast_1d(flags)
    
    flagged_xds = xds.copy()

    # loop over each flag dimension
    # flag each data var with matching dimensions
    for fv in flags:
        for dv in xds.data_vars:
            if dv == fv: continue   # dont flag the flags
            if flagged_xds[dv].dims == flagged_xds[fv].dims:
                flagged_xds[dv] = flagged_xds[dv].where(flagged_xds[fv] == 0).astype(xds[dv].dtype)
        
    return mxds_copier(mxds, vis, flagged_xds)
