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
