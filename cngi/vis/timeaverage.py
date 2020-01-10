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


################################################
def timeaverage(xds, timebin=0.0, timespan=None, maxuvdistance=0.0):
    """
    .. todo::
        This function is not yet implemented

    Average data across time axis

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    timebin : float
        Bin width for time averaging (in seconds). Default 0.0
    timespan : str
        Span the timebin. Allowed values are None, 'scan', 'state' or 'both'

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset
    """
    import xarray

    new_xds = xarray.Dataset(attrs=xds.attrs)

    # find all variables with time dimension (vwtd)
    vwtds = [dv for dv in xds.data_vars if 'chan' in xds[dv].dims]
    
    for dv in xds.data_vars:
        xda = xds.data_vars[dv]
    
        # apply chan averaging to compatible variables
        #if dv in vwtds:
        #    xda = xda.coarsen(chan=width, boundary='trim').mean().astype(xda.dtype)
    
        new_xds = new_xds.assign(dict([(dv, xda)]))

    return new_xds
