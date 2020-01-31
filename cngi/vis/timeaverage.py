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
def timeaverage(xds, width=1, timebin=None, timespan='both'):
    """
    Average data across time axis

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    width : int
        number of adjacent times to average (fast), used when timebin is None. Default=1 (no change)
    timebin (future) : float
        time bin width to averaging (in seconds) (slow - requires interpolation). Default None uses width parameter
    timespan (future) : str
        Span of the timebin. Allowed values are 'scan', 'state' or 'both'

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset
    """
    import xarray
    import numpy as np

    # find all variables with time dimension (vwtd)
    vwtds = [dv for dv in xds.data_vars if 'time' in xds[dv].dims]

    # create list of single-scan datasets from this parent dataset
    ssds = [xds]
    if 'scan' in xds.coords:
        ssds = [xds.where(xds.scan == ss, drop=True) for ss in np.unique(xds.scan.values)]

    ndss = []  # list of new datasets
    for ss in ssds:
        xdas = {}
        for dv in ss.data_vars:
            xda = ss.data_vars[dv]

            # apply time averaging to compatible variables
            if dv in vwtds:
                if (dv == 'DATA') and ('SIGMA_SPECTRUM' in vwtds):
                    xda = (xda.DATA * xda.SIGMA_SPECTRUM).rolling(time=width).sum() * xda.SIGMA_SPECTRUM.rolling(time=width).sum()
                elif (dv == 'CORRECTED_DATA') and ('WEIGHT_SPECTRUM' in vwtds):
                    xda = (xda.CORRECTED_DATA * xda.WEIGHT_SPECTRUM).rolling(time=width).sum() * xda.WEIGHT_SPECTRUM.rolling(time=width).sum()
                else:
                    xda = xda.rolling(time=width).mean()
                xdas[dv] = xda

        ndss += [xarray.Dataset(xdas)]

    new_xds = xarray.concat(ndss, dim='time', coords='all')
    new_xds = new_xds.thin({'time': width})
    new_xds.attrs = xds.attrs

    return new_xds
