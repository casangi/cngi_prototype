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
def timeaverage(xds, width=1, timespan='state', timebin=None):
    """
    Average data across time axis

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    width : int
        number of adjacent times to average (fast), used when timebin is None. Default=1 (no change)
    timespan : str
        Span of the timebin. Allowed values are 'none', 'scan', 'state' or 'both'.  Default is 'state' (meaning all states in a scan)
    timebin (future) : float
        time bin width to averaging (in seconds) (slow - requires interpolation). Default None uses width parameter

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset
    """
    import xarray
    import numpy as np

    # find all variables with time dimension (vwtd)
    vwtds = [dv for dv in xds.data_vars if 'time' in xds[dv].dims]

    # create list of single-span datasets from this parent dataset
    ssds = []
    if timespan == 'none':
        scans = [xds.where(xds.scan == ss, drop=True) for ss in np.unique(xds.scan.values)]
        for scan in scans:
            ssds += [scan.where(scan.state == ss, drop=True) for ss in np.unique(scan.state.values)]
    elif timespan == 'scan':  # span across scans by separating out states
        ssds = [xds.where(xds.state == ss, drop=True) for ss in np.unique(xds.state.values)]
    elif timespan == 'state':  # span across state by separating out scans
        ssds = [xds.where(xds.scan == ss, drop=True) for ss in np.unique(xds.scan.values)]
    else:  # span across both
        ssds = [xds]

    ndss = []  # list of new datasets
    for ss in ssds:  # loop over each single-span dataset and average within that span
        xdas = {}
        for dv in ss.data_vars:
            xda = ss.data_vars[dv]

            # apply time averaging to compatible variables
            if dv in vwtds:
                if (dv == 'DATA') and ('SIGMA_SPECTRUM' in ss.data_vars):
                    weight_spectrum = 1.0 / ss.SIGMA_SPECTRUM**2
                    xda = (ss.DATA * weight_spectrum).rolling(time=width, min_periods=1, center=True).sum() / \
                          weight_spectrum.rolling(time=width, min_periods=1, center=True).sum()
                elif (dv == 'CORRECTED_DATA') and ('WEIGHT_SPECTRUM' in ss.data_vars):
                    xda = (ss.CORRECTED_DATA * ss.WEIGHT_SPECTRUM).rolling(time=width, min_periods=1, center=True).sum() / \
                          ss.WEIGHT_SPECTRUM.rolling(time=width, min_periods=1, center=True).sum()
                elif (dv == 'DATA') and ('SIGMA' in ss.data_vars):
                    weight = 1.0 / ss.SIGMA**2
                    xda = (ss.DATA * weight).rolling(time=width, min_periods=1, center=True).sum() / \
                          weight.rolling(time=width, min_periods=1, center=True).sum()
                elif (dv == 'CORRECTED_DATA') and ('WEIGHT' in ss.data_vars):
                    xda = (ss.CORRECTED_DATA * ss.WEIGHT).rolling(time=width, min_periods=1, center=True).sum() / \
                          ss.WEIGHT.rolling(time=width, min_periods=1, center=True).sum()
                else:
                    xda = xda.rolling(time=width, min_periods=1, center=True).mean()
                xdas[dv] = xda.astype(ss.data_vars[dv].dtype)

        ndss += [xarray.Dataset(xdas)]

    new_xds = xarray.concat(ndss, dim='time', coords='all')
    new_xds = new_xds.thin({'time': width})
    new_xds.attrs = xds.attrs

    return new_xds
