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

    # save names of coordinates, then reset them all to variables
    coords = [cc for cc in list(xds.coords) if cc not in xds.dims]
    xds = xds.reset_coords()

    # find all variables with time dimension (vwtd)
    vwtds = [dv for dv in xds.data_vars if 'time' in xds[dv].dims]

    # find all remaining coordinates and variables without a time dimension
    remaining = [dv for dv in list(xds.data_vars) + list(xds.coords) if 'time' not in xds[dv].dims]

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

    # loop over each single-span dataset and average within that span
    # build up a list of new averaged single span datasets
    # only time-dependent variables are included here, non-time variables will be added back later
    ndss = []  # list of new datasets
    for ss in ssds:
        xdas = {}
        for dv in ss.data_vars:
            xda = ss.data_vars[dv].astype(xds[dv].dtype)

            # apply time averaging to compatible variables
            if (dv in vwtds) and (xda.dtype.type != np.str_):
                if (dv == 'DATA') and ('SIGMA_SPECTRUM' in ss.data_vars):
                    xda = (ss.DATA / ss.SIGMA_SPECTRUM**2).coarsen(time=width, boundary='trim').sum()
                    xdas[dv] = xda * (ss.SIGMA_SPECTRUM**2).coarsen(time=width, boundary='trim').sum()
                elif (dv == 'CORRECTED_DATA') and ('WEIGHT_SPECTRUM' in ss.data_vars):
                    xda = (ss.CORRECTED_DATA * ss.WEIGHT_SPECTRUM).coarsen(time=width, boundary='trim').sum()
                    xdas[dv] = xda / ss.WEIGHT_SPECTRUM.coarsen(time=width, boundary='trim').sum()
                elif (dv == 'DATA') and ('SIGMA' in ss.data_vars):
                    xda = (ss.DATA / ss.SIGMA**2).coarsen(time=width, boundary='trim').sum()
                    xdas[dv] = xda * (ss.SIGMA**2).coarsen(time=width, boundary='trim').sum()
                elif (dv == 'CORRECTED_DATA') and ('WEIGHT' in ss.data_vars):
                    xda = (ss.CORRECTED_DATA * ss.WEIGHT).coarsen(time=width, boundary='trim').sum()
                    xdas[dv] = xda / ss.WEIGHT.coarsen(time=width, boundary='trim').sum()
                else:
                    xdas[dv] = (xda.coarsen(time=width, boundary='trim').sum() / width).astype(ss.data_vars[dv].dtype)

            # decimate variables with string types
            elif dv in vwtds:
                xdas[dv] = xda.thin(width)

        ndss += [xarray.Dataset(xdas)]

    # concatenate back to a single dataset of all scans/states
    # then merge with a dataset of non-time dependent variables
    new_xds = xarray.concat(ndss, dim='time', coords='all')
    new_xds = xarray.merge([new_xds, xds[remaining]])
    new_xds = new_xds.assign_attrs(xds.attrs)
    new_xds = new_xds.set_coords(coords)

    return new_xds
