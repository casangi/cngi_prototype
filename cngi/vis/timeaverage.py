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
    Average data across the time axis

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

    # function to be mapped out over time groups
    def group_average(gxds):

        # function to be mapped out over variables in a group
        def var_average(xda):
            if ('time' in xda.dims) and (xda.dtype.type != np.str_) and (xda.dtype.type != np.bool_):
                xda = xda.coarsen(time=width, boundary='pad').sum() / width
            elif 'time' in xda.dims:
                xda = xda.coarsen(time=width, boundary='pad').max()
            return xda

        nxds = gxds.map(var_average, keep_attrs=False)

        if ('DATA' in gxds.data_vars) and ('SIGMA_SPECTRUM' in gxds.data_vars):
            xda = (gxds.DATA * gxds.SIGMA_SPECTRUM ** 2).coarsen(time=width, boundary='pad').sum()
            nxds['DATA'] = xda / (gxds.SIGMA_SPECTRUM ** 2).coarsen(time=width, boundary='pad').sum()
        elif ('DATA' in gxds.data_vars) and ('SIGMA' in gxds.data_vars):
            xda = (gxds.DATA * gxds.SIGMA ** 2).coarsen(time=width, boundary='pad').sum()
            nxds['DATA'] = xda / (gxds.SIGMA ** 2).coarsen(time=width, boundary='pad').sum()

        if ('CORRECTED_DATA' in gxds.data_vars) and ('WEIGHT_SPECTRUM' in gxds.data_vars):
            xda = (gxds.CORRECTED_DATA * gxds.WEIGHT_SPECTRUM).coarsen(time=width, boundary='pad').sum()
            nxds['CORRECTED_DATA'] = xda / gxds.WEIGHT_SPECTRUM.coarsen(time=width, boundary='pad').sum()
        elif ('CORRECTED_DATA' in gxds.data_vars) and ('WEIGHT' in gxds.data_vars):
            xda = (gxds.CORRECTED_DATA * gxds.WEIGHT).coarsen(time=width, boundary='pad').sum()
            nxds['CORRECTED_DATA'] = xda / gxds.WEIGHT.coarsen(time=width, boundary='pad').sum()

        return nxds
    ########### end def group_average

    # push time coords in to data_vars and remove non-time coords
    time_coords = [cc for cc in list(xds.coords) if (cc not in xds.dims) and ('time' in xds[cc].dims)]
    notime_coords = [cc for cc in list(xds.coords) if (cc not in xds.dims) and ('time' not in xds[cc].dims)]
    nxds = xds.reset_coords(time_coords)
    nxds = nxds.reset_coords(notime_coords, drop=True)

    if (timespan == 'none') or (timespan is None):
        cgps = [group_average(sgp[1]) for gp in nxds.groupby('scan') for sgp in gp[1].groupby('state')]
    elif timespan == 'scan':  # span across scans by separating out states
        cgps = [group_average(gp[1]) for gp in nxds.groupby('state')]
    elif timespan == 'state':  # span across state by separating out scans
        cgps = [group_average(gp[1]) for gp in nxds.groupby('scan')]
    else:  # span across both
        cgps = [group_average(nxds)]

    txds = xarray.concat(cgps, dim='time')
    txds = xarray.merge([xds[[cc for cc in list(xds.coords) if 'time' not in xds[cc].dims]], txds]).assign_attrs(xds.attrs).set_coords(time_coords)

    # coarsen can change int/bool dtypes to float, so they need to be manually set back
    for dv in txds.data_vars:
        txds[dv] = txds[dv].astype(xds[dv].dtype)

    return txds
