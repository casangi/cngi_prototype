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

################################################
def timeaverage(xds, bin=1, width=None, span='state', maxuvwdistance=None):
    """
    Average data across the time axis

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    bin : int
        number of adjacent times to average, used when width is None. Default=1 (no change)
    width : str
        resample to width freq (i.e. '10s') and produce uniform time steps over span. Ignores bin. Default None uses bin value.
        see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html.
    span : str
        span of the binning. Allowed values are 'scan', 'state' or 'both'.  Default is 'state' (meaning all states in a scan)
    maxuvwdistance (future) : float
        NOT IMPLEMENTED. maximum separation of start-to-end baselines that can be included in an average. (meters)

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset
    """
    import xarray
    import numpy as np

    intnan = np.full((1), np.nan, dtype=np.int32)[0]

    #######
    # mapped out over groups
    def timebin(gxds, stacked=True):
        if stacked: gxds = gxds.unstack('stb')
        
        # mean coarsen/resample everything but data and weight
        dvs = [dv for dv in gxds.data_vars if dv not in ['DATA', 'CORRECTED_DATA', 'WEIGHT']] + list(gxds.coords)
        if width is None:
            nxds = gxds[dvs].coarsen(time=bin, boundary='pad').mean()
        else:
            nxds = gxds[dvs].resample(time=width).mean()
        
        # sum coarsen/resample weight
        if 'WEIGHT' in gxds.data_vars:
            if width is None:
                nxds['WEIGHT'] = gxds.WEIGHT.coarsen(time=bin, boundary='pad').sum()
            else:
                nxds['WEIGHT'] = gxds.WEIGHT.resample(time=width).sum()
                
        # use weight in coarsening/resampling data cols
        for col in ['DATA', 'CORRECTED_DATA']:
            if (col in gxds.data_vars) and ('WEIGHT' in gxds.data_vars):
                if width is None:
                    xda = (gxds[col] * gxds.WEIGHT).coarsen(time=bin, boundary='pad').sum()
                else:
                    xda = (gxds[col] * gxds.WEIGHT).resample(time=width).sum()
                nxds[col] = xda / nxds['WEIGHT']
        
        if stacked: nxds = nxds.stack({'stb': ('time', 'baseline')})
        return nxds

    #############
    # span across state by grouping on scans (keeps scans separate)
    if span == 'state':
        txds = xds.stack({'stb': ('time', 'baseline')})
        txds = txds.groupby('SCAN_NUMBER').map(timebin)
        txds = txds.where(txds.SCAN_NUMBER.notnull() & (txds.SCAN_NUMBER > intnan), drop=True).unstack('stb')
        txds = txds.transpose('time', 'baseline', 'chan', 'pol', 'uvw_index', 'spw_id', 'pol_id')

    # span across scans by grouping on states (keeps states separate)
    elif span == 'scan':
        txds = xds.stack({'stb': ('time', 'baseline')})
        txds = txds.groupby('STATE_ID').map(timebin)
        txds = txds.where(txds.STATE_ID.notnull() & (txds.STATE_ID > intnan), drop=True).unstack('stb')
        txds = txds.transpose('time', 'baseline', 'chan', 'pol', 'uvw_index', 'spw_id', 'pol_id')

    # span across both
    else:
        txds = timebin(xds, stacked=False)

    # coarsen can change int/bool dtypes to float, so they need to be manually set back
    for dv in txds.data_vars:
        txds[dv] = txds[dv].astype(xds[dv].dtype)

    # put the attributes back in
    txds = txds.assign_attrs(xds.attrs)

    # verify values
    #cxds1 = xds_state.assign_coords({'time_s': xds_state.time.astype('datetime64[s]')}).swap_dims({'time':'time_s'})
    #cxds2 = txds.assign_coords({'time_s': txds.time.astype('datetime64[s]')}).swap_dims({'time':'time_s'})
    #cxds = cxds1.DATA - cxds2.DATA
    #cxds[51].values

    return txds
