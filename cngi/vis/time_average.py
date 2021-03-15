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

################################################
def time_average(mxds, vis, bin=1, width=None, span='state', maxuvwdistance=None):
    """
    Average data across the time axis

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    vis : str
        visibility partition in the mxds to use
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
        New output multi-xarray Dataset with global data
    """
    import numpy as np
    from cngi._utils._io import mxds_copier

    xds = mxds.attrs[vis]
    intnan = np.full((1), np.nan, dtype=np.int32)[0]
    
    # drop vars that don't have time so they don't get stacked later on
    notime_vars = [cc for cc in list(xds.data_vars) if 'time' not in xds[cc].dims]
    xds = xds.drop_vars(notime_vars)

    #######
    # mapped out over groups
    def timebin(gxds, stacked=True):
        if stacked: gxds = gxds.unstack('stb')
        
        # mean coarsen/resample everything but data and weight
        dvs = [dv for dv in gxds.data_vars if dv not in ['DATA', 'CORRECTED_DATA', 'DATA_WEIGHT', 'CORRECTED_DATA_WEIGHT']] + list(gxds.coords)
        if width is None:
            nxds = gxds[dvs].coarsen(time=bin, boundary='pad').mean()
        else:
            nxds = gxds[dvs].resample(time=width).mean()
        
        # sum coarsen/resample weight
        for wt in ['DATA_WEIGHT', 'CORRECTED_DATA_WEIGHT']:
            if wt in gxds.data_vars:
                if width is None:
                    nxds[wt] = gxds[wt].coarsen(time=bin, boundary='pad').sum()
                else:
                    nxds[wt] = gxds[wt].resample(time=width).sum()
                
        # use weight in coarsening/resampling data cols
        for col in ['DATA', 'CORRECTED_DATA']:
            if (col in gxds.data_vars) and (col+'_WEIGHT' in gxds.data_vars):
                if width is None:
                    xda = (gxds[col] * gxds[col+'_WEIGHT']).coarsen(time=bin, boundary='pad').sum()
                else:
                    xda = (gxds[col] * gxds[col+'_WEIGHT']).resample(time=width).sum()
                nxds[col] = xda / nxds[col+'_WEIGHT']
        
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

    # put the attributes and dropped data vars back in
    txds = txds.assign_attrs(xds.attrs).assign(dict([(dv, mxds.attrs[vis][dv]) for dv in notime_vars]))

    # verify values
    #cxds1 = xds_state.assign_coords({'time_s': xds_state.time.astype('datetime64[s]')}).swap_dims({'time':'time_s'})
    #cxds2 = txds.assign_coords({'time_s': txds.time.astype('datetime64[s]')}).swap_dims({'time':'time_s'})
    #cxds = cxds1.DATA - cxds2.DATA
    #cxds[51].values

    return mxds_copier(mxds, vis, txds)
