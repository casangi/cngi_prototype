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

##########################
def chan_smooth(mxds, vis, type='triang', size=3, gain=1.0, window=None):
    """
    Apply a smoothing kernel to the channel axis

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    vis : str
        visibility partition in the mxds to use
    type : str or tuple
        type of window function to use: 'boxcar', 'triang', 'hann' etc. Default is 'triang'.  Scipy.signal is used to generate the
        window weights, refer to https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows for a
        complete list of supported windows. If your window choice requires additional parameters, use a tuple e.g. ('exponential', None, 0.6)
    size : int
        width of window (# of channels). Default is 3
    gain : float
        gain factor after convolution. Used to set weights. Default is unity gain (1.0)
    window : list of floats
        user defined window weights to apply (all other options ignored if this is supplied). Default is None
        
    Returns
    -------
    xarray.core.dataset.Dataset
        New output multi-xarray Dataset with global data
    """
    import xarray
    import numpy as np
    from scipy.signal import get_window
    from cngi._utils._io import mxds_copier

    xds = mxds.attrs[vis]
    
    if window is None:
        window = gain * get_window(type, size, False) / (np.sum(get_window(type, size, False)))
    else:
        window = np.atleast_1d(window)
        
    window = xarray.DataArray(window, dims=['window'])
    
    # save names of coordinates, then reset them all to variables
    coords = [cc for cc in list(xds.coords) if cc not in xds.dims]
    new_xds = xds.reset_coords()
    
    # create rolling window view of dataset along channel dimension
    rolling_xds = new_xds.rolling(chan=size, min_periods=1, center=True).construct('window')
    
    for dv in rolling_xds.data_vars:
        xda = rolling_xds.data_vars[dv]
    
        # apply chan smoothing to compatible variables
        if ('window' in xda.dims) and (new_xds[dv].dtype.type != np.str_) and (new_xds[dv].dtype.type != np.bool_):
            new_xds[dv] = xda.dot(window).astype(new_xds[dv].dtype)
        
    # return the appropriate variables to coordinates and stick attributes back in
    new_xds = new_xds.set_coords(coords).assign_attrs(xds.attrs)
    
    return mxds_copier(mxds, vis, new_xds)


