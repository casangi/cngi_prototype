#   Copyright 2020 AUI, Inc. Washington DC, USA
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


##########################
def chansmooth(xds, type='triang', size=3, gain=1.0, window=None):
    """
    Apply a smoothing kernel to the channel axis

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
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
        New Visibility Dataset with updated data
    """
    import xarray
    import numpy as np
    from scipy.signal import get_window
    
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
    
    return new_xds


