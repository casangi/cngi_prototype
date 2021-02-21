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

def implot(xda, axis=['l','m'], chans=None, pols=None, overplot=False, drawplot=True, tsize=250):
    """
    Plot a preview of Image xarray DataArray contents
    
    Parameters
    ----------
    xda : xarray.core.dataarray.DataArray
        input DataArray
    axis : str or list
        DataArray coordinate(s) to plot against data. Default ['d0', 'd1']. All other coordinates will be averaged
    chans : int or list of ints
        channel axis indices to select prior to averaging
    pols : int or list of ints
        polarization axis indices to select prior to averaging
    overplot : bool
        Overlay new plot on to existing window. Default of False makes a new window for each plot
    drawplot : bool
        Display plot window. Should pretty much always be True unless you want to overlay things
        in a Jupyter notebook.
    tsize : int
        target size of the preview plot (might be smaller). Default is 250 points per axis

    Returns
    -------
      Open matplotlib window
    """
    import matplotlib.pyplot as plt
    import xarray
    import numpy as np
    import warnings
    warnings.simplefilter("ignore", category=RuntimeWarning)  # suppress warnings about nan-slices
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    
    if overplot:
        axes = None
    else:
        fig, axes = plt.subplots(1, 1)
    
    axis = np.atleast_1d(axis)
    if chans is not None:
        xda = xda.isel(chan=chans)
    if pols is not None:
        xda = xda.isel(pol=pols)
    
    # fast decimate to roughly the desired size
    thinf = np.ceil(np.array(xda.shape) / tsize)
    txda = xda.thin(dict([(xda.dims[ii], int(thinf[ii])) for ii in range(len(thinf))]))
    
    # can't plot complex numbers, bools (sometimes), or strings
    if txda.dtype == 'complex128':
        txda = (txda.real**2 + txda.imag**2)**0.5
    elif txda.dtype == 'bool':
        txda = txda.astype(int)
    elif txda.dtype.type is np.str_:
        txda = xarray.DataArray(np.unique(txda, return_inverse=True)[1], dims=txda.dims, coords=txda.coords, name=txda.name)
    
    # no axis - plot against range of data
    # collapse all but first dimension
    if axis[0] is None:
        collapse = [ii for ii in range(1, txda.ndim)]
        if len(collapse) > 0: txda = txda.max(axis=collapse)
        txda[txda.dims[0]] = np.arange(txda.shape[0])
        txda.plot.line(ax=axes, marker='.', linewidth=0.0)
        
    # single axis, coord ndim is 1
    elif (len(axis) == 1) and (txda[axis[0]].ndim == 1):
        collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in txda[axis[0]].dims]
        if len(collapse) > 0: txda = txda.max(axis=collapse)
        txda.plot.line(ax=axes, x=axis[0], marker='.', linewidth=0.0)

    # single axis, coord ndim is 2
    elif (len(axis) == 1) and (txda[axis[0]].ndim == 2):
        collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in txda[axis[0]].dims]
        if len(collapse) > 0: txda = txda.max(axis=collapse)
        txda.plot.pcolormesh(ax=axes, x=axis[0], y=txda.dims[0])
    
    # two axes
    elif (len(axis) == 2):
        collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in (txda[axis[0]].dims + txda[axis[1]].dims)]
        if len(collapse) > 0: txda = txda.max(axis=collapse)
        txda.plot.pcolormesh(ax=axes, x=axis[0], y=axis[1])

    plt.title(txda.name)
    if drawplot:
        plt.show()
