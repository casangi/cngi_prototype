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

def visplot(xda, axis=None, overplot=False, drawplot=True, tsize=250):
    """
    Plot a preview of Visibility xarray DataArray contents

    Parameters
    ----------
    xda : xarray.core.dataarray.DataArray
        input DataArray to plot
    axis : str or list or xarray.core.dataarray.DataArray
        Coordinate(s) within the xarray DataArray, or a second xarray DataArray to plot against. Default None uses range.
        All other coordinates will be maxed across dims
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

    # fast decimate to roughly the desired size
    thinf = np.ceil(np.array(xda.shape) / tsize)
    txda = xda.thin(dict([(xda.dims[ii], int(thinf[ii])) for ii in range(len(thinf))]))

    # can't plot complex numbers, bools (sometimes), or strings
    if txda.dtype == 'complex128':
        txda = (txda.real ** 2 + txda.imag ** 2) ** 0.5
    elif txda.dtype == 'bool':
        txda = txda.astype(int)
    elif txda.dtype.type is np.int32:
        txda = txda.where(txda > np.full((1), np.nan, dtype=np.int32)[0])
    elif txda.dtype.type is np.str_:
        txda = xarray.DataArray(np.unique(txda, return_inverse=True)[1], dims=txda.dims, coords=txda.coords, name=txda.name)

    ######################
    # decisions based on supplied axis to plot against
    # no axis - plot against range of data
    # collapse all but first dimension
    if axis is None:
        collapse = [ii for ii in range(1, txda.ndim)]
        if len(collapse) > 0: txda = txda.max(axis=collapse)
        txda[txda.dims[0]] = np.arange(txda.shape[0])
        txda.plot.line(ax=axes, marker='.', linewidth=0.0)

    # another xarray DataArray as axis
    elif type(axis) == xarray.core.dataarray.DataArray:
        txda2 = axis.thin(dict([(xda.dims[ii], int(thinf[ii])) for ii in range(len(thinf))]))
        if txda2.dtype.type is np.int32: txda2 = txda2.where(txda2 > np.full((1), np.nan, dtype=np.int32)[0])
        xarray.Dataset({txda.name: txda, txda2.name: txda2}).plot.scatter(txda.name, txda2.name)
    
    # single axis
    elif len(np.atleast_1d(axis)) == 1:
        axis = np.atleast_1d(axis)[0]
        # coord ndim is 1
        if txda[axis].ndim == 1:
            collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in txda[axis].dims]
            if len(collapse) > 0: txda = txda.max(axis=collapse)
            txda.plot.line(ax=axes, x=axis, marker='.', linewidth=0.0)

        # coord ndim is 2
        elif txda[axis].ndim == 2:
            collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in txda[axis].dims]
            if len(collapse) > 0: txda = txda.max(axis=collapse)
            txda.plot.pcolormesh(ax=axes, x=axis, y=txda.dims[0])

    # two axes
    elif len(axis) == 2:
        collapse = [ii for ii in range(txda.ndim) if txda.dims[ii] not in (txda[axis[0]].dims + txda[axis[1]].dims)]
        if len(collapse) > 0: txda = txda.max(axis=collapse)
        txda.plot.pcolormesh(ax=axes, x=axis[0], y=axis[1])

    plt.title(txda.name)
    if drawplot:
        plt.show()
