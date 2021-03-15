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

########################
def rebin(xds, factor=1, axis='chan'):
    """
    Rebin an n-dimensional image across any single (spatial or spectral) axis
    
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image Dataset
    factor : int
        scaling factor for binning, Default=1 (no change)
    axis : str
        dataset dimension upon which to rebin ('d0', 'd1', 'chan', 'pol'). Default is 'chan'
    
    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    import numpy as np
    
    # .mean() produces runtimewarning errors (still works though), using .sum() / width is cleaner
    new_xds = xds.coarsen({axis:factor}, boundary='trim').sum() / factor
    new_xds = new_xds.assign_attrs(xds.attrs)
    
    # integer and bool variables are set to floats after coarsen, reset them back now
    dns = [dn for dn in xds.data_vars if xds[dn].dtype.type in [np.int_, np.bool_]]
    for dn in dns:
        new_xds[dn] = new_xds[dn].astype(xds[dn].dtype)
    
    return new_xds



