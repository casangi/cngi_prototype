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



