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
def chan_average(mxds, vis, width=1):
    """
    Average data across the channel axis

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    vis : str
        visibility partition in the mxds to use
    width : int
        number of adjacent channels to average. Default=1 (no change)

    Returns
    -------
    xarray.core.dataset.Dataset
        New output multi-xarray Dataset with global data
    """
    from cngi._utils._io import mxds_copier
    
    xds = mxds.attrs[vis]
    
    # save names of coordinates, then reset them all to variables
    coords = [cc for cc in list(xds.coords) if cc not in xds.dims]
    xds = xds.reset_coords()

    # use remaining non-chan coordinates and attributes to initialize new return xds
    new_xds = xds[[cc for cc in list(xds.coords) if cc not in ['chan']]]
    
    for dv in xds.data_vars:
        xda = xds.data_vars[dv]
        
        # apply chan averaging to compatible variables
        if 'chan' in xda.dims:
            if (dv == 'DATA') and ('SIGMA_SPECTRUM' in xds.data_vars):
                xda = (xds.DATA / xds.SIGMA_SPECTRUM**2).coarsen(chan=width, boundary='trim').sum()
                xda = xda * (xds.SIGMA_SPECTRUM**2).coarsen(chan=width, boundary='trim').sum()
            elif (dv == 'CORRECTED_DATA') and ('WEIGHT_SPECTRUM' in xds.data_vars):
                xda = (xds.CORRECTED_DATA * xds.WEIGHT_SPECTRUM).coarsen(chan=width, boundary='trim').sum()
                xda = xda / xds.WEIGHT_SPECTRUM.coarsen(chan=width, boundary='trim').sum()
            else:
                # .mean() produces runtimewarning errors (still works though), using .sum() / width is cleaner
                xda = (xda.coarsen(chan=width, boundary='trim').sum() / width).astype(xds.data_vars[dv].dtype)
        
        new_xds = new_xds.assign(dict([(dv,xda)]))

    # return the appropriate variables to coordinates
    new_xds = new_xds.set_coords(coords)

    return mxds_copier(mxds, vis, new_xds)
