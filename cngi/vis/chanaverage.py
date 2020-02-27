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


########################
def chanaverage(xds, width=1):
    """
    Average data across channels

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    width : int
        number of adjacent channels to average. Default=1 (no change)

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset
    """
    import xarray
    
    new_xds = xarray.Dataset(attrs=xds.attrs)
    
    # find all variables with chan dimension (vwcd)
    vwcds = [dv for dv in xds.data_vars if 'chan' in xds[dv].dims]
    
    for dv in xds.data_vars:
        xda = xds.data_vars[dv]
        
        # apply chan averaging to compatible variables
        if dv in vwcds:
            if (dv == 'DATA') and ('SIGMA_SPECTRUM' in vwcds):
                weight_spectrum = 1.0/xds.SIGMA_SPECTRUM**2
                xda = (xds.DATA*weight_spectrum).rolling(chan=width, min_periods=1, center=True).sum() / \
                      weight_spectrum.rolling(chan=width, min_periods=1, center=True).sum()
            elif (dv == 'CORRECTED_DATA') and ('WEIGHT_SPECTRUM' in vwcds):
                xda = (xds.CORRECTED_DATA * xds.WEIGHT_SPECTRUM).rolling(chan=width, min_periods=1, center=True).sum() / \
                      xds.WEIGHT_SPECTRUM.rolling(chan=width, min_periods=1, center=True).sum()
            else:
                xda = xda.rolling(chan=width, min_periods=1, center=True).mean()
            xda = xda.thin({'chan':width})
            xda = xda.astype(xds.data_vars[dv].dtype)  # make sure bool / int etc stay as such
        
        new_xds = new_xds.assign(dict([(dv,xda)]))
    
    return new_xds
