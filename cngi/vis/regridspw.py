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


###############################################
def regridspw(xds, field=None, spw=None, timerange=None, uvrange=None, antenna=None, scan=None, mode='channel', nchan=None, start=0, width=1,
              interpolation='linear', phasecenter=None, restfreq=None, outframe=None, veltype='radio'):
    """
    .. todo::
        This function is not yet implemented

    Transform channel labels and visibilities to a spectral reference frame which is appropriate for analysis, e.g. from TOPO to LSRK or to correct for doppler shifts throughout the time of observation

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Visibility Dataset
    field : int
        field selection. If None, use all fields
    spw : int
        spw selection. If None, use all spws
    timerange : int
        time selection. If None, use all times
    uvrange : int
        uvrange selection. If None, use all uvranges
    antenna : int
        antenna selection. If None, use all antennas
    scan : int
        scan selection. If None, use all scans
    mode : str
        regridding mode
    nchan : int
        number of channels in output spw. None=all
    start : int
        first input channel to use
    width : int
        number of input channels to average
    interpolation : str
        spectral interpolation method
    phasecenter : int
        image phase center position or field index
    restfreq : float
        rest frequency
    outframe : str
        output frame, None=keep input frame
    veltype : str
        velocity definition

    Returns
    -------
    xarray.core.dataset.Dataset
        New Visibility Dataset with updated data
    """
    return {}

