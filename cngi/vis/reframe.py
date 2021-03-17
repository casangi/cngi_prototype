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

###############################################
def reframe(mxds, vis, mode='channel', nchan=None, start=0, width=1, interpolation='linear', phasecenter=None, restfreq=None, outframe=None, veltype='radio'):
    """
    Transform channel labels and visibilities to a spectral reference frame which is appropriate for analysis, e.g. from TOPO to LSRK or to correct for doppler shifts throughout the time of observation

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        input multi-xarray Dataset with global data
    vis : str
        visibility partition in the mxds to use
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
        New output multi-xarray Dataset with global data
    """
    import xarray
    import datetime
    import numpy as np
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SpectralCoord, SkyCoord
    
    xds = mxds.attrs[vis]

    fields = xds.FIELD_ID.values.clip(0).flatten()
    sources = mxds.FIELD.sel(field_id=fields).source_id.values #[[xds.FIELD_ID.values.clip(0)]]
    unique_sources = np.unique(sources)
    
    #directions = mxds.SOURCE.DIRECTION.where(mxds.SOURCE.source_id
    targets = SkyCoord(directions[...,0], directions[...,1], unit='rad')
    
    #location = EarthLocation.of_site(input_xds['OBS_TELESCOPE_NAME']).get_itrs(obstime=Time(reference_time))
    #location = EarthLocation(input_xds['ANT_POSITION'].mean()).get_itrs(obstime=Time(reference_time))
    location = EarthLocation.of_site('ALMA')
    alma = location.get_itrs(obstime=Time(xds.time.values))
    
    time = _reference_time(global_xds)
    place = _reference_location(global_xds, reference_time)
    source = _target_location(global_xds, ddi)

    # epoch lookup or assume J2000
    #target_frame = 'FK5'

    aspc = SpectralCoord(input_array,
                         unit=u.Hz,
                         observer=place,
                         target=source)
    # doppler_reference=img_xds.attrs['spectral__reference'],
    # doppler_convention=img_xds.attrs['velocity__type'])


    output_xds = xarray.apply_ufunc(change_frame, place, source, vis_xds.DATA.chunk({'chan': -1}), input_core_dims=[['chan']],
                                    dask='parallelized', output_dtypes=[vis_xds.DATA.dtype])

    # update some properties of global_xds after conversion?

    # ouptut_xds.compute()
    return output_xds
