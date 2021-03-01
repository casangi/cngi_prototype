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

###############################################
def reframe(mxds, vis, mode='channel', nchan=None, start=0, width=1, interpolation='linear', phasecenter=None, restfreq=None, outframe=None, veltype='radio'):
    """
    .. todo::
        This function is not yet implemented

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
    return {}

def _reference_time(input_xds):

  import datetime

  # the data variables expected by this function come from the global_xds

  try:
    reference_time = input_xds['ASDM_startValidTime'][0]
    reference_time = input_xds.OBS_TIME_RANGE.values[0]
  except:
    reference_time = datetime.datetime.now()
    reference_time = reference_time.strftime("%Y-%m-%dT%H:%M:%S")

  return reference_time


def _reference_location(input_xds, reference_time):

  from astropy.coordinates import EarthLocation
  from astropy.time import Time

  # the data variables expected by this function come from the global_xds

  try:
    # astropy.time.Time accepts string input of the form '2019-04-24T02:32:10'
    observer_location = EarthLocation.of_site(input_xds['OBS_TELESCOPE_NAME']).get_itrs(obstime=Time(reference_time))
  except:
    # use a crude reference position selection algorithm
    # could also try accessing the'ASDM_POSITION' data variable
    observer_location = EarthLocation(input_xds['ANT_POSITION'].mean()).get_itrs(obstime=Time(reference_time))

  return observer_location

def _target_location(input_xds):

  from astropy.coordinates import Angle, SkyCoord
  from astropy import units as u
  
  try:
    # assuming these direction axes are sensibly and consistently defined in radians
    # either way, this will be far easier for images
    radec_string = Angle(input_xds.SRC_DIRECTION * u.rad, input_xds.SRC_DIRECTION * u.rad)
  except:
    radec = '04h21m59.43s +19d32m06.4'

  try:
    # telescope dependent... is this is kept anywhere in the ASDM/MS/image formats?
    target_frame ='icrs'
  except:
    # epoch lookup or assume J2000
    target_frame = 'FK5'

  try:
    # another assumption lacking coordinate associated with the d2 dimension
    recession = global_xds.SRC_PROPER_MOTION * u.Hz
  except:
    # set a weird assumed default
    recession = 23.9 * u.km / u.s

  # examples...need to get these from somehwere in the dataset, likely global_xds
  try:
    distance = something
  except:
    distance = 144.321 * u.pc

  target_location =  SkyCoord(radec, frame=target_frame, radial_velocity=recession, distance=distance)

  return target_location

def _change_frame(input_array, place, source):
  from astropy import units as u
  from astropy.coordinates import EarthLocation, SkyCoord, SpectralCoord
  from astropy.time import Time
  import xarray
  import numpy as np

  # the input_array will act on the relevant xr.DataArray
  SpectralCoord(input_array, 
                unit=u(img_xds.attrs['rest_frequency'].split('')[:-1]),
                observer=place, 
                target=source,
                #doppler_reference=img_xds.attrs['spectral__reference'], 
                #doppler_convention=img_xds.attrs['velocity__type'])
                )

  output_array = input_array.with_observer_stationary_relative_to('lsrk')

  return output_array(axis=-1, keepdims=True)


def topo_to_lsrk(xds, ddi, reference_time, observer_location, target_location):
  import xarray
  from cngi.dio import read_vis

  vis_xds = cngi.dio.read_vis(xds, ddi=ddi)
  global_xds = cngi.dio.read_vis(xds, ddi='global')

  time = _reference_time(global_xds)
  place = _reference_location(global_xds, reference_time)
  source = _target_location(global_xds, ddi)

  output_xds = xarray.apply_ufunc(change_frame, place, source, vis_xds.DATA.chunk({'chan':-1}), input_core_dims=[['chan']], dask='parallelized', output_dtypes=[vis_xds.DATA.dtype])

  # update some properties of global_xds after conversion?

  # ouptut_xds.compute()
  return output_xds

