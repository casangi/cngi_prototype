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

def mask(xds, name='MASK1', ra=None, dec=None, pixels=None, pol=-1, channels=-1):
    """
    Create a new mask Data variable in the Dataset \n
    .. note:: This function currently only supports rectangles and integer pixel boundaries

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input Image Dataset
    name : str
        dataset variable name for mask, overwrites if already present
    ra : list
        right ascension coordinate range in the form of [min, max]. Default None means all
    dec : list
        declination coordinate range in the form of [min, max]. Default None means all
    pixels : numpy.ndarray
        array of shape (N,2) containing pixel box. AND'd with ra/dec
    pol : int or list
        polarization dimension(s) to include in mask.  Default of -1 means all
    channels : int or list
        channel dimension(s) to include in mask.  Default of -1 means all

    Returns
    -------
    xarray.core.dataset.Dataset
        output Image
    """
    import numpy as np
    import xarray as xr

    # type checking/conversion
    if not name.strip(): name = 'maskX'
    if ra is None: ra = [0.0, 0.0]
    if dec is None: dec = [0.0, 0.0]
    if pixels is None: pixels = np.zeros((1,2), dtype=int)
    pixels = np.array(pixels, dtype=int)
    if (pixels.ndim != 2) or (pixels.shape[1] != 2):
        print('ERROR: pixels parameter not a (N,2) array')
        return None
    pol = np.array(np.atleast_1d(pol), dtype=int)
    if pol[0] == -1: pol = [-1]
    channels = np.array(np.atleast_1d(channels), dtype=int)
    if channels[0] == -1: channels = [-1]

    # define mask within ra/dec range
    mask = xr.zeros_like(xds.IMAGE, dtype=bool).where((xds.right_ascension > np.min(ra)) &
                                                      (xds.right_ascension < np.max(ra)) &
                                                      (xds.declination > np.min(dec)) &
                                                      (xds.declination < np.max(dec)), True)

    # AND pixel values with ra/dec values
    mask = mask & xr.zeros_like(xds.IMAGE, dtype=bool).where((xds.l > xds.l[np.min(pixels[:,0]):np.max(pixels[:,0])+1].min()) &
                                                             (xds.l < xds.l[np.min(pixels[:,0]):np.max(pixels[:,0])+1].max()) &
                                                             (xds.m > xds.m[np.min(pixels[:,1]):np.max(pixels[:,1])+1].min()) &
                                                             (xds.m < xds.m[np.min(pixels[:,1]):np.max(pixels[:,1])+1].max()), True)

    # apply polarization and channels selections
    if pol[0] >= 0:
      mask = mask & xr.zeros_like(xds.IMAGE, dtype=bool).where(xds.pol.isin(xds.pol[pol]), True)
    if channels[0] >= 0:
      mask = mask & xr.zeros_like(xds.IMAGE, dtype=bool).where(xds.chan.isin(xds.chan[channels]), True)

    # assign region to a rest of image dataset
    xds = xds.assign(dict([(name, mask)]))

    return xds
