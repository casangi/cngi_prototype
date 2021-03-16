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

def region(xds, name='REGION1', ra=None, dec=None, pixels=None, pol=-1, channels=-1):
    """
    Create a new region Data variable in the Dataset
    
    .. note:: This function currently only supports rectangles and integer pixel boundaries
    
    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        input image dataset
    name : str
        dataset variable name for region, overwrites if already present
    ra : list
        right ascension coordinate range in the form of [min, max]. Default None means all
    dec : list
        declination coordinate range in the form of [min, max]. Default None means all
    pixels : array_like
        array of shape (N,2) containing pixel box. OR'd with ra/dec
    pol : int or list
        polarization dimension(s) to include in region.  Default of -1 means all
    channels : int or list
        channel dimension(s) to include in region.  Default of -1 means all
        
    Returns
    -------
    xarray.core.dataset.Dataset
        New Dataset
    """
    import numpy as np
    import xarray as xr
    
    # type checking/conversion
    if not name.strip(): name = 'REGIONX'
    if ra is None: ra=[0.0, 0.0]
    if dec is None: dec=[0.0, 0.0]
    if pixels is None: pixels = np.zeros((1,2), dtype=int)
    pixels = np.array(pixels, dtype=int)
    if (pixels.ndim != 2) or (pixels.shape[1] != 2):
      print('ERROR: pixels parameter not a (N,2) array')
      return None
    pol = np.array(np.atleast_1d(pol), dtype=int)
    if pol[0] == -1: pol = list(range(len(xds['pol'])))
    channels = np.array(np.atleast_1d(channels), dtype=int)
    if channels[0] == -1: channels = list(range(len(xds['chan'])))
    
    # TBD: allow arbitrary pixels, not just rectangles
    #ind_x = xr.DataArray(list(pixels[:,0]), dims=['d0'])
    #ind_y = xr.DataArray(list(pixels[:,1]), dims=['d1'])
    #region = xds.IMAGE[ind_x, ind_y]
    
    # TESTING only
    # ra = [2.88788, 2.88793]
    # dec = [-0.60573, -0.60568]
    # pixels = np.array([[20,40],[80,500]])
    
    # define region within ra/dec range
    region = xr.ones_like(xds.IMAGE,dtype=bool).where((xds.right_ascension > np.min(ra)) &
                                                      (xds.right_ascension < np.max(ra)) &
                                                      (xds.declination > np.min(dec)) & 
                                                      (xds.declination < np.max(dec)), False)
    
    # OR pixel values with ra/dec values
    #region = region | xr.ones_like(xds.IMAGE,dtype=bool).where(xds.d0.isin(pixels[:,0]) &
    #                                                           xds.d1.isin(pixels[:,1]), False)
    region = region | xr.ones_like(xds.IMAGE,dtype=bool).where((xds.l > xds.l[np.min(pixels[:,0]):np.max(pixels[:,0])+1].min()) &
                                                               (xds.l < xds.l[np.min(pixels[:,0]):np.max(pixels[:,0])+1].max()) &
                                                               (xds.m > xds.m[np.min(pixels[:,1]):np.max(pixels[:,1])+1].min()) &
                                                               (xds.m < xds.m[np.min(pixels[:,1]):np.max(pixels[:,1])+1].max()), False)
    
    # apply polarization and channels selections
    region = region.where(xds.pol.isin(xds.pol[pol]), False)
    region = region.where(xds.chan.isin(xds.chan[channels]), False)
    
    # assign region to a rest of image dataset
    xds = xds.assign(dict([(name,region)]))
    return xds
