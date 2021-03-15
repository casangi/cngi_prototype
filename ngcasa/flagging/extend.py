#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#  Copyright (C) 2021 European Southern Observatory, ALMA partnership
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


def extend(xds, **kwargs):  # args to be defined, storage_parms?):
    """
    .. todo::
        This function is not yet implemented

    Parameters
    ----------
    xds : xarray.core.dataset.Dataset
        Input dataset.
    TBD - additional parameters and functions for different methods:
          grow-around, extendflags, growtime, growfreq, antint, etc.

    Returns
    -------
    xarray.core.dataset.Dataset
        Visibility dataset with updated flags
    """
    raise NotImplementedError('This method is not implemented')
