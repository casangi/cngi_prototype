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
from ._flagging_utils._flag_attributes import _ensure_flags_attr


def manager_remove(vis_dataset, name):
    """
    Remove flag variable from the dataset.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset
    name : string
        The flag variable name to remove

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset without the removed flag variable
    """
    if name not in vis_dataset.variables:
        raise RuntimeError('Flag variable not found in dataset: {}'.
                           format(name))

    xds = vis_dataset.copy()
    flags_attr = _ensure_flags_attr(xds)
    del xds.attrs[flags_attr][name]

    return xds
