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


def auto_clip(vis_dataset, clip_min, clip_max):  # , storage_parms?):
    """
    Apply the clip flagging method. Data with values lower than clip_min
    or bigger than clip_max are flagged. Values are compared against the abs
    of the visibility values (no other expression supported at the moment).

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset.
    clip_min : float
        Minimum below which data should be flagged
    max_clip : float
        Maximum above which data should be flagged

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset with updated flags
    """
    flag_var = 'FLAG'
    data_var = 'DATA'

    to_clip = (abs(vis_dataset[data_var]) < clip_min) |\
              (abs(vis_dataset[data_var]) > clip_max)
    xds = vis_dataset.assign()
    xds[flag_var] = vis_dataset[flag_var] | to_clip

    # ? return _store(xds, list_xarray_data_variables, _storage_parms)
    return xds
