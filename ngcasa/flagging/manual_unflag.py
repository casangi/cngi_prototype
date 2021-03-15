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
import xarray as xr
from ._flagging_utils._parse_sel_cmd import _parse_sel_cmd


def manual_unflag(mxds, xds_idx, commands=None):  # , storage_parms?):
    """
    Unflags the selected data. Flags corresponding to the selections are unset.

    Parameters
    ----------
    mxds: xarray.core.dataset.Dataset
        Input Dataset
    xds_idx: int
        Index of the xarray datasets to get counts from (index in the xds'i'
        attributes of mxds). This is an oversimplification (early prototyping)
    commands : List[Dict]
        List of selections, each expressed as an xarray selection dictionary,
        using the same schema as in manual_flag. If empty, unflag all.
    TBD - Additional selection parameters / criteria

    Returns:
    -------
    xarray.core.dataset.Dataset
        Visibility dataset with updated (unset) flags
    """
    if not isinstance(commands, list):
        raise ValueError('Parameter selection must be a list of selection dicts')
    xds = mxds.attrs['xds' + '{}'.format(xds_idx)]
    return _unflag_with_reindex_like(mxds, xds, commands)


def _unflag_with_reindex_like(mxds, xds, cmds):
    flag_var = 'FLAG'
    ret_xds = xds.assign()
    if not cmds:
        ret_xds[flag_var] = xr.zeros_like(xds[flag_var], dtype=bool)

    for cmd in cmds:
        sel = _parse_sel_cmd(mxds, xds, cmd)
        fsel = xds[flag_var].sel(sel)
        if 0 in fsel.shape:
            print('WARNING: selection results in 0 shape. Sel: {}. Shape: {}'.
                  format(sel, fsel.shape))
            continue

        unflag_slice = xr.zeros_like(fsel, dtype=bool)
        reindexed_slice = unflag_slice.reindex_like(xds[flag_var],
                                                    fill_value=True)
        ret_xds[flag_var] &= reindexed_slice

    return ret_xds
