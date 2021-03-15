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


def _ensure_flags_attr(xds):
    """
    Returns the name of the dataset attribute that holds info on flag variables.
    """
    flag_var = 'FLAG'
    flags_attr = 'flag_variables'
    if flags_attr not in xds.attrs:
        xds.attrs[flags_attr] = {flag_var: 'Default flags variable'}
    return flags_attr


def _add_descr(xds, add_name=None, descr=None):
    """
    Adds into the dict of flag variables a new flag variable and its description.
    """
    flags_attr = _ensure_flags_attr(xds)
    if add_name:
        xds.attrs[flags_attr][add_name] = descr
