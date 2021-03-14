#   Copyright 2020-21 European Southern Observatory, ALMA partnership
#   Copyright 2020 AUI, Inc. Washington DC, USA
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
import xarray as xr
from ._flagging_utils._flag_attributes import _add_descr


def manager_add(vis_dataset, name, descr, source=None):
    """
    Add a new flag variable to the dataset. All flags in the new variable take
    the values from the source flag variable. If no source is found, the flags
    are all set to false.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset
    name : string
        The new flag variable name
    descr : string
        Text description of the flag variable (for example, 'backup_beginning')
    source : name
        Name of an existing flag variable. If specified its values will be used
        to initialize the new flag variable being added

    Returns:
    -------
    xds: xarray.core.dataset.Dataset
        Visibility dataset with updated set of flag variables
    """
    if not name.startswith('FLAG_'):
        raise ValueError('Current convention is that flag variables names should '
                         'start with FLAG_')
    if name in vis_dataset.variables:
        raise RuntimeError('Flag variable already in dataset: {}'.
                           format(name))

    # Add variable
    flag_var = 'FLAG'
    if source:
        if source not in vis_dataset.variables:
            raise RuntimeError('Source variable not found in dataset: {}'.
                               format(source))
        xds = vis_dataset.copy()
        xds[name] = vis_dataset[source]
    else:
        xds = vis_dataset.copy()
        xds[name] = xr.zeros_like(vis_dataset[flag_var], dtype=bool)

    _add_descr(xds, name, descr)

    return xds
