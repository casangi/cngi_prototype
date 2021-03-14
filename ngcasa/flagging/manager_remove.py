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
