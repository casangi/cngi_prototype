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
