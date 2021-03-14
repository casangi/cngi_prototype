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
import pandas as pd
from ._flagging_utils._flag_attributes import _ensure_flags_attr


def manager_list(vis_dataset):
    """
    Add a new flag variable to the dataset. All flags in the new variable are
    set to false, unless a source variable is given in which case the values of
    the source are copied.

    Parameters
    ----------
    vis_dataset : xarray.core.dataset.Dataset
        Input dataset

    Returns:
    -------
    pandas.core.frame.DataFrame
        Information on flag variables from the input dataset
    """
    flags_attr = _ensure_flags_attr(vis_dataset)

    res = {'Flag variable name':
           [key for key in vis_dataset.attrs[flags_attr]],
           'Description':
           [val for _key, val in vis_dataset.attrs[flags_attr].items()]}
    return pd.DataFrame.from_dict(res)
